"""Reconcile verified metric sources and the marked legacy W&B streams."""

import math
import re


LEGACY_TRAIN = "wandb_train_update"
LEGACY_FAIRSEQ = "wandb_layer_update"
DEV_UPDATE = "wandb_dev_update"
TRAIN_UPDATE = "wandb_train_update_v2"
TRAIN_WINDOW = "wandb_train_window_v2"
DEV_WINDOW = "wandb_dev_full_v2"
STREAMS = (LEGACY_TRAIN, LEGACY_FAIRSEQ, DEV_UPDATE, TRAIN_UPDATE, TRAIN_WINDOW, DEV_WINDOW)

TRAIN_WINDOW_FIELDS = {
    "main_loss_fairseq", "loss_fairseq", "nll_loss", "perplexity", "loss_scale",
    "clip_fraction", "oom", "gradient_norm",
}
DEV_WINDOW_FIELDS = {"loss", "main_loss_fairseq", "nll_loss", "ppl"}
MULTITASK_FIELDS = {f"multitask_{task}_{stat}"
                    for task in ("source_letter", "target_letter", "decoder_target_ctc")
                    for stat in ("loss", "loss_weight")}
TRAIN_WINDOW_FIELDS |= MULTITASK_FIELDS
DEV_WINDOW_FIELDS |= MULTITASK_FIELDS
# The first legacy Fairseq readers used fractional epoch "update". The
# optimizer-counter reader introduced these fields and kept the old marker.
FAIRSEQ_OPTIMIZER_FIELDS = ({"train/" + name for name in TRAIN_WINDOW_FIELDS - {"main_loss_fairseq"}}
                            | {"dev/" + name for name in DEV_WINDOW_FIELDS})
TRAIN_UPDATE_FIELDS = {"learning_rate", "seconds"}
ALIGNMENT_FIELDS = {
    "task_loss", "loss", "teacher_entropy", "student_entropy", "kl_mean", "kl_worst",
    "lambda", "tau", "top_p", "annealing_progress", "task_only_phase",
}
ALIGNMENT_LAYER = re.compile(
    r"(?:D|w|kl_layer|valid_rows|selected_frames|region_mass|coverage|worst_layer_fraction)_\d+"
)
CLIP_FRACTION = "train/window/clip_fraction"
DESTINATION_STREAMS = {("train", "update"): LEGACY_TRAIN, ("train", "window"): LEGACY_FAIRSEQ,
                       ("dev", "full"): LEGACY_FAIRSEQ, ("alignment", "update"): LEGACY_TRAIN,
                       ("alignment", "window"): LEGACY_FAIRSEQ}


def number(value):
    if type(value) not in (int, float, str):
        return None
    try:
        value = float(value) if isinstance(value, str) else value
        return value if math.isfinite(value) else None
    except (ValueError, OverflowError):
        return None


def legacy_destination(stream, key):
    prefix, _, name = key.partition("/")
    if stream == LEGACY_TRAIN and prefix == "train" and name in TRAIN_UPDATE_FIELDS:
        return TRAIN_UPDATE, "train/update/" + name
    if stream == LEGACY_FAIRSEQ and prefix == "train" and name in TRAIN_WINDOW_FIELDS:
        return TRAIN_WINDOW, "train/window/" + name
    if stream == LEGACY_FAIRSEQ and prefix == "dev" and name in DEV_WINDOW_FIELDS:
        return DEV_WINDOW, "dev/full/" + name
    if prefix == "alignment" and (name in ALIGNMENT_FIELDS or ALIGNMENT_LAYER.fullmatch(name)
                                  or name == "task_only_start" and stream == LEGACY_TRAIN):
        if stream == LEGACY_TRAIN:
            return TRAIN_UPDATE, "alignment/update/" + name
        if stream == LEGACY_FAIRSEQ:
            return TRAIN_WINDOW, "alignment/window/" + name
    return None


def legacy_points(row, update):
    # Old Reporter writes a stream marker and optimizer update in the same log call.
    markers = [stream for stream in STREAMS if row.get(stream) is not None]
    if len(markers) != 1 or markers[0] not in (LEGACY_TRAIN, LEGACY_FAIRSEQ):
        return
    stream = markers[0]
    if number(row[stream]) != update:
        return
    if stream == LEGACY_FAIRSEQ and not any(number(row[key]) is not None for key in FAIRSEQ_OPTIMIZER_FIELDS & row.keys()):
        return
    for key, raw in row.items():
        target = legacy_destination(stream, key)
        value = number(raw)
        if target is None or value is None:
            continue
        progress, name = target
        if progress == TRAIN_WINDOW and update != 1 and update % 100:
            continue
        if name == CLIP_FRACTION:
            if not 0 <= value <= 100:
                continue
            value /= 100
        if name == "alignment/update/task_only_start" and value < 0:
            continue
        yield progress, name, value


def is_legacy_destination(key):
    # During the single full scan a v2 point may appear before its legacy source.
    domain, _, tail = key.partition("/")
    scope, _, name = tail.partition("/")
    stream = DESTINATION_STREAMS.get((domain, scope))
    return stream is not None and legacy_destination(stream, domain + "/" + name) is not None


def same_metric(name, left, right):
    if left == right:
        return True
    # Conversion of a printed percentage to a fraction needs one division.
    # Admit one float64 ULP for that representation only.
    return (name == CLIP_FRACTION and left is not None and right is not None and 0 <= left <= 1 and 0 <= right <= 1
            and abs(left - right) <= max(math.ulp(left), math.ulp(right)))


def add_candidate(points, point, progress, value, source):
    previous = points.get(point)
    if previous is not None and (progress != previous[0] or not same_metric(point[1], value, previous[1])):
        raise ValueError(f"Conflicting source metric {point[1]} at update {point[0]} ({previous[2]} vs {source})")
    points.setdefault(point, (progress, value, source))


def summarize_points(points):
    summary = {}
    for update, name in points:
        stats = summary.setdefault(name, {"points": 0, "first_update": update, "last_update": update})
        stats["points"] += 1
        stats["first_update"] = min(stats["first_update"], update)
        stats["last_update"] = max(stats["last_update"], update)
    return summary


def reconcile_history(remote, sources, *, published=None, include_legacy=False):
    candidates, existing, disagreements = {}, {}, set()
    provenance = {}
    for source, rows in sources.items():
        points = set()
        for progress, update, values in rows:
            for name, value in values.items():
                point = (update, name)
                add_candidate(candidates, point, progress, value, source)
                points.add(point)
        provenance[source] = summarize_points(points)

    legacy = set()
    history = remote.scan_history(page_size=1000, use_cache=False) if candidates or include_legacy else ()
    for row in history:
        update = number(row.get("update"))
        if update is None or update < 1 or update != int(update) or published is not None and update > published:
            continue
        update = int(update)
        for name, raw in row.items():
            point = (update, name)
            if raw is None or not (point in candidates or include_legacy and is_legacy_destination(name)):
                continue
            value = number(raw)
            if not same_metric(name, existing.setdefault(point, value), value):
                disagreements.add(point)
        if include_legacy:
            for progress, name, value in legacy_points(row, update):
                point = (update, name)
                add_candidate(candidates, point, progress, value, "wandb_legacy_history")
                legacy.add(point)
    provenance["wandb_legacy_history"] = summarize_points(legacy)

    missing, missing_points = {}, set()
    for point, (progress, value, _) in sorted(candidates.items()):
        update, name = point
        if point in existing:
            if point in disagreements or not same_metric(name, existing[point], value):
                raise ValueError(f"W&B metric conflict for {name} at update {update}")
            continue
        missing.setdefault((progress, update), {})[name] = value
        missing_points.add(point)
    report = {"sources": provenance, "missing_metrics": summarize_points(missing_points),
              "missing_rows": len(missing), "matching_points": len(candidates) - len(missing_points)}
    return missing, report
