"""Model-independent validation of original Schema v3 measurements."""

import json
import math
from pathlib import Path


MAX_UPDATE = 400000
DEV_SAMPLES = 1453
TRAIN_JSONL = "alignment-train.jsonl"
DEV_JSONL = "native-dev.jsonl"
METHODS = ("distillation", "annealed_distillation", "mask_distillation")
AUXILIARIES = ("source_letter", "target_letter", "decoder_target_ctc")
TRAIN = ("train/main_loss", "train/task_loss", "train/total_loss", "train/learning_rate",
         *(f"train/aux/{name}_loss" for name in AUXILIARIES))
DEV = ("dev/main_loss", "dev/total_loss", *(f"dev/aux/{name}_loss" for name in AUXILIARIES))
ALIGNMENT = ("loss", "weighted_loss", "kl_mean", "kl_worst", "teacher_entropy", "student_entropy", "worst_layer",
             "lambda", "tau", *(f"{key}_{layer}" for key in ("D", "w", "kl_layer", "valid_rows", "worst_layer_fraction")
                                for layer in range(6)))
MASK = ("top_p", *(f"{key}_{layer}" for key in ("region_mass", "coverage", "selected_frames") for layer in range(6)))
ANNEAL = ("annealing_progress", "task_only_phase")


def required_metrics(method, update):
    keys = [*ALIGNMENT]
    if method == "mask_distillation":
        keys += MASK
    if method == "annealed_distillation":
        keys += ANNEAL
        # The criterion uses the pre-update counter and continues recording diagnostics.
        if update > 320000:
            keys.append("task_only_start")
    return TRAIN + tuple("alignment/update/" + name for name in keys)


def plan_tau(plan):
    transfer = plan["transfer"]
    key = "tau_mask" if transfer["method"] == "mask_distillation" else "tau_kl"
    tau = transfer["calibrated"][key]
    if not valid_number(tau) or tau <= 0 or tau != transfer["coefficient_policy"]["tau"]:
        raise ValueError("Plan tau does not match its registered Calibration")
    return tau


def check_alignment(values, method, update, tau):
    def value(name):
        return values["alignment/update/" + name]

    def layers(name):
        return [value(f"{name}_{layer}") for layer in range(6)]

    tolerance = 2e-6
    measured_tau = value("tau")
    if measured_tau <= 0 or not math.isclose(measured_tau, tau, rel_tol=tolerance):
        raise ValueError("Alignment tau differs from the registered Calibration")
    counts = layers("valid_rows")
    eligible = [layer for layer, count in enumerate(counts) if count > 0]
    if not eligible or any(count < 0 or count != int(count) for count in counts):
        raise ValueError("Invalid Alignment valid_rows counts")
    for name in ("w", "worst_layer_fraction"):
        weights = layers(name)
        if (any(not -tolerance <= weight <= 1 + tolerance for weight in weights)
                or any(abs(weights[i]) > tolerance for i in range(6) if not counts[i])
                or not math.isclose(math.fsum(weights), 1.0, abs_tol=2 * tolerance)):
            raise ValueError(f"Invalid Alignment {name} weights")
    losses = layers("D")
    worst = value("worst_layer")
    largest = max(eligible, key=losses.__getitem__)
    if (worst not in eligible or not math.isclose(losses[int(worst)], losses[largest],
            rel_tol=tolerance, abs_tol=tolerance)
            or losses[int(worst)] == losses[largest] and worst != largest):
        raise ValueError("Alignment worst_layer disagrees with eligible layer losses")
    if not math.isclose(value("weighted_loss"), value("lambda") * value("loss"),
                        rel_tol=4e-6, abs_tol=2e-7):
        raise ValueError("Alignment weighted_loss disagrees with lambda × loss")
    start = 10.0 if method == "mask_distillation" else 1.0
    phase = method == "annealed_distillation" and update > 320000
    if method == "annealed_distillation":
        fraction = min(1.0, (update - 1) / 320000)
        expected = start * (1.0 + math.cos(math.pi * fraction)) / 2
        if (values["alignment/update/task_only_phase"] != int(phase)
                or not math.isclose(values["alignment/update/annealing_progress"], fraction, abs_tol=1e-7)
                or phase and (values["alignment/update/task_only_start"] != 320000
                              or values["alignment/update/weighted_loss"] != 0)):
            raise ValueError("Annealed KL metric does not match the applied schedule")
    else:
        expected = start
    if not math.isclose(values["alignment/update/lambda"], expected, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Measured alignment coefficient differs from the registered schedule")
    if method == "mask_distillation":
        if not math.isclose(value("top_p"), 0.8, abs_tol=1e-7):
            raise ValueError("Measured Mask Top-p differs from the registered value")
        for name in ("region_mass", "coverage"):
            if any(not -tolerance <= measurement <= 1 + tolerance for measurement in layers(name)):
                raise ValueError(f"Invalid Mask {name} probabilities")
        frames = layers("selected_frames")
        if any(frame < 0 or frame != int(frame) or not counts[i] and frame for i, frame in enumerate(frames)):
            raise ValueError("Invalid Mask selected_frames counts")


def metric_values(row, split):
    if row.get("metric_schema") != 3:
        raise ValueError("Expected an original Schema v3 measurement")
    result = {}
    for key, value in row.items():
        selected = key.startswith(("train/", "alignment/")) if split == "train" else key in DEV
        if selected:
            # Native logical diagnostics are unscoped in JSONL.
            result["alignment/update/" + key.removeprefix("alignment/") if key.startswith("alignment/") else key] = value
    return result


def valid_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def check_raw(row, split):
    if row.get("measurement_error"):
        raise ValueError(str(row["measurement_error"]))
    raw = row.get("loss_sums")
    if not isinstance(raw, dict):
        raise ValueError("missing raw loss sums")

    def terms(name):
        numerator, count = raw[name]
        if (not valid_number(numerator) or type(count) is not int or count <= 0
                or name != "weighted_alignment" and numerator < 0):
            raise ValueError(f"invalid raw measurement {name}")
        value = row.get(f"{split}/{name}")
        if value is not None and value != numerator / count:
            raise ValueError(f"normalized value differs from raw sums for {name}")
        return numerator, count

    ce, size = terms("main_loss")
    task, task_size = terms("task_loss")
    total, total_size = terms("total_loss")
    terms("unit_nll")
    samples, batches = terms("samples")
    if samples != row.get("sample_count") or samples < batches or samples != int(samples):
        raise ValueError("invalid criterion sample count")
    if split == "dev" and samples != DEV_SAMPLES:
        raise ValueError(f"full Dev expected {DEV_SAMPLES} samples, measured {samples}")
    weighted = (ce + math.fsum(weight * terms(f"aux/{task_name}_loss")[0]
                               for task_name, weight in zip(AUXILIARIES, (8, 8, 1.6), strict=True)))
    alignment, alignment_size = terms("weighted_alignment") if split == "train" else (0, size)
    if (task_size != size or total_size != size or alignment_size != size
            or not math.isclose(task, weighted, rel_tol=1e-6, abs_tol=1e-7)
            or not math.isclose(total, task + alignment, rel_tol=1e-6, abs_tol=1e-7)):
        raise ValueError("inconsistent task or total objective sums")
    if split == "train" and (row.get("alignment/weighted_loss") != alignment / size
            or row.get("alignment/task_loss") != task / size):
        raise ValueError("alignment objective diagnostics differ from the captured raw sums")


def validate_row(row, method, contract, split, *, tau):
    if row.get("alignment_contract_hash") != contract:
        raise ValueError("foreign metric row")
    values = metric_values(row, split)
    check_raw(row, split)
    needed = required_metrics(method, row["step"]) if split == "train" else DEV
    missing = [key for key in needed if key not in values]
    invalid = [key for key, value in values.items() if not valid_number(value)]
    if missing or invalid:
        raise ValueError(f"missing required metrics {missing}; nonfinite or invalid metrics {invalid}")
    if split == "train":
        if values["train/learning_rate"] < 0:
            raise ValueError("negative measured Learning Rate")
        check_alignment(values, method, row["step"], tau)
    return values


def coverage(path, maximum, method, contract, split, *, tau, allow_unpublished=False):
    required = (range(1, maximum + 1) if split == "train" else range(1000, maximum + 1, 1000))
    expected = set(required)
    seen, bad, rows, previous = set(), [], {}, 0
    if Path(path).is_file():
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                update = None
                try:
                    row = json.loads(line)
                    update = row["step"]
                    if type(update) is not int or update < 1:
                        raise ValueError("invalid optimizer update")
                    if update > maximum:
                        if not allow_unpublished or update > MAX_UPDATE:
                            raise ValueError("update exceeds the verified checkpoint or registered budget")
                        continue
                    if not line.endswith("\n") or update not in expected or update <= previous:
                        raise ValueError(f"unexpected or duplicated {split} update {update}")
                    seen.add(update)
                    previous = update
                    values = validate_row(row, method, contract, split, tau=tau)
                    if split == "dev" or update == 1 or update % 100 == 0:
                        rows[update] = values
                except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                    if len(bad) < 20:
                        bad.append(f"{split} update {update}: {error}")
    absent = sorted(expected - seen)
    comparison = TRAIN if split == "train" else DEV
    return rows, {"expected_rows": len(expected), "recorded_rows": len(seen), "missing_rows": len(absent),
                  "missing_examples": absent[:20], "invalid_examples": bad, "complete": not absent and not bad,
                  "comparison_point_counts": {key: sum(key in values for update, values in rows.items() if update != 1)
                                              for key in comparison},
                  "additional_diagnostic_updates": [1] if 1 in rows else []}


def checkpoint_originals(folder, update, plan, *, tail_only=False):
    """Fail before resume, or check the boundary of an immutable freeze of checked rows."""
    method, contract, tau = plan["transfer"]["method"], plan["contract_hash"], plan_tau(plan)
    for split, name, point in (("train", TRAIN_JSONL, update), ("dev", DEV_JSONL, update // 1000 * 1000)):
        path = folder / name
        if not tail_only:
            _, report = coverage(path, update, method, contract, split, tau=tau)
            if not report["complete"]:
                raise ValueError(f"Incomplete Schema v3 {split} originals through update {update}: "
                                 f"{report['missing_examples']} {report['invalid_examples']}")
        elif point or path.exists():
            # Each row and the restored prefix have already been validated. The
            # bounded read also catches failure-path freezes with rows ahead of last.pt.
            with path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                data = stream.read()
            if not point or not data.endswith(b"\n"):
                raise ValueError(f"Invalid frozen Schema v3 {split} originals for update {update}")
            row = json.loads(data.splitlines()[-1])
            if row.get("step") != point:
                raise ValueError(f"Frozen Schema v3 {split} ends at {row.get('step')}, expected {point}")
            validate_row(row, method, contract, split, tau=tau)
