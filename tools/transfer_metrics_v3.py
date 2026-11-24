"""Frozen comparison contract and read-only coverage checks for transfer telemetry."""

from copy import deepcopy
from hashlib import sha256
import json
from alignment_transfer_s2ut.metric_integrity_v3 import (
    MAX_UPDATE, METHODS, TRAIN, DEV, check_raw, check_alignment, coverage,
    metric_values, required_metrics, valid_number, validate_row, plan_tau,
)


ENTITY = "yaaamashiro-national-institute-of-technology-okinawa-college"
PROJECT = "direct-s2st-v3"
GROUP = "experiment-001/transfer-fixed-v2"
SCOPE = "-spm-all-layers-run-transfer-fixed-v2"
OLD_SCOPE = "-spm-all-layers-run-transfer-fixed-v1"
MILESTONES = (1000, 5000, 10000, 25000, 50000, 100000, 200000, 400000)
SACREBLEU = "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1"
BLEU_POLICY = dict(version=1, aux_interval=1000, aux_samples=500, asr_samples=100,
                   asr_updates=list(MILESTONES), seed=1, beam=10, max_len_a=1, max_len_b=200,
                   max_aux_tokens=256, max_seconds=1800, asr_model="openai/whisper-large-v3-turbo",
                   asr_revision="60be3615a4d667e1258e8ad29130467587c489aa")
BLEU = tuple(f"eval/{kind}_{key}" for kind in ("aux", "asr") for key in ("bleu", "failures", "samples"))
SUMMARY = ("train/target_update", "train/max_update", "train/durable_update")
COMPARISON_KEYS = frozenset((*TRAIN, *DEV, *BLEU))
TRAIN_DIAGNOSTICS = frozenset("train/" + key for key in
                              ("unit_nll", "gradient_norm", "clip_applied", "loss_scale", "overflow_retries", "seconds"))
WINDOW_PREFIXES = ("train/window/", "alignment/window/", "dev/full/")


def enabled(args):
    return getattr(args, "metric_schema", 2) == 3


def scope(method, work, backup, suffix=SCOPE):
    return (method in METHODS and work.name == method and work.parent.name == "research-v3" + suffix
            and backup.name == "research-v3-" + method + suffix)


def verify_destination(args):
    v3_path = scope(args.method, args.work, args.backup)
    if not enabled(args):
        if args.project == PROJECT or v3_path:
            raise ValueError("Legacy telemetry requires a legacy project and training scope")
        return
    if ((args.entity, args.project, args.group) != (ENTITY, PROJECT, GROUP) or not v3_path
            or any(getattr(args, key, False) for key in ("include_legacy_history", "audit_history_only", "audit_metrics_only"))):
        raise ValueError("Schema v3 requires its registered experiment, project, entity and group")
    results = getattr(args, "bleu_results", None)
    if results is not None and (results.name != "transfer-fixed-v2" or results.parent.name != "transfer-bleu"):
        raise ValueError("Schema v3 requires the separate transfer-fixed-v2 BLEU results")


def namespace(contract, args):
    return json.dumps(["transfer", 3, args.entity, args.project, args.group, args.method, contract],
                      separators=(",", ":")) if enabled(args) else contract


def binding(contract, args):
    return {"metric_schema": 3, "condition": args.method, "alignment_contract_hash": contract,
            "transfer_metric_identity": sha256(namespace(contract, args).encode()).hexdigest()}


def scientific_plan(previous, current):
    """Only the save/log locations change between v1 and v2."""
    def comparison(plan):
        plan = deepcopy(plan)
        plan.pop("contract_hash")
        cfg = plan["config"]
        for section, keys in (("checkpoint", ("save_dir",)), ("common", ("log_file", "tensorboard_logdir")),
                              ("task", ("alignment_config",))):
            for key in keys:
                cfg[section].pop(key, None)
        return plan

    old, new = comparison(previous), comparison(current)
    if old != new:
        fields = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
        raise ValueError("New Transfer differs from the published experiment: " + ", ".join(fields))
    if (current["training_budget"] != MAX_UPDATE or not current["transfer"]["all_layers"]
            or current["initialization"] != "fresh native fairseq seed; no baseline weights loaded"):
        raise ValueError("Schema v3 requires the registered fresh, six-layer, 400k design")
    return {key: current[key] for key in ("contract_hash", "baseline_record_hash", "initialization", "training_budget")}


def segment_targets(latest, stop):
    limit = stop or MAX_UPDATE
    if type(limit) is not int or not latest <= limit <= MAX_UPDATE or limit < 1:
        raise ValueError("The requested pause must be within the registered budget and not behind its snapshot")
    milestones = [step for step in MILESTONES if latest <= step <= limit]
    if limit not in milestones and latest < limit:
        milestones.append(limit)
    return milestones


def check_bleu(row):
    from transfer_bleu import FAILURE_POLICY

    if row.get("policy") != BLEU_POLICY or row.get("failure_policy") != FAILURE_POLICY:
        raise ValueError("BLEU result has an incompatible policy")
    aux, asr = row["aux_pair_ids"], row["asr_pair_ids"]
    if len(aux) != 500 or len(set(aux)) != 500 or asr != aux[:100]:
        raise ValueError("BLEU populations differ from the fixed contract")
    for kind, count in (("aux", 500), ("asr", 100)):
        score, failures, samples = (row[f"{kind}_{field}"] for field in ("bleu", "failures", "samples"))
        records = row["generation_failures"][kind]
        ids = [failure["pair_id"] for failure in records]
        if (not valid_number(score) or not 0 <= score <= 100 or type(samples) is not int or samples != count
                or type(failures) is not int or len(set(ids)) != failures or len(ids) != failures
                or not set(ids) <= set(row[f"{kind}_pair_ids"])
                or any(not item["error"].startswith("GenerationFailure: ") for item in records)):
            raise ValueError(f"Incomplete or unqualified {kind} BLEU triplet")
    if any(row.get(f"{kind}_scoring", {}).get("signature") != SACREBLEU for kind in ("aux", "asr")):
        raise ValueError("BLEU signature differs from the registered scorer")


def remote_audit(target, sources, published):
    from wandb_history import number

    expected = {}
    for rows in sources.values():
        for progress, update, metrics in rows:
            for key, value in metrics.items():
                if expected.setdefault((update, key), (progress, value)) != (progress, value):
                    raise ValueError(f"Conflicting v3 source for {key} at update {update}")
    existing = {}
    errors = {"invalid": [], "duplicate": [], "conflict": [], "unexpected": []}
    history = target.scan_history(page_size=1000, use_cache=False) if target is not None else ()
    for row in history:
        values = {key: value for key, value in row.items() if value is not None and
                  key.startswith(("train/", "dev/", "alignment/", "eval/"))}
        update = number(row.get("update"))
        if update is None or update != int(update) or not 1 <= update <= published:
            if values:
                errors["unexpected"].append({"update": update, "metrics": sorted(values)})
            continue
        update = int(update)
        for key, value in values.items():
            point = (update, key)
            if point in existing:
                errors["duplicate"].append(point)
            existing[point] = value
            if not valid_number(value):
                errors["invalid"].append(point)
            if key not in COMPARISON_KEYS | TRAIN_DIAGNOSTICS and not key.startswith((*WINDOW_PREFIXES, "alignment/update/")):
                errors["unexpected"].append(point)
            if point in expected and expected[point][1] != value:
                errors["conflict"].append(point)
            if point not in expected and (key in COMPARISON_KEYS | TRAIN_DIAGNOSTICS or key.startswith("alignment/update/")):
                errors["unexpected"].append(point)
    missing, pending = {}, {}
    for point in expected.keys() - existing.keys():
        update, key = point
        progress = expected[point][0]
        watermark = (number(target.summary.get(progress)) or 0) if target is not None else 0
        target_map = pending if update <= watermark else missing
        target_map.setdefault(key, []).append(update)
    counts = {}
    comparison_counts = {}
    for update, key in expected:
        delivered = (update, key) in existing
        counts[key] = counts.get(key, 0) + delivered
        if update != 1 and key in COMPARISON_KEYS:
            comparison_counts[key] = comparison_counts.get(key, 0) + delivered
    status = ("CONFLICT" if any(errors.values()) else "ACK_PENDING" if pending
              else "NOT_SENT" if missing else "VERIFIED")
    return {"status": status,
            "metric_point_counts": counts,
            "comparison_point_counts": comparison_counts,
            "missing": {key: sorted(updates) for key, updates in sorted(missing.items())},
            "ack_pending": {key: sorted(updates) for key, updates in sorted(pending.items())},
            "errors": {key: {"count": len(points), "examples": points[:20]} for key, points in errors.items()}}


def reconcile_delivery(target, sources, published):
    from wandb_writer import WriterBusy

    materialized = {source: list(rows) for source, rows in sources.items()}
    audit = remote_audit(target, materialized, published)
    if audit["status"] == "CONFLICT":
        raise ValueError("W&B Schema v3 history conflict: " + json.dumps(audit["errors"], sort_keys=True))
    if audit["status"] == "ACK_PENDING":
        raise WriterBusy("ACK_PENDING: W&B has not indexed previously acknowledged Schema v3 history")
    missing = {key: set(updates) for key, updates in audit["missing"].items()}
    rows = {}
    for points in materialized.values():
        for progress, update, values in points:
            subset = {key: value for key, value in values.items() if key in missing and update in missing[key]}
            if subset:
                rows.setdefault((progress, update), {}).update(subset)
    return dict(sorted(rows.items(), key=lambda item: (item[0][1], item[0][0]))), audit
