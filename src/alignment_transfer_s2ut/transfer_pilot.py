"""Read-only checkpoints and separate dev diagnostics for native transfer pilots."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from . import benchmark
from .alignment.calibration import gradient_diagnostics, gradient_norm
from .alignment.losses import lambda_at
from .alignment_calibration import logical_batches, logical_gradients, native_session, sample_ids, write_result
from .alignment_report import _diagnose
from .io import digest, file_hash, read_json, read_jsonl


def baseline_ready(backup):
    from direct_s2st.s2ut.paper_recipe import MAX_UPDATES

    required = {"checkpoints/checkpoint_last.pt", "checkpoints/checkpoint_best.pt",
                "research-metadata.json", "s2ut-recipe-lock.json"}
    for marker in Path(backup).glob("*/snapshot.json"):
        receipt = read_json(marker)
        if receipt.get("updates") == MAX_UPDATES and required <= receipt.get("files", {}).keys():
            return {"status": "READY_FOR_VERIFIED_RESTORE", "receipt": str(marker), "updates": MAX_UPDATES}
    raise ValueError("Transfer requires the Baseline's published 400,000-update snapshot. Wait for the existing run to finish.")


def baseline_evidence(record):
    metadata, config = record["metadata"], record["scientific_config"]
    return {"alignment_revision": metadata.get("alignment_revision"),
            "benchmark_runtime_revision": metadata.get("repository_revision"),
            "fairseq_revision": record["fairseq_revision"], "recipe_fingerprint": record["baseline_fingerprint"],
            "prepared_fingerprint": digest(record["prepared"]), "seed": config["common"]["seed"],
            "training_budget": record["training_budget"], "dataset": config["dataset"],
            "optimization": config["optimization"], "optimizer": config.get("optimizer"),
            "lr_scheduler": config.get("lr_scheduler"),
            "precision": {key: config["common"].get(key) for key in ("fp16", "memory_efficient_fp16", "bf16", "amp")}}


def checkpoint_for_probe(run):
    run = Path(run)
    plan = read_json(run / "native-run.json")
    if plan.get("contract_hash") != digest({key: value for key, value in plan.items() if key != "contract_hash"}):
        raise ValueError("corrupt native pilot contract")
    baseline = benchmark.read_record(plan["baseline_record"])
    checkpoint = run / "checkpoints/checkpoint_last.pt"
    saved = benchmark.load_native(checkpoint)
    updates = benchmark.checkpoint_updates(saved)
    if (baseline["record_hash"] != plan["baseline_record_hash"] or
            saved.get("extra_state", {}).get("alignment_contract_hash") != plan["contract_hash"] or
            benchmark.scientific_config(benchmark.plain_config(saved)) != plan["scientific_config"] or
            not 0 < updates <= plan["training_budget"] or plan["transfer"]["method"] == "baseline"):
        raise ValueError("pilot checkpoint differs from its registered native training contract")
    return plan, baseline, checkpoint, saved, updates


def gradient_probe(plan, saved, updates, calibration, directory):
    from fairseq import utils

    policy = plan["transfer"]
    method = policy["method"]
    kind = "mask" if method == "mask_distillation" else "kl"
    tau = policy["calibrated"]["tau_" + kind]
    coefficient = lambda_at(updates, method, policy["calibrated"]["lambda_" + kind],
                            plan["training_budget"], policy["anneal_fraction"])
    scale = float(saved["last_optimizer_state"].get("loss_scale", 1.))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid saved FP16 loss scale")
    with native_session(plan, directory) as (trainer, model, criterion, _):
        model.load_state_dict(saved["model"], strict=True)
        model.set_num_updates(updates)
        for name, task in trainer.task.multitask_tasks.items():
            criterion.set_multitask_loss_weight(name, task.args.get_loss_weight(updates))
        parameters = [p for block in model.decoder.layers for p in block.encoder_attn.parameters() if p.requires_grad]
        samples = next(logical_batches(trainer, 1))
        ids = sample_ids(trainer, samples)
        if ids != calibration["calibration_pair_ids"][0]:
            raise ValueError("pilot probe batch differs from the first calibration logical batch")
        utils.set_torch_seed(calibration["per_batch"][0]["virtual_seed"])
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        gradients, weights = logical_gradients(model, criterion, samples, parameters, {kind: tau},
                                               policy["mask_top_p"], loss_scale=scale)
        norms = {key: gradient_norm(value) for key, value in gradients.items()}
        metrics = gradient_diagnostics(gradients["task"], gradients[kind], norms["task"], norms[kind], coefficient)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if trainer.get_num_updates() != 0 or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("gradient probe performed an update or populated training gradients")
        return dict(metrics, pair_ids=ids, gradient_norms=norms, lambda_t=coefficient, tau=tau,
                    per_microbatch_layer_weights=weights[kind], gradient_seconds=time.perf_counter() - started,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    device=str(device), loss_scale=scale,
                    warnings=["negative task/alignment gradient cosine"] if (metrics["gradient_cosine"] or 0) < 0 else [])


def probe(run, requested_update, calibration_path, output):
    plan, baseline, checkpoint, saved, updates = checkpoint_for_probe(run)
    if requested_update != updates or updates >= plan["training_budget"]:
        raise ValueError(f"probe requires an incomplete checkpoint at update {requested_update}; latest is {updates}")
    calibration = read_json(calibration_path)
    if file_hash(calibration_path) != plan["transfer"]["calibration_hash"]:
        raise ValueError("pilot calibration differs from the checkpoint contract")
    policy = plan["transfer"]
    record = {**baseline, "method": policy["method"], "checkpoint": str(checkpoint),
              "checkpoint_hash": file_hash(checkpoint)}
    identity = {"step": updates, "contract_hash": plan["contract_hash"], "checkpoint_hash": record["checkpoint_hash"]}
    output = Path(output)
    if output.is_file():
        existing = read_json(output)
        if (any(existing.get(key) != value for key, value in identity.items()) or
                existing.get("artifact_hash") != digest({k: v for k, v in existing.items() if k != "artifact_hash"})):
            raise ValueError("existing pilot probe differs from the requested checkpoint")
        return existing
    result = gradient_probe(plan, saved, updates, calibration, output.parent / "gradient-model")
    del saved
    dev = _diagnose(plan, record, calibration_path, output.with_name("dev-alignment.json"), samples=32)
    logs = Path(run) / "native-dev.jsonl"
    full_dev = next((row["dev/main_loss"] for row in read_jsonl(logs) if row["step"] == updates), None) if logs.is_file() else None
    return write_result(output, dict(format="native-transfer-probe-v1", **identity, **result,
                                    method=policy["method"], status="PILOT_PAUSED", dev=dev,
                                    full_dev_unit_loss=full_dev))


def baseline_dev(baseline, manifest, units, teacher, calibration, output):
    record = benchmark.read_record(baseline)
    plan = benchmark.make_plan(baseline, Path(output).parent / "diagnostic-model", manifest=manifest,
                               units=units, teacher=teacher, calibration=calibration)
    return _diagnose(plan, record, calibration, output, samples=32)


def translate(output, manifest, units, *, run=None, baseline=None, vocoder=None, asr=None):
    from argparse import Namespace
    from .evaluation.metrics import evaluate
    from .inference.native import infer_native
    from .prepare_inference_input import prepare_inference_input

    if run:
        plan, record, checkpoint, state, updates = checkpoint_for_probe(run)
        if updates >= plan["training_budget"]:
            raise ValueError("use the completed-model evaluation workflow for a full checkpoint")
        if any(str(Path(path).resolve()) != plan["transfer"][key]
               for key, path in (("manifest", manifest), ("units", units))):
            raise ValueError("pilot dev inputs differ from the training contract")
        record.update(checkpoint=str(checkpoint), checkpoint_hash=file_hash(checkpoint))
        del state
    else:
        record = benchmark.read_record(baseline)
        updates = record["checkpoint_updates"]
    directory = Path(output).parent
    source, predictions = directory / "dev-source.jsonl", directory / "dev-predictions.jsonl"
    prepare_inference_input(manifest, "dev", source, resume=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer_native(Namespace(input=source, output=predictions, device=device, beam_size=10, max_len_a=1.,
                           max_len_b=200, length_penalty=1., units_only=not vocoder, vocoder_config=vocoder,
                           resume=True, overwrite=False), checkpoint_record=record)
    result = evaluate(Namespace(manifest=manifest, split="dev", predictions=predictions, units=units,
                                output=directory / "dev-metrics.json", asr_config=asr,
                                device=device, resume=True, overwrite=False))
    return write_result(output, {**result, "checkpoint_updates": updates, "purpose": "pilot dev only"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("baseline-ready").add_argument("--backup", required=True)
    sub.add_parser("baseline-evidence").add_argument("--baseline", required=True)
    child = sub.add_parser("probe")
    for option in ("run", "calibration", "output"):
        child.add_argument("--" + option, required=True)
    child.add_argument("--update", type=int, required=True)
    child = sub.add_parser("baseline-dev")
    for option in ("baseline", "manifest", "units", "teacher", "calibration", "output"):
        child.add_argument("--" + option, required=True)
    child = sub.add_parser("translate")
    source = child.add_mutually_exclusive_group(required=True)
    source.add_argument("--run")
    source.add_argument("--baseline")
    for option in ("manifest", "units", "output"):
        child.add_argument("--" + option, required=True)
    child.add_argument("--vocoder")
    child.add_argument("--asr")
    args = parser.parse_args(argv)
    if args.action == "baseline-ready":
        result = baseline_ready(args.backup)
    elif args.action == "baseline-evidence":
        result = baseline_evidence(benchmark.read_record(args.baseline))
    elif args.action == "probe":
        result = probe(args.run, args.update, args.calibration, args.output)
    elif args.action == "baseline-dev":
        result = baseline_dev(args.baseline, args.manifest, args.units, args.teacher, args.calibration, args.output)
    else:
        if args.asr and not args.vocoder:
            parser.error("--asr requires --vocoder")
        result = translate(args.output, args.manifest, args.units, run=args.run, baseline=args.baseline,
                           vocoder=args.vocoder, asr=args.asr)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
