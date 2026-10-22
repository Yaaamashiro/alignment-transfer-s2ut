"""Prepare and record a strong Baseline / Monotonic / T2TT pilot."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from .data.manifest import load_manifest, manifest_hash, prepare_manifest, validate
from .io import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl

CONDITIONS = (("baseline", "baseline", "t2tt"),
              ("annealed_monotonic", "annealed_guidance", "monotonic"),
              ("annealed_t2tt", "annealed_guidance", "t2tt"))


def prepare(source, output, corpus_root=None, train_limit=10000, dev_limit=500, seed=1, resume=False):
    """Select original train/dev pairs before audio hashing; never re-split test."""
    if train_limit < 1 or dev_limit < 1:
        raise ValueError("positive train/dev limits required")
    source, root = Path(source).resolve(), Path(output).resolve()
    if source == root / "manifest.jsonl" or (corpus_root and root.is_relative_to(Path(corpus_root).resolve())):
        raise ValueError("pilot output must be separate from the source corpus/manifest")
    original = read_jsonl(source)
    checked = []
    for row in original:
        value = dict(row)
        for language in ("ja", "en"):
            raw = (row.get(language + "_audio_corpus_relative") or row.get(language + "_audio") or
                   row.get(language + "_wav_16k")) if corpus_root else row.get(language + "_audio")
            if not raw:
                raise ValueError("missing audio path for " + str(row.get("pair_id")))
            value[language + "_audio"] = str(((Path(corpus_root) if corpus_root else source.parent) / raw).resolve())
        checked.append(value)
    # Check global IDs, text and known audio-pair leakage even for excluded rows.
    validate(checked, check_audio=False)
    selected = []
    for split, limit in (("train", train_limit), ("dev", dev_limit)):
        population = [i for i, row in enumerate(original) if row["split"] == split]
        if not population:
            raise ValueError("pilot requires original train and dev splits")
        rng = random.Random(digest([seed, split]))
        selected.extend(rng.sample(population, min(limit, len(population))))
    selected.sort()
    contract = {"source": str(source), "source_sha256": file_hash(source),
                "corpus_root": str(Path(corpus_root).resolve()) if corpus_root else None,
                "train_limit": train_limit, "dev_limit": dev_limit, "seed": seed,
                "pair_ids": [original[i]["pair_id"] for i in selected], "selection_version": 1}
    metadata = root / "selection.json"
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("pilot data exists; use --resume or a new output directory")
    if resume and metadata.is_file() and read_json(metadata)["selection_contract"] != contract:
        raise ValueError("pilot selection changed; use a new experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    # Canonical relative paths refer to the original manifest, not this staging file.
    subset = [dict(original[i] if corpus_root else checked[i]) for i in selected]
    write_jsonl(root / "selected-input.jsonl", subset, overwrite=resume)
    rows = prepare_manifest(root / "selected-input.jsonl", root / "manifest.jsonl",
                            corpus_root=corpus_root, resume=resume)
    result = {"selection_contract": contract, "manifest_hash": manifest_hash(rows),
              "population_counts": dict(Counter(row["split"] for row in original)),
              "selected_counts": dict(Counter(row["split"] for row in rows)),
              "test_used": False}
    write_json(metadata, result, overwrite=resume)
    return result


def check_runtime():
    import torch
    import torchaudio
    from .s2ut.fairseq_extension import FAIRSEQ_REVISION, verify_fairseq
    verify_fairseq()
    if not torch.cuda.is_available():
        raise RuntimeError("pilot execution requires CUDA PyTorch and a GPU runtime")
    if torch.__version__.split("+")[0] != "2.7.1" or torchaudio.__version__.split("+")[0] != "2.7.1":
        raise RuntimeError("use the pinned torch/torchaudio 2.7.1 environment")
    return {"python": sys.version, "torch": str(torch.__version__), "torchaudio": str(torchaudio.__version__),
            "cuda": torch.version.cuda, "fairseq_revision": FAIRSEQ_REVISION,
            "gpus": [{"name": torch.cuda.get_device_name(i),
                       "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                      for i in range(torch.cuda.device_count())]}


def commands(args):
    if not 1 <= args.anneal_steps < args.max_updates:
        raise ValueError("pilot requires 1 <= anneal_steps < max_updates for a teacher-free tail")
    if min(args.batch_size, args.update_frequency, args.max_tokens, args.interval) < 1 or args.warmup_updates < 0:
        raise ValueError("invalid pilot batch, interval or warmup")
    shared = [sys.executable, "-m", "alignment_transfer_s2ut.train_s2ut", "--manifest", str(args.manifest),
              "--config", str(args.config), "--units", str(args.units), "--recipe", "strong",
              "--max-updates", str(args.max_updates), "--anneal-steps", str(args.anneal_steps),
              "--seed", str(args.seed), "--batch-size", str(args.batch_size),
              "--update-frequency", str(args.update_frequency), "--max-tokens", str(args.max_tokens),
              "--warmup-updates", str(args.warmup_updates), "--validation-interval", str(args.interval),
              "--save-interval-updates", str(args.interval), "--prior-smoothing", "0.05",
              "--source-mapping", "frontend", "--patience", "0", "--device", "cuda", "--resume-if-exists"]
    result = {}
    for name, method, prior in CONDITIONS:
        command = shared + ["--method", method, "--prior-type", prior, "--output", str(Path(args.output) / name)]
        if name == "annealed_t2tt":
            command += ["--teacher", str(args.teacher)]
        result[name] = command
    return result


def run_logged(name, command, logs, monitor_gpu=False, env=None):
    """Fail on command errors, persist every attempt, and stop our GPU sampler."""
    root = Path(logs)
    root.mkdir(parents=True, exist_ok=True)
    key = digest(name)[:12] + "-" + str(time.time_ns())
    output, gpu = root / (key + ".log"), root / (key + "-gpu.csv")
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    monitor, gpu_stream, process = None, None, None
    gpu_error, code, failure = None, None, None
    try:
        if monitor_gpu:
            gpu_stream = gpu.open("w", encoding="utf-8")
            try:
                monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,nounits", "--loop=5"], stdout=gpu_stream, stderr=subprocess.DEVNULL)
            except OSError as error:
                gpu_error = str(error)
        with output.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
            last_echo, last_line, last_printed = 0.0, "", ""
            for line in process.stdout:
                stream.write(line)
                last_line = line
                # Keep Colab output responsive without retaining every training update.
                if time.monotonic() - last_echo >= 5:
                    print(line, end="", flush=True)
                    last_printed = line
                    stream.flush()
                    last_echo = time.monotonic()
            code = process.wait()
            if last_line != last_printed:
                print(last_line, end="", flush=True)
    except BaseException as error:
        failure = type(error).__name__ + ": " + str(error)
        raise
    finally:
        monitor_exit = monitor.poll() if monitor is not None else None
        for child in (process, monitor):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        if gpu_stream:
            gpu_stream.close()
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if monitor_exit is not None and monitor_exit != 0:
            gpu_error = "nvidia-smi sampler exited: " + str(monitor_exit)
        record = {"name": name, "command": list(map(str, command)), "started_utc": started,
                  "host_utc_offset_seconds": int(datetime.now().astimezone().utcoffset().total_seconds()),
                  "elapsed_seconds": time.perf_counter() - clock, "returncode": code,
                  "status": "succeeded" if code == 0 else "failed_or_interrupted",
                  "exception": failure,
                  "log": str(output.resolve()), "gpu_csv": str(gpu.resolve()) if monitor is not None else None,
                  "gpu_monitor_error": gpu_error}
        write_json(root / (key + ".json"), record)
    if code:
        raise subprocess.CalledProcessError(code, command)
    return record


def report(runs, logs, output):
    from .compare_experiments import compare_experiments
    roots = [Path(runs) / name for name, _, _ in CONDITIONS]
    comparison = compare_experiments(roots)
    if not comparison["fair_comparison"]:
        raise ValueError("pilot conditions differ: " + json.dumps(comparison["differences"]))
    attempts = [read_json(path) for path in Path(logs).glob("*.json")]
    records, evaluation_contracts, curves = [], [], []
    for root, (name, method, prior) in zip(roots, CONDITIONS):
        experiment = read_json(root / "experiment.json")
        if experiment["recipe"]["speech_recipe"]["name"] != "strong":
            raise ValueError("pilot report requires the strong recipe")
        if experiment["method"] != method or (name != "baseline" and experiment["transfer"]["prior_type"] != prior):
            raise ValueError("unexpected pilot condition: " + name)
        # A restart can repeat metrics after the most recent saved checkpoint.
        metrics = {row["step"]: row for row in read_jsonl(root / "metrics.jsonl")}
        if not metrics or any(not math.isfinite(row["train/main_loss"]) for row in metrics.values()):
            raise ValueError("missing/non-finite pilot metrics: " + name)
        dev = [row for row in metrics.values() if row.get("dev/main_loss") is not None]
        if not dev or any(not math.isfinite(row["dev/main_loss"]) for row in dev):
            raise ValueError("missing/non-finite dev validation: " + name)
        best = min(dev, key=lambda row: row["dev/main_loss"])
        curves.append((name, sorted(dev, key=lambda row: row["step"])))
        condition_attempts = [a for a in attempts if a["name"] == name]
        record = {"condition": name, "step": max(metrics), "planned_updates": experiment["planned_max_updates"],
                  "best_dev_main_loss": best["dev/main_loss"], "best_dev_step": best["step"],
                  "final_beta": metrics[max(metrics)]["guidance/beta"],
                  "checkpoint_best": str((root / "checkpoint_best.pt").resolve()),
                  "recorded_seconds": sum(a["elapsed_seconds"] for a in condition_attempts),
                  "attempts": len(condition_attempts), "failed_attempts": sum(a["status"] != "succeeded" for a in condition_attempts)}
        evaluation = root / "dev-metrics.json"
        if evaluation.is_file():
            values = read_json(evaluation)
            if values["split"] != "dev":
                raise ValueError("pilot evaluation must use dev")
            evaluation_contracts.append({key: value for key, value in values["evaluation_contract"].items()
                                         if key != "predictions_hash"} | {"backend": values["backend"],
                                                                           "vocoder_hash": values["vocoder_hash"]})
            record["dev_evaluation"] = {key: values[key] for key in ("samples", "unit_error_rate", "asr_bleu") if key in values}
        records.append(record)
    if evaluation_contracts and (len(evaluation_contracts) != 3 or any(
            value != evaluation_contracts[0] for value in evaluation_contracts[1:])):
        raise ValueError("evaluate all three conditions on the same dev data/config")
    result = {"fair_comparison": comparison, "conditions": records,
              "complete": all(row["step"] >= row["planned_updates"] for row in records),
              "scope": "Pilot diagnostics on train/dev; not final test results."}
    write_json(output, result, overwrite=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .io import atomic_write
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, values in curves:
        axis.plot([row["step"] for row in values], [row["dev/main_loss"] for row in values], marker=".", label=name)
    axis.set(xlabel="Optimizer updates", ylabel="Teacher-free dev main loss", title="Strong S2UT pilot (dev)")
    axis.legend()
    axis.grid(alpha=.2)
    atomic_write(Path(output).with_suffix(".png"), lambda stream: figure.savefig(stream, format="png", dpi=150), overwrite=True)
    plt.close(figure)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--input", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--corpus-root")
    prepare_parser.add_argument("--train-limit", type=int, default=10000)
    prepare_parser.add_argument("--dev-limit", type=int, default=500)
    prepare_parser.add_argument("--seed", type=int, default=1)
    prepare_parser.add_argument("--resume", action="store_true")
    sub.add_parser("runtime")
    child = sub.add_parser("run")
    child.add_argument("--name", required=True)
    child.add_argument("--logs", required=True)
    child.add_argument("--monitor-gpu", action="store_true")
    child.add_argument("command", nargs=argparse.REMAINDER)
    for action in ("plan", "train"):
        child = sub.add_parser(action)
        for field in ("manifest", "config", "units", "teacher", "output", "logs"):
            child.add_argument("--" + field, required=True)
        for field, default in (("max-updates", 10000), ("anneal-steps", 5000), ("seed", 1), ("batch-size", 8),
                               ("update-frequency", 4), ("max-tokens", 20000), ("warmup-updates", 1000), ("interval", 500)):
            child.add_argument("--" + field, type=int, default=default)
    child = sub.add_parser("report")
    for field in ("runs", "logs", "output"):
        child.add_argument("--" + field, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.input, args.output, args.corpus_root, args.train_limit, args.dev_limit, args.seed, args.resume)
    elif args.action == "runtime":
        result = check_runtime()
    elif args.action == "report":
        result = report(args.runs, args.logs, args.output)
    elif args.action == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("run requires a command after --")
        result = run_logged(args.name, command, args.logs, args.monitor_gpu)
    else:
        result = commands(args)
        if args.action == "train":
            check_runtime()
            rows = load_manifest(args.manifest)
            if {r["split"] for r in rows} != {"train", "dev"}:
                raise ValueError("pilot manifest must contain train/dev only")
            for name, command in result.items():
                run_logged(name, command, args.logs, monitor_gpu=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
