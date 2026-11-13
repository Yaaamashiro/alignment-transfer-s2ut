"""Verify saved Baseline provenance and report a short native transfer recovery check."""

import argparse
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

import torch

from alignment_transfer_s2ut import benchmark
from alignment_transfer_s2ut.colab_runtime import BENCHMARK_REVISION, BENCHMARK_RUNTIME_REVISION
from alignment_transfer_s2ut.io import digest, file_hash, read_json, read_jsonl
from alignment_transfer_s2ut.alignment_calibration import write_result
from alignment_transfer_s2ut.transfer_pilot import baseline_evidence


SHA = re.compile(r"[0-9a-f]{40}")
BASELINE_MEMBERS = ("checkpoints/checkpoint_last.pt", "checkpoints/checkpoint_best.pt",
                    "research-metadata.json", "s2ut-recipe-lock.json")
RECOVERY_UPDATES = {"save": 2, "resume": 4}


def option(command, name):
    if name not in command:
        return None
    index = command.index(name) + 1
    return command[index] if index < len(command) else None


def checkout_evidence(logs, work, backup):
    """Correlate the checkout and Baseline launch receipts in an exported execution log."""
    events = []
    for path in Path(logs).glob("*.json"):
        entry = read_json(path)
        if entry.get("name") in {"checkout-alignment", "baseline"} and "started_utc" in entry:
            events.append((datetime.fromisoformat(entry["started_utc"]), path, entry))
    checkout = None
    evidence = []
    for started, path, entry in sorted(events):
        command = entry.get("command", [])
        if entry["name"] == "checkout-alignment":
            checkout = None
            if entry.get("status") == "succeeded" and entry.get("returncode") == 0 and "--detach" in command:
                revision = option(command, "--detach")
                if revision and SHA.fullmatch(revision):
                    checkout = (revision, path, option(command, "-C"),
                                started + timedelta(seconds=entry.get("elapsed_seconds", 0)))
        elif ("alignment_transfer_s2ut.corpus_runtime" in command and "baseline" in command and "--train" in command
              and option(command, "--work") == work and option(command, "--backup") == backup
              and entry.get("pid") is not None and checkout):
            revision, checkout_path, checkout_repo, finished = checkout
            if checkout_repo != entry.get("cwd") or finished > started:
                continue
            evidence.append({"revision": revision, "checkout_receipt_sha256": file_hash(checkout_path),
                             "launch_receipt_sha256": file_hash(path)})
    if not evidence or len({entry["revision"] for entry in evidence}) != 1:
        raise ValueError("execution logs must identify one successful checkout for the matching Baseline launch")
    return evidence


def verify_adapter(repo, revision, expected):
    """Use Git's source bytes and the compatibility rules at the candidate revision."""
    if not SHA.fullmatch(revision):
        raise ValueError("Baseline revision must be a full commit SHA")
    source = "src/alignment_transfer_s2ut/"
    def git_file(name):
        try:
            return subprocess.check_output(["git", "-C", str(repo), "show", revision + ":" + source + name])
        except subprocess.CalledProcessError as error:
            raise ValueError("Baseline source is unavailable; fetch the original commit before verifying") from error

    adapter = "corpus_runtime.py"
    rules = json.loads(git_file("streaming-compatibility.json"))
    files = {name: git_file(name) for name in {adapter} | {key for rule in rules for key in rule["current"]}}
    # Colab uses LF source. Also recognize the exact CRLF variants already supported by the repo.
    for crlf in (False, True):
        hashes = {name: sha256(data.replace(b"\n", b"\r\n") if crlf else data).hexdigest()
                  for name, data in files.items()}
        rule = next((rule for rule in rules if all(hashes[name] == value for name, value in rule["current"].items())), None)
        locked_hash = rule["previous"].get(adapter, hashes[adapter]) if rule else hashes[adapter]
        if locked_hash == expected:
            return
    raise ValueError("Baseline revision does not match the snapshot's adapter lock")


def verify_baseline(baseline, backup, output, repo, revision=None, logs=None):
    from direct_s2st.s2ut.paper_recipe import MAX_UPDATES

    record = benchmark.read_record(baseline)
    selected = Path(record["checkpoint"])
    work = selected.parent.parent
    backup, output = Path(backup).resolve(), Path(output).resolve()
    if (record["method"] != "baseline" or selected.name != "checkpoint_best.pt"
            or output.is_relative_to(backup) or output.is_relative_to(work) or output.is_relative_to(Path(record["data"]))):
        raise ValueError("Baseline provenance output must be separate from protected inputs")
    hashes = {"checkpoints/checkpoint_best.pt": record["checkpoint_hash"],
              "research-metadata.json": record["provenance"]["metadata_hash"],
              "s2ut-recipe-lock.json": record["provenance"]["recipe_lock_hash"]}
    for name in ("research-metadata.json", "s2ut-recipe-lock.json"):
        if file_hash(work / name) != hashes[name]:
            raise ValueError("restored Baseline does not match the registered evidence")
    last = work / BASELINE_MEMBERS[0]
    hashes[BASELINE_MEMBERS[0]] = file_hash(last)
    for marker in sorted(backup.glob("*/snapshot.json"), reverse=True):
        receipt = read_json(marker)
        if receipt.get("updates") == MAX_UPDATES and all(receipt.get("files", {}).get(k) == v for k, v in hashes.items()):
            break
    else:
        raise ValueError("no completed Baseline snapshot matches the restored checkpoints and recipe")

    identity = receipt["identity"]
    config = identity["config"]
    launch_work = identity["work"]
    locks = [path for path in config["identity_files"] if Path(path).name == "baseline-runtime.json"]
    if len(locks) != 1:
        raise ValueError("Baseline snapshot must bind one runtime lock")
    lock_path = locks[0]
    runtime_lock = work.parent / Path(lock_path).name
    if (not {"alignment_transfer_s2ut.corpus_runtime", "_fairseq"} <= set(config["command"])
            or config["kind"] != "s2ut" or file_hash(runtime_lock) != identity["inputs"][lock_path]):
        raise ValueError("Baseline snapshot runtime lock differs")
    lock = read_json(runtime_lock)
    if lock["repository"] != BENCHMARK_REVISION or lock["fairseq"] != record["fairseq_revision"]:
        raise ValueError("Baseline runtime lineage differs")

    saved = benchmark.load_native(last)
    native = benchmark.scientific_config(benchmark.plain_config(saved))
    if (record["training_budget"] != MAX_UPDATES or benchmark.checkpoint_updates(saved) != MAX_UPDATES
            or saved.get("extra_state", {}).get("s2st_recipe_fingerprint") != record["baseline_fingerprint"]
            or native != record["scientific_config"]
            or record["metadata"].get("repository_revision") != BENCHMARK_RUNTIME_REVISION):
        raise ValueError("completed Baseline checkpoint, recipe, and benchmark revision disagree")

    entries = checkout_evidence(logs, launch_work, str(backup)) if logs else []
    recorded = record["metadata"].get("alignment_revision")
    candidates = {value for value in (revision, recorded, entries[0]["revision"] if entries else None) if value}
    if len(candidates) > 1:
        raise ValueError("Baseline revision sources disagree")
    candidate = next(iter(candidates), None)
    if candidate:
        verify_adapter(repo, candidate, lock["adapter_sha256"])
    source = "launch_log" if entries else "metadata" if recorded else "notebook_pin" if revision else "unavailable"
    evidence = {**baseline_evidence(record),
        "comparison_recipe_verified": True, "baseline_record_hash": record["record_hash"],
        "snapshot_receipt_sha256": file_hash(marker), "checkpoint_hashes": hashes,
        "adapter_sha256": lock["adapter_sha256"], "alignment_revision": candidate,
        "alignment_revision_source": source, "launch_evidence": entries}
    if output.suffix != ".json":
        output = output / (digest(evidence) + ".json")
    return {**write_result(output, evidence), "report_path": str(output)}


def recovery_check(work, backup, method, phase, *, baseline=None, report=False, output=None):
    work = Path(work)
    if baseline is not None:
        registered = read_json(baseline)
        if not torch.cuda.is_available() or not registered["config"]["common"]["fp16"]:
            raise ValueError("the real-corpus recovery check requires the registered CUDA FP16 recipe")
    receipts = {}
    for path in Path(backup).glob("*/snapshot.json"):
        receipt = read_json(path)
        receipts.setdefault(receipt["updates"], []).append(receipt)
    target = RECOVERY_UPDATES[phase]
    if receipts and max(receipts) > target:
        raise ValueError("this transfer run is past the short recovery check")
    saved_step = RECOVERY_UPDATES["save"]
    if phase == "resume" and saved_step not in receipts:
        raise ValueError("publish the update-2 snapshot before the recovery check")
    if not report:
        if phase == "resume" and work.exists() and any(work.iterdir()):
            raise ValueError("open a fresh Colab runtime for the recovery check")
        return {"phase": phase, "target": target}

    plan = read_json(work / "native-run.json")
    session = read_json(work / "native-session.json")
    if (target not in receipts or session["status"] != "PILOT_PAUSED" or session["durable_updates"] != target
            or plan["transfer"]["method"] != method):
        raise ValueError("recovery check did not publish the requested Pilot checkpoint")
    checkpoint = work / BASELINE_MEMBERS[0]
    checkpoint_hash = file_hash(checkpoint)
    if not any(receipt["files"].get(BASELINE_MEMBERS[0]) == checkpoint_hash for receipt in receipts[target]):
        raise ValueError("recovery checkpoint differs from the published receipt")
    state = benchmark.load_native(checkpoint)
    extra = state["extra_state"]
    steps = [row["step"] for row in read_jsonl(work / "alignment-train.jsonl")]
    if (steps != list(range(1, target + 1)) or benchmark.checkpoint_updates(state) != target
            or extra.get("alignment_contract_hash") != plan["contract_hash"]):
        raise ValueError("recovery check found an inconsistent update sequence or checkpoint")
    if phase == "resume" and not any(entry["num_updates"] == saved_step for entry in state["optimizer_history"][:-1]):
        raise ValueError("native optimizer history does not show resume from the update-2 checkpoint")
    fp16 = plan["config"]["common"]["fp16"]
    if fp16 and (not extra.get("alignment_fp16_scaler") or not extra.get("alignment_fp32_master")):
        raise ValueError("recovery checkpoint lacks native FP16 resume state")
    result = {"status": "RECOVERY_CHECK_PASSED" if phase == "resume" else "CHECKPOINT_SAVED",
            "method": method, "contract_hash": plan["contract_hash"], "checkpoint_sha256": checkpoint_hash,
            "updates": steps, "iterator": extra["train_iterator"], "fp16": fp16,
            "configured_workers": plan["config"]["dataset"]["num_workers"],
            "checkpoint_workers": benchmark.plain_config(state)["dataset"]["num_workers"],
            "warmup_updates": plan["config"]["lr_scheduler"].get("warmup_updates"),
            "training_budget": plan["training_budget"]}
    if output is not None:
        output = Path(output).resolve()
        if output.is_relative_to(Path(backup).resolve()) or output.is_relative_to(work.resolve()):
            raise ValueError("recovery report must be separate from the run and snapshots")
        return {**write_result(output, result), "report_path": str(output)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    child = sub.add_parser("baseline")
    for name in ("baseline", "backup", "output", "repo"):
        child.add_argument("--" + name, required=True)
    child.add_argument("--revision")
    child.add_argument("--logs")
    child = sub.add_parser("recovery")
    for name in ("work", "backup", "method"):
        child.add_argument("--" + name, required=True)
    child.add_argument("--phase", choices=RECOVERY_UPDATES, required=True)
    child.add_argument("--baseline", required=True)
    child.add_argument("--report", action="store_true")
    child.add_argument("--output")
    args = vars(parser.parse_args())
    action = args.pop("action")
    result = verify_baseline(**args) if action == "baseline" else recovery_check(**args)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
