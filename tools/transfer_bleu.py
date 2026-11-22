"""Evaluate published transfer checkpoints with the pinned benchmark's evaluator."""

import argparse
from copy import deepcopy
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alignment_transfer_s2ut.process_logging import run_process


PLAN = "native-run.json"
CHECKPOINT = "checkpoints/checkpoint_last.pt"
METHODS = ("distillation", "annealed_distillation", "mask_distillation")
SHA256 = re.compile("[0-9a-f]{64}")


def result_hash(result):
    fields = {key: value for key, value in result.items() if key != "result_hash"}
    return sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def snapshots(backup, work, method, plan_sha=None, *, allow_new=False):
    """Discover candidates from receipts; the payload is verified before evaluation."""
    from alignment_transfer_s2ut.colab_runtime import BENCHMARK_REVISION

    found = {}
    identities = set()
    for marker in sorted(backup.glob("*/snapshot.json")):
        receipt = json.loads(marker.read_bytes())
        identity, files = receipt["identity"], receipt["files"]
        command = identity["command"]
        step = receipt["updates"]
        if (identity["work"] != str(work) or identity["benchmark_revision"] != BENCHMARK_REVISION
                or "alignment_transfer_s2ut.benchmark" not in command or "train" not in command
                or command[command.index("--output") + 1] != str(work)
                or command[command.index("--method") + 1] != method):
            raise ValueError(f"Unexpected training identity in {marker}")
        if (type(step) is not int or step < 1 or int(marker.parent.name.split("-", 1)[0]) != step
                or receipt.get("storage") != "zip-v1"):
            raise ValueError(f"Invalid native snapshot receipt: {marker}")
        if not (marker.parent / "snapshot.zip").is_file():
            continue
        if any(not SHA256.fullmatch(files.get(name, "")) for name in (PLAN, CHECKPOINT)):
            raise ValueError(f"Snapshot is missing plan or checkpoint hashes: {marker}")
        if plan_sha and files[PLAN] != plan_sha:
            raise ValueError("Saved plan differs from the current training plan")
        identities.add((files[PLAN], json.dumps(identity, sort_keys=True)))
        previous = found.get(step)
        if previous and previous[1]["files"][CHECKPOINT] != files[CHECKPOINT]:
            raise ValueError(f"Conflicting checkpoints for update {step}")
        found[step] = marker.parent, receipt
    if len(identities) > 1:
        raise ValueError("The training stage contains multiple contracts")
    if not found:
        if not allow_new:
            raise FileNotFoundError(f"No published transfer snapshots at {backup}")
        if (any(path.exists() and any(path.iterdir()) for path in (work, backup))
                or any(work.parent.glob(work.name + ".interrupted-*"))):
            raise ValueError("No published snapshot and training files already exist; refusing to start from update 0")
    return found


def inventory(found, milestones):
    latest = max(found, default=0)
    return {"latest": latest, "available": [step for step in milestones if step in found],
            "missing_past": [step for step in milestones if step <= latest and step not in found],
            "online": [step for step in milestones if step >= latest]}


def baseline_dev_hash(backup, work, budget):
    common = str(work.parents[1] / "data/common/dev.jsonl")
    hashes = set()
    for marker in sorted(backup.glob("*/snapshot.json"), reverse=True):
        receipt = json.loads(marker.read_bytes())
        if (receipt["updates"] == budget and (marker.parent / "snapshot.zip").is_file()):
            checksum = receipt["identity"]["inputs"].get(common, "")
            if SHA256.fullmatch(checksum):
                hashes.add(checksum)
    if len(hashes) != 1:
        raise ValueError("Completed Baseline receipts must agree on the Dev references")
    return hashes.pop()


def verified_state(work, snapshot, scratch, update):
    from alignment_transfer_s2ut.benchmark import checkpoint_updates, load_native
    from alignment_transfer_s2ut.colab_runtime import runtime
    from alignment_transfer_s2ut.io import digest, file_hash, read_json

    files = snapshot[1]["files"]
    # Prefer a verified local copy, then use the benchmark's verified ZIP staging.
    with TemporaryDirectory(dir=scratch, prefix="bleu-checkpoint-") as temp:
        target = Path(temp) / "inputs"
        target.mkdir()
        for name in (PLAN, CHECKPOINT):
            if (work / name).is_file() and file_hash(work / name) == files[name]:
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(work / name, target / name)
        if not all((target / n).is_file() and file_hash(target / n) == files[n] for n in (PLAN, CHECKPOINT)):
            runtime().restore(snapshot, Path(temp) / "restored")
            target = Path(temp) / "restored"
        plan = read_json(target / PLAN)
        if plan["contract_hash"] != digest({k: v for k, v in plan.items() if k != "contract_hash"}):
            raise ValueError("Saved training plan has a corrupt contract")
        state = load_native(target / CHECKPOINT)
        if (checkpoint_updates(state) != update
                or state["extra_state"]["alignment_contract_hash"] != plan["contract_hash"]):
            raise ValueError("Checkpoint does not match the requested update and contract")
        return plan, state


def evaluation_model(config, weights, device):
    from fairseq import tasks
    from alignment_transfer_s2ut.benchmark import native_config

    cfg = deepcopy(config)
    cfg["task"]["_name"] = "speech_to_speech"
    cfg["task"].pop("alignment_config", None)
    cfg["task"]["eval_inference"] = False
    cfg["model"]["_name"] = "s2ut_transformer_fisher"
    if "arch" in cfg["model"]:
        cfg["model"]["arch"] = cfg["model"]["_name"]
    native = native_config(cfg)
    task = tasks.setup_task(native.task)
    task.load_dataset("dev")
    model = task.build_model(native.model)
    model.load_state_dict(weights, strict=True)
    model = (model.half() if cfg["common"]["fp16"] else model.float()).to(device)
    return task, model


def score_checkpoint(plan, state, scratch, update, policy, common_sha):
    import torch
    from direct_s2st.s2ut import bleu_monitor
    from alignment_transfer_s2ut.colab_runtime import native_inputs
    from alignment_transfer_s2ut.io import file_hash, read_jsonl
    from alignment_transfer_s2ut.s2ut.fairseq_extension import verify_fairseq

    verify_fairseq()
    if not torch.cuda.is_available():
        raise RuntimeError("Transfer BLEU requires a CUDA runtime")
    data = Path(plan["config"]["task"]["data"])
    common = data.parents[1] / "common/dev.jsonl"
    if not common_sha or file_hash(common) != common_sha:
        raise ValueError("Dev reference text differs from the Baseline snapshot")
    for name, checksum in plan["prepared"]["files"].items():
        if (not name.endswith(".tsv") or Path(name).name == "dev.tsv") and file_hash(data / name) != checksum:
            raise ValueError(f"Evaluation input differs from the registered data: {name}")
    with native_inputs():
        task, model = evaluation_model(plan["config"], state["model"], "cuda")
        if len(task.dataset("dev")) < policy["aux_samples"]:
            raise ValueError("The registered Dev set has fewer than 500 examples")
        before = bleu_monitor.weight_hash(model)
        os.environ["S2ST_S2UT_BLEU_POLICY"] = json.dumps(policy)
        chosen = bleu_monitor.selected_rows(read_jsonl(common), policy["aux_samples"], policy["seed"])
        result = bleu_monitor.evaluate(SimpleNamespace(get_model=lambda: model), task, scratch, update)
        if result["identity"]["ids"] != [row["pair_id"] for row in chosen]:
            raise ValueError("The evaluator returned a different fixed Dev subset")
        if bleu_monitor.weight_hash(model) != before:
            raise RuntimeError("Evaluation modified model weights")
    return result


def publish_result(path, result):
    from direct_s2st.io import atomic_write_json, atomic_write_text

    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=sorted(result), lineterminator="\n")
    writer.writeheader()
    writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                     for key, value in result.items()})
    atomic_write_json(path, result, resume=True)
    atomic_write_text(path.with_suffix(".csv"), table.getvalue(), resume=True)


def evaluate(args, snapshot, policy):
    from direct_s2st.io import atomic_write_json
    from alignment_transfer_s2ut.colab_runtime import runtime, session_paths
    from alignment_transfer_s2ut.io import digest
    from filelock import FileLock

    work, backup = session_paths(args.work, args.backup)
    scratch, output = session_paths(args.scratch, args.output)
    protected = [work.parent, backup.parent, work.parents[1] / "baseline", work.parents[1] / "data"]
    if args.baseline_backup:
        protected.append(args.baseline_backup.resolve())
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a in (scratch, output) for b in protected):
        raise ValueError("BLEU output and scratch must be separate from training files")
    files = snapshot[1]["files"]
    update = snapshot[1]["updates"]
    runtime()
    scope = digest({"plan": files[PLAN], "policy": policy, "common_dev_sha": args.dev_sha})
    directory = output / args.method / scope / f"{update:09d}"
    target = directory / "result.json"
    job = scratch / args.method / scope / f"{update:09d}"
    local_result = job / "publication.json"
    job.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".evaluation.lock"), timeout=0):
        recorded = target if target.is_file() else local_result
        if recorded.is_file():
            result = json.loads(recorded.read_bytes())
            if (result["checkpoint_sha256"] != files[CHECKPOINT] or result["update"] != update
                    or result["method"] != args.method or result["policy"] != policy
                    or result["dev_sha256"] != args.dev_sha
                    or result["result_hash"] != result_hash(result)):
                raise ValueError("Existing BLEU result differs from the published evaluation identity")
            publish_result(target, result)
            print(f"Reused BLEU at update {update}: {target}", flush=True)
            return
        try:
            plan, state = verified_state(work, snapshot, job, update)
            if plan["transfer"]["method"] != args.method:
                raise ValueError("Saved plan has the wrong transfer method")
            result = score_checkpoint(plan, state, job, update, policy, args.dev_sha)
            if (result["update"] != update or result["aux_samples"] != policy["aux_samples"]
                    or result["asr_samples"] != policy["asr_samples"]
                    or result["status"] not in {"completed", "completed_with_failures"}):
                raise ValueError("Incomplete evaluation result")
            for name in ("aux", "asr"):
                if result[f"{name}_failures"] == result[f"{name}_samples"]:
                    raise RuntimeError(f"Every {name} sample failed; inspect the local evaluation journal")
                journal = job / "training-bleu" / f"{update:09d}" / f"{name}.jsonl"
                if journal.is_file() and "TimeoutError:" in journal.read_text(encoding="utf-8"):
                    raise RuntimeError("Evaluation exhausted its time budget; retry to continue from the local journal")
            evaluation = result.pop("identity")
            ids = evaluation.pop("ids")
            if len(ids) != policy["aux_samples"] or len(set(ids)) != len(ids):
                raise ValueError("Evaluation returned unexpected sample IDs")
            evaluation.pop("references", None)
            repo = Path(__file__).resolve().parents[1]
            command = ["git", "-C", repo, "show", "-s", '--format="%H"', "HEAD"]
            revision = json.loads(run_process(command, args.logs, name="bleu-source-revision",
                                              capture_json=True).stdout)
            result.update(method=args.method, checkpoint_sha256=files[CHECKPOINT],
                          contract_hash=plan["contract_hash"], aux_pair_ids=ids, asr_pair_ids=ids[:policy["asr_samples"]],
                          dev_sha256=args.dev_sha, policy=policy, revision=revision,
                          benchmark_revision=snapshot[1]["identity"]["benchmark_revision"], evaluation=evaluation)
            result["result_hash"] = result_hash(result)
            atomic_write_json(local_result, result, resume=True)
            publish_result(target, result)
            print(json.dumps({"update": update, "bleu": str(target), "status": result["status"]}), flush=True)
        except Exception as error:
            atomic_write_json(directory / f"failed-{uuid4().hex}.json",
                              {"status": "failed", "update": update, "checkpoint_sha256": files[CHECKPOINT],
                               "error": f"{type(error).__name__}: {error}", "local_retry": str(job)})
            raise


def sync_wandb(args):
    import wandb

    rows = [json.loads(path.read_bytes()) for path in args.output.glob(f"{args.method}/*/*/result.json")]
    rows = [r for r in rows if r["method"] == args.method
            and (not args.contract or r["contract_hash"] == args.contract)
            and r["status"] in {"completed", "completed_with_failures"}]
    if not rows:
        return
    contracts = {r["contract_hash"] for r in rows}
    if len(contracts) != 1:
        raise ValueError("Select the contract to publish to W&B")
    points = {}
    for row in sorted(rows, key=lambda row: row["update"]):
        if row["result_hash"] != result_hash(row):
            raise ValueError("Corrupt BLEU result")
        step = row["update"]
        if step in points and points[step] != row:
            raise ValueError(f"Multiple BLEU results for update {step}")
        points[step] = row
    identifier = sha256(f"transfer-bleu:{contracts.pop()}".encode()).hexdigest()[:24]
    with wandb.init(project=args.project, entity=args.entity or None, group=args.group, id=identifier,
                    name=f"{args.method}-bleu", dir=str(args.logs), resume="allow",
                    settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                            x_disable_meta=True, x_disable_stats=True,
                                            init_timeout=20, finish_timeout=20)) as run:
        run.define_metric("eval/checkpoint_update")
        run.define_metric("eval/*", step_metric="eval/checkpoint_update")
        for step, row in points.items():
            marker = f"bleu_checkpoint_{step}"
            previous = run.summary.get(marker)
            if previous == row["result_hash"]:
                continue
            if previous is not None:
                raise ValueError(f"W&B already has a different BLEU result for update {step}")
            values = {f"eval/{name}": row[name] for name in ("aux_bleu", "asr_bleu", "aux_failures", "asr_failures")}
            run.log({**values, "eval/checkpoint_update": step, marker: row["result_hash"]})
        print(f"BLEU W&B: {run.url}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "evaluate", "backfill", "wandb"))
    parser.add_argument("--method", choices=METHODS, required=True)
    for name in ("work", "backup", "baseline-backup", "scratch", "output", "logs"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--update", type=int)
    parser.add_argument("--allow-new", action="store_true")
    for name in ("project", "entity", "group", "contract"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.plan_sha and not SHA256.fullmatch(args.plan_sha):
        parser.error("--plan-sha must be a SHA256")
    if args.allow_new and (args.action != "inventory" or not args.plan_sha):
        parser.error("--allow-new requires inventory with the current --plan-sha")
    if args.action == "wandb":
        if not all((args.output, args.logs, args.project)):
            parser.error("wandb requires output, logs and project")
        sync_wandb(args)
        return

    from alignment_transfer_s2ut.colab_runtime import runtime

    runtime()
    from direct_s2st.s2ut.paper_recipe import MAX_UPDATES, MILESTONES, bleu_policy

    if not args.work or not args.backup:
        parser.error("work and backup are required")
    found = snapshots(args.backup.resolve(), args.work.resolve(), args.method, args.plan_sha,
                      allow_new=args.allow_new)
    listing = inventory(found, MILESTONES)
    if args.action == "inventory":
        print(json.dumps(listing))
        return
    if not all((args.scratch, args.output, args.logs, args.baseline_backup)):
        parser.error("evaluation requires scratch, output, logs and a Baseline backup")
    args.dev_sha = baseline_dev_hash(args.baseline_backup, args.work.resolve(), MAX_UPDATES)
    if args.action == "evaluate":
        if args.update not in listing["available"]:
            raise ValueError(f"No published checkpoint at the requested milestone {args.update}")
        evaluate(args, found[args.update], bleu_policy())
        return
    print(json.dumps(listing), flush=True)
    if args.update is not None and args.update not in listing["available"]:
        parser.error(f"no published checkpoint at milestone {args.update}")
    for step in ([args.update] if args.update is not None else listing["available"]):
        child = [sys.executable, str(Path(__file__).resolve()), "evaluate", "--update", str(step)]
        for key in ("method", "work", "backup", "baseline_backup", "scratch", "output", "logs", "plan_sha"):
            if value := getattr(args, key):
                child.extend(("--" + key.replace("_", "-"), str(value)))
        run_process(child, args.logs, name=f"bleu-backfill-{args.method}-{step}")


if __name__ == "__main__":
    main()
