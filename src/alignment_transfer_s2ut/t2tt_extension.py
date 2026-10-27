"""Verified, read-only-source continuation of an existing T2TT training run.

The original Python source tree is recovered by its complete implementation hash.
The trainer itself is never patched: only its operational checkpoint callback is
wrapped to schedule durable ZIPs and record lineage/milestones.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import uuid

from alignment_transfer_s2ut.io import digest, file_hash, read_json, write_json, atomic_write
from alignment_transfer_s2ut.progress import progress
from alignment_transfer_s2ut.training import load_checkpoint

REQUIRED = {"checkpoint_last.pt", "checkpoint_best.pt", "experiment.json", "metrics.jsonl"}
EXTRA = ("extension-lineage.json", "extension-plan.json", "extension-milestones.json", ".extension-control.py")


def progress_update(operation, progress_path):
    """Only Windows reader/atomic-replace contention on the progress file retries."""
    for attempt in range(20):
        try:
            return operation()
        except PermissionError as error:
            target = getattr(error, "filename2", None)
            if (os.name != "nt" or target is None or Path(target).resolve() != progress_path.resolve() or attempt == 19):
                raise
            time.sleep(.01)


def option(command, name):
    if command.count(name) != 1 or command.index(name) + 1 >= len(command):
        raise ValueError("exactly one " + name + " required in the saved T2TT command")
    return command[command.index(name) + 1]


def changed_option(command, name, value):
    result = list(command)
    option(result, name)
    result[result.index(name) + 1] = str(value)
    return result


def implementation_hash(package):
    return digest({str(p.relative_to(package)): file_hash(p) for p in sorted(Path(package).rglob("*.py"))})


def separated(source_work, source_backup, work, backup):
    sources, targets = list(map(lambda p: Path(p).resolve(), (source_work, source_backup))), list(map(
        lambda p: Path(p).resolve(), (work, backup)))
    for a in sources:
        for b in targets:
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("source 20k and target extension paths must be separate; source is read-only")
    if targets[0].is_relative_to(targets[1]) or targets[1].is_relative_to(targets[0]):
        raise ValueError("target work/backup must be separate")
    return *sources, *targets


def saved_identity(state, work, updates, manifest, config):
    from alignment_transfer_s2ut.colab_runtime import BENCHMARK_REVISION
    identity = state.get("identity", {})
    command = identity.get("command", [])
    if (state.get("storage") != "zip-v1" or identity.get("benchmark_revision") != BENCHMARK_REVISION or
            identity.get("work") != str(work) or len(command) < 3 or
            command[1:3] != ["-m", "alignment_transfer_s2ut.train_t2tt"] or
            option(command, "--output") != str(work) or int(option(command, "--max-updates")) != updates):
        raise ValueError("saved T2TT stage/module/output/budget/revision differs")
    for flag, expected in (("--manifest", manifest), ("--config", config)):
        actual = Path(option(command, flag)).resolve()
        if actual != expected or identity.get("inputs", {}).get(str(actual)) != file_hash(expected):
            raise ValueError("saved T2TT " + flag + " path/content differs")
    return identity, command


def latest_snapshot(backup):
    from alignment_transfer_s2ut.colab_runtime import runtime
    markers = sorted(Path(backup).glob("*/snapshot.json"), reverse=True)
    if not markers:
        return None
    state = read_json(markers[0])
    snapshot = runtime().latest(backup, state["identity"])
    if snapshot is None:
        raise ValueError("no verified snapshot; refusing to restart or seed from corrupt ZIPs")
    return snapshot


def git_command(command, logs):
    from alignment_transfer_s2ut.process_logging import run_process
    run_process(command, logs, name="extension-git:" + command[3])


def recover_implementation(work, expected, repo, revision=None):
    """Copy exact current sources, or find and export the matching Git revision."""
    destination = work / ".training-runtime"
    package = destination / "src/alignment_transfer_s2ut"
    if package.is_dir():
        if implementation_hash(package) != expected:
            raise ValueError("isolated T2TT implementation changed")
        return str(destination / "src"), revision
    from alignment_transfer_s2ut import training
    current = Path(training.__file__).parent
    if implementation_hash(current) == expected:
        shutil.copytree(current, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if implementation_hash(package) != expected:
            raise ValueError("source tree changed while copying")
        return str(destination / "src"), None
    logs = work / "execution"
    history = work / ".source-history.txt"
    if revision:
        revisions = [revision]
    else:
        git_command(["git", "-C", str(repo), "log", "--format=%H", "--output=" + str(history),
                     "--", "src/alignment_transfer_s2ut"], logs)
        revisions = history.read_text(encoding="utf-8").splitlines()
    with tempfile.TemporaryDirectory(dir=work, prefix=".source-search-") as temp:
        temp = Path(temp)
        for i, candidate in enumerate(revisions):
            archive = temp / "source.tar"
            git_command(["git", "-C", str(repo), "archive", "--format=tar", "--output=" + str(archive),
                         candidate, "src/alignment_transfer_s2ut"], logs)
            exported = temp / candidate
            with tarfile.open(archive) as pack:
                for member in pack:
                    name = Path(member.name)
                    if member.isdir():
                        continue
                    if (not member.isfile() or name.is_absolute() or ".." in name.parts or
                            name.parts[:2] != ("src", "alignment_transfer_s2ut")):
                        raise ValueError("unsafe Git source archive")
                    target = exported / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with pack.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
            found = exported / "src/alignment_transfer_s2ut"
            progress("extension: verify original source revision", i + 1, len(revisions))
            if implementation_hash(found) == expected:
                shutil.copytree(found, package)
                return str(destination / "src"), candidate
    raise ValueError("No Git source tree matches the checkpoint implementation_hash; contract is not rewritten")


def metric_rows(path, through=None):
    expected, gaps, rows = 1, [], []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            step = row["step"]
            if through is not None and step > through:
                continue
            if type(step) is not int or step != expected:
                if len(gaps) < 20:
                    gaps.append({"expected": expected, "actual": step})
            expected = step + 1
            rows.append({k: row[k] for k in ("step", "train/main_loss", "dev/main_loss", "train/learning_rate") if k in row})
    return rows, gaps


def verify_state(work, expected_step=None):
    if not all((work / name).is_file() for name in REQUIRED):
        raise ValueError("extension requires last, best, experiment and full metrics history")
    last, best = [load_checkpoint(work / name) for name in ("checkpoint_last.pt", "checkpoint_best.pt")]
    if (last.get("modality") != "text" or best.get("modality") != "text" or
            (expected_step is not None and last.get("step") != expected_step) or
            not 0 < best["step"] <= last["step"] or not last.get("optimizer", {}).get("state") or
            not last["optimizer"].get("param_groups") or not last.get("model")):
        raise ValueError("invalid T2TT last/best state or actual source update")
    contract_fields = ("backbone", "backend", "modality", "runtime", "implementation_hash", "source_vocabulary",
                       "target_vocabulary", "recipe", "auxiliary", "train_hash", "dev_hash", "method", "transfer",
                       "teacher_signature", "unit_config_hash")
    contract = {key: last[key] for key in contract_fields}
    experiment = read_json(work / "experiment.json")
    if (digest(contract) != last["contract_hash"] or best["contract_hash"] != last["contract_hash"] or
            digest({k: best[k] for k in contract_fields}) != last["contract_hash"] or
            digest({k: experiment[k] for k in contract_fields}) != last["contract_hash"] or
            experiment["contract_hash"] != last["contract_hash"] or best["best_main_loss"] != last["best_main_loss"]):
        raise ValueError("saved scientific contract differs between checkpoint/experiment")
    for key in ("step", "epoch", "cursor", "best_main_loss", "bad_validations", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"):
        if key not in last:
            raise ValueError("missing complete continuation state: " + key)
    rows, gaps = metric_rows(work / "metrics.jsonl", last["step"])
    if gaps or not rows or rows[-1]["step"] != last["step"]:
        raise ValueError("source metrics are not a continuous prefix through the saved last update")
    return last, best


def milestone(work, step, last=None, best=None):
    last = last or load_checkpoint(work / "checkpoint_last.pt")
    best = best or load_checkpoint(work / "checkpoint_best.pt")
    path = work / "extension-milestones.json"
    doc = read_json(path) if path.is_file() else {}
    doc[str(step)] = dict(update=step, best_main_loss=last["best_main_loss"], best_checkpoint_step=best["step"])
    write_json(path, doc, overwrite=path.exists())


def trim_frozen_metrics(path, step):
    """Exclude uncommitted failure-exit tail only in the immutable backup copy."""
    rows, gaps = metric_rows(path, step)
    if gaps or not rows or rows[-1]["step"] != step:
        raise ValueError("cannot publish discontinuous extension metric history")
    def copy_prefix(output):
        with path.open("rb") as source:
            for line in source:
                if line.strip() and json.loads(line)["step"] <= step:
                    output.write(line)
    atomic_write(path, copy_prefix, overwrite=True)


def prune_target(backup, identity, keep_last, milestones):
    """Delete only owned old ZIP/receipt pairs after a verified new publication."""
    backup = Path(backup).resolve()
    entries = []
    for marker in backup.glob("*/snapshot.json"):
        try:
            state = read_json(marker)
            if not isinstance(state, dict) or type(state.get("updates")) is not int:
                progress("extension retention: retained invalid receipt " + str(marker))
                continue
            if state.get("identity") == identity:
                entries.append((state["updates"], marker.parent))
        except (ValueError, OSError, KeyError) as error:
            progress("extension retention: retained unreadable receipt: " + str(error))
    entries.sort(key=lambda x: (x[0], x[1].name), reverse=True)
    for update, directory in entries[keep_last:]:
        if update in milestones:
            continue
        try:
            if (directory.is_symlink() or directory.resolve().parent != backup or
                    {p.name for p in directory.iterdir()} != {"snapshot.zip", "snapshot.json"} or
                    any(p.is_symlink() for p in directory.iterdir())):
                progress("extension retention: retained unexpected directory " + str(directory))
                continue
            (directory / "snapshot.zip").unlink()
            (directory / "snapshot.json").unlink()
            directory.rmdir()
            progress("extension retention: removed old non-milestone snapshot " + str(update))
        except OSError as error:
            progress("extension retention: cleanup failed; verified new snapshot is retained: " + str(error))


@contextmanager
def extension_recovery_files():
    from alignment_transfer_s2ut.colab_runtime import recovery_files
    from direct_s2st import recovery
    with recovery_files():
        saved = recovery.ARTIFACTS
        recovery.ARTIFACTS = saved + EXTRA
        try:
            yield
        finally:
            recovery.ARTIFACTS = saved


def worker(plan_path, preflight=False):
    """Executed as a script with the exact recovered package first on PYTHONPATH."""
    from alignment_transfer_s2ut import training, colab_runtime
    from alignment_transfer_s2ut.cli import parser_for, run
    plan = read_json(plan_path)
    work = Path(plan["work"])
    if (implementation_hash(Path(training.__file__).parent) != plan["implementation_hash"] or
            file_hash(__file__) != plan["control_hash"]):
        raise ValueError("isolated trainer/control code changed")
    saved = load_checkpoint(work / "checkpoint_last.pt")
    if saved["contract_hash"] != plan["contract_hash"]:
        raise ValueError("continuation contract changed")
    args = parser_for("train_t2tt").parse_args(plan["command"][3:])
    args.resume = True
    if preflight:
        args.max_updates = saved["step"]  # Zero optimizer steps; original contract validation stays enabled.
        result = run("train_t2tt", args)
        if read_json(work / "experiment.json")["contract_hash"] != plan["contract_hash"]:
            raise ValueError("original trainer rejected the target contract")
        return {"contract_hash": plan["contract_hash"], "step": result["step"]}
    original = colab_runtime.saved_checkpoint
    colab_runtime.EXTRA_RECOVERY = colab_runtime.EXTRA_RECOVERY + EXTRA
    def scheduled(work_dir, checkpoint, updates, force=False):
        path = Path(checkpoint)
        stat = path.stat()
        progress_path = Path(work_dir) / "training-local-progress.json"
        if updates in plan["milestones"]:
            milestone(Path(work_dir), updates)
        if updates == plan["target_updates"] or updates in plan["milestones"] or updates % plan["snapshot_updates"] == 0:
            progress_update(lambda: original(work_dir, checkpoint, updates, force=True), progress_path)
        else:
            progress_update(lambda: write_json(progress_path,
                dict(version=1, updates=updates, checkpoint=path.relative_to(work_dir).as_posix(),
                     bytes=stat.st_size, mtime_ns=stat.st_mtime_ns), overwrite=True), progress_path)
    colab_runtime.saved_checkpoint = scheduled
    progress("T2TT extension: checkpoint_last.pt; scientific contract unchanged", saved["step"], plan["target_updates"])
    return run("train_t2tt", args)


def extend_train_session(source_work, source_backup, work, backup, manifest, config,
                         source_updates=20000, target_updates=200000, snapshot_updates=10000,
                         keep_last=2, milestones=(50000, 100000, 150000, 200000), repo=None, command=None):
    from alignment_transfer_s2ut import colab_runtime as c
    from alignment_transfer_s2ut.process_logging import run_process
    from direct_s2st.session_runtime import run_continuous
    from direct_s2st.translatotron2.monitor import absolute_update_limit
    source_work, source_backup, work, backup = separated(source_work, source_backup, work, backup)
    if not 0 < source_updates < target_updates or snapshot_updates < 1 or keep_last < 1:
        raise ValueError("invalid extension budget/persistence settings")
    if absolute_update_limit("tt2", target_updates) != target_updates:
        raise ValueError("inherited benchmark monitor settings would change the extension update budget")
    manifest, config = Path(manifest).resolve(), Path(config).resolve()
    if any(p.is_relative_to(work) for p in (manifest, config)):
        raise ValueError("training inputs must be outside extension output")
    # Ensure benchmark runtime and local work constraints before restoring anything.
    c.runtime()
    work, backup = c.session_paths(work, backup)
    target = latest_snapshot(backup)
    if target:
        identity = target[1]["identity"]
        if identity.get("kind") != "t2tt_training_extension":
            raise ValueError("target backup is not an extension stage")
        _, target_command = saved_identity(target[1], work, target_updates, manifest, config)
        if command is not None and list(map(str, command)) != target_command:
            raise ValueError("target training command changed")
        current_inputs = c.identity_for(target_command, work)
        if current_inputs != {k: identity[k] for k in current_inputs}:
            raise ValueError("target input identity changed")
        c.runtime().restore(target, work)
        plan = read_json(work / "extension-plan.json")
        lineage = read_json(work / "extension-lineage.json")
        if (lineage["source_updates"] != source_updates or lineage["target_updates"] != target_updates or
                lineage["source_stage"] != str(source_backup) or plan["command"] != target_command):
            raise ValueError("extension lineage/settings changed")
        last, best = verify_state(work, target[1]["updates"])
        if last["contract_hash"] != lineage["source_contract_hash"] or lineage["target_contract_hash"] != last["contract_hash"]:
            raise ValueError("extension scientific contract changed")
        progress("extension: resume from target verified snapshot", last["step"], target_updates)
    else:
        source = latest_snapshot(source_backup)
        if source is None:
            raise FileNotFoundError("verified source T2TT snapshot required; no training was started")
        source_identity, source_command = saved_identity(source[1], source_work, source_updates, manifest, config)
        actual_source_identity = c.identity_for(source_command, source_work)
        if actual_source_identity != {k: source_identity.get(k) for k in actual_source_identity}:
            raise ValueError("source input/interpreter identity differs from the verified 20k snapshot")
        if source[1]["updates"] != source_updates or not REQUIRED <= source[1]["files"].keys():
            raise ValueError("source snapshot is incomplete or lacks full T2TT state")
        target_command = changed_option(changed_option(source_command, "--output", work), "--max-updates", target_updates)
        if command is not None and list(map(str, command)) != target_command:
            raise ValueError("only output directory and planned max updates may change")
        c.runtime().restore(source, work)
        last, best = verify_state(work, source_updates)
        if last["recipe"]["patience"] != 0:
            raise ValueError("extension requires the saved patience=0 recipe")
        identity = {**c.identity_for(target_command, work), "kind": "t2tt_training_extension",
                    "source_identity_hash": digest(source_identity), "source_contract_hash": last["contract_hash"]}
        lineage = dict(kind="t2tt_training_extension", source_stage=str(source_backup), source_updates=source_updates,
            target_updates=target_updates, source_snapshot_identity=source_identity,
            source_snapshot_zip_sha256=source[1]["zip_sha256"], source_snapshot=str(source[0]),
            source_checkpoint_last_sha256=source[1]["files"]["checkpoint_last.pt"],
            source_checkpoint_best_sha256=source[1]["files"]["checkpoint_best.pt"],
            source_contract_hash=last["contract_hash"], target_contract_hash=last["contract_hash"])
        plan = dict(work=str(work), command=target_command, source_updates=source_updates, target_updates=target_updates,
                    implementation_hash=last["implementation_hash"], contract_hash=last["contract_hash"], training_revision=None)
        shutil.copyfile(__file__, work / ".extension-control.py")
        plan["control_hash"] = file_hash(work / ".extension-control.py")
        milestone(work, source_updates, last, best)
    if (work / "training-monitor.json").exists():
        raise ValueError("unexpected early-stop monitor in extension; saved patience=0 must remain authoritative")
    if snapshot_updates % last["recipe"]["save_interval_updates"]:
        raise ValueError("durable interval must be a multiple of the unchanged local save interval")
    milestones = sorted({int(x) for x in [*milestones, 50000, 100000, 150000, 200000]
                         if source_updates < int(x) <= target_updates} | {target_updates})
    if any(x != target_updates and x % last["recipe"]["save_interval_updates"] for x in milestones):
        raise ValueError("milestones must coincide with existing local checkpoint saves")
    plan.update(snapshot_updates=snapshot_updates, milestones=milestones)
    runtime_path, revision = recover_implementation(work, plan["implementation_hash"],
        Path(repo or Path(__file__).resolve().parents[2]), plan.get("training_revision"))
    plan["training_revision"] = revision
    write_json(work / "extension-plan.json", plan, overwrite=True)
    env = dict(os.environ, PYTHONPATH=runtime_path + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("S2ST_TRAIN_STAGING", None)
    control = work / ".extension-control.py"
    if file_hash(control) != plan["control_hash"]:
        raise ValueError("extension control script changed")
    check = run_process([target_command[0], str(control), "_worker", "--plan", str(work / "extension-plan.json"), "--preflight"],
                        work / "execution", name="extension-contract-preflight", env=env, capture_json=True)
    actual_contract = json.loads(check.stdout)["contract_hash"]
    if actual_contract != last["contract_hash"]:
        raise ValueError("source/target contract mismatch")
    lineage["target_contract_hash"] = actual_contract
    for name, expected in (("checkpoint_last.pt", lineage["source_checkpoint_last_sha256"]),
                            ("checkpoint_best.pt", lineage["source_checkpoint_best_sha256"])):
        if not target and file_hash(work / name) != expected:
            raise ValueError("preflight changed the seed checkpoint bytes")
    experiment = read_json(work / "experiment.json")
    experiment["planned_max_updates"] = target_updates
    write_json(work / "experiment.json", experiment, overwrite=True)
    write_json(work / "extension-lineage.json", lineage, overwrite=True)
    source_best_step = read_json(work / "extension-milestones.json")[str(source_updates)]["best_checkpoint_step"]
    progress("T2TT extension: source stage=" + str(source_backup) + " source update=" + str(source_updates) +
             " target update=" + str(target_updates) + " resume checkpoint=checkpoint_last.pt source best step=" + str(source_best_step))
    def publish(files, destination, run_identity, updates):
        trim_frozen_metrics(Path(files) / "metrics.jsonl", updates)
        published = c.runtime().publish_packed(files, destination, run_identity, updates)
        prune_target(destination, run_identity, keep_last, milestones)
        return published
    with extension_recovery_files():
        if target is None:
            from direct_s2st.recovery import freeze_files
            with tempfile.TemporaryDirectory(dir=work.parent, prefix=".extension-seed-") as temp:
                frozen = Path(temp) / "files"
                freeze_files(work, work / "checkpoint_last.pt", frozen)
                publish(frozen, backup, identity, source_updates)
        if last["step"] >= target_updates:
            return dict(status="COMPLETE", durable_updates=last["step"], lineage=str(work / "extension-lineage.json"))
        config_run = dict(kind="tt2", checkpoint="checkpoint_last.pt", resume_args=[],
            command=[target_command[0], str(control), "_worker", "--plan", str(work / "extension-plan.json")],
            runtime=dict(cache_gb=8, readers=2, precision="default", adaptive_batch=False))
        def launch(arguments, env):
            import subprocess
            env = dict(env, PYTHONPATH=runtime_path + os.pathsep + os.environ.get("PYTHONPATH", ""))
            return subprocess.Popen(arguments, env=env)
        return run_continuous(config_run, work=work, backup=backup, identity=identity, completed=last["step"],
            total=target_updates, seconds=0, inspect_checkpoint=c.inspect_checkpoint, publish=publish, popen=launch)


def report(work, output):
    work, output = Path(work), Path(output)
    rows, gaps = metric_rows(work / "metrics.jsonl")
    last = load_checkpoint(work / "checkpoint_last.pt")
    if not rows or rows[-1]["step"] != last["step"]:
        gaps.append({"checkpoint_step": last["step"], "last_metric_step": rows[-1]["step"] if rows else None})
    result = dict(updates=last["step"], rows=len(rows), continuity_warnings=gaps,
                  milestones=read_json(work / "extension-milestones.json"))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "learning-curve.json", result, overwrite=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, field in zip(axes, ("train/main_loss", "dev/main_loss", "train/learning_rate")):
        selected = [r for r in rows if field in r]
        axis.plot([r["step"] for r in selected], [r[field] for r in selected])
        axis.axvline(read_json(work / "extension-lineage.json")["source_updates"], color="gray", linestyle="--")
        axis.set_ylabel(field)
    axes[-1].set_xlabel("optimizer update")
    figure.tight_layout(); figure.savefig(output / "learning-curve.png"); plt.close(figure)
    if gaps:
        progress("extension: WARNING metric update gaps/duplicates: " + json.dumps(gaps))
    return result


def status(work, target_updates):
    last, best = verify_state(Path(work), target_updates)
    lineage = read_json(Path(work) / "extension-lineage.json")
    if lineage["target_updates"] != target_updates or last["contract_hash"] != lineage["source_contract_hash"]:
        raise ValueError("extended teacher lineage/contract differs")
    return dict(completed_updates=last["step"], best_checkpoint_step=best["step"], best_main_loss=last["best_main_loss"],
                checkpoint_hash=file_hash(Path(work) / "checkpoint_best.pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    extend = sub.add_parser("extend")
    for name in ("source-work", "source-backup", "work", "backup", "manifest", "config"):
        extend.add_argument("--" + name, required=True)
    for name, default in (("source-updates", 20000), ("target-updates", 200000), ("snapshot-updates", 10000), ("keep-last", 2)):
        extend.add_argument("--" + name, type=int, default=default)
    extend.add_argument("--milestones", nargs="+", type=int, default=[50000, 100000, 150000, 200000])
    extend.add_argument("--repo")
    child = sub.add_parser("_worker")
    child.add_argument("--plan", required=True)
    child.add_argument("--preflight", action="store_true")
    child = sub.add_parser("report")
    child.add_argument("--work", required=True)
    child.add_argument("--output", required=True)
    child = sub.add_parser("status")
    child.add_argument("--work", required=True)
    child.add_argument("--target-updates", required=True, type=int)
    args = parser.parse_args()
    if args.action == "extend":
        result = extend_train_session(args.source_work, args.source_backup, args.work, args.backup,
            args.manifest, args.config, args.source_updates, args.target_updates, args.snapshot_updates,
            args.keep_last, args.milestones, args.repo)
    elif args.action == "_worker":
        result = worker(args.plan, args.preflight)
    elif args.action == "status":
        result = status(args.work, args.target_updates)
    else:
        result = report(args.work, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
