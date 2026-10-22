"""Adapters to the benchmark's pinned Drive staging and recovery implementation."""
import argparse
from contextlib import contextmanager
import importlib
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from .io import digest, file_hash, read_json, write_json
from .progress import progress

BENCHMARK_REVISION = "4361e55a99c7f4ea3a00c5c138f6820c34a07d06"
EXTRA_RECOVERY = ("native-run.json", "native-dev.jsonl", "checkpoint_best.pt", "experiment.json", "metrics.jsonl")


@lru_cache(maxsize=1)
def runtime():
    import direct_s2st
    root = Path(direct_s2st.__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != BENCHMARK_REVISION:
        raise ValueError("benchmark runtime revision differs from the tested pin")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "src"], check=True)
    return importlib.import_module("direct_s2st.colab")


@contextmanager
def recovery_files():
    from direct_s2st import recovery
    saved = recovery.ARTIFACTS
    recovery.ARTIFACTS = saved + EXTRA_RECOVERY
    try:
        yield
    finally:
        recovery.ARTIFACTS = saved


def saved_checkpoint(work, checkpoint, updates, force=False):
    if os.environ.get("S2ST_TRAIN_STAGING"):
        runtime()
        from direct_s2st.train_runtime import checkpoint_saved
        with recovery_files():
            checkpoint_saved(work, checkpoint, updates, force=force)


@contextmanager
def benchmark_training():
    if not os.environ.get("S2ST_TRAIN_STAGING"):
        yield
        return
    runtime()
    from direct_s2st.s2ut.fairseq_train import TrainingAudit
    from direct_s2st.s2ut.performance import runtime_hooks
    from fairseq.criterions.speech_to_speech_criterion import SpeechToUnitMultitaskTaskCriterion as Criterion
    audit = TrainingAudit()
    original = Criterion.forward
    def forward(self, model, *a, **kw):
        if model.training:
            audit.attach(model)
        result = original(self, model, *a, **kw)
        if model.training:
            audit.record(result[0], self.native_main_loss_sum, result[2])
        return result
    Criterion.forward = forward
    try:
        with recovery_files(), runtime_hooks(audit):
            yield
    finally:
        Criterion.forward = original
        for handle in audit.handles:
            handle.remove()


@contextmanager
def native_inputs():
    """Redirect physical reads, keeping original paths/configuration as identities."""
    if not os.environ.get("S2ST_INPUT_MAP"):
        yield
        return
    runtime()
    from .io import local_path
    from fairseq.data import Dictionary
    from fairseq.data.audio import data_cfg, speech_to_text_dataset as audio
    creator = audio.SpeechToTextDatasetCreator
    yaml_load, audio_load = data_cfg.get_config_from_yaml, audio.get_features_or_waveform
    dict_load, tsv_load = Dictionary.__dict__["load"], creator.__dict__["_load_samples_from_tsv"]
    def load_dictionary(cls, path, *a, **kw):
        return dict_load.__func__(cls, str(local_path(path)) if isinstance(path, (str, Path)) else path, *a, **kw)
    def load_tsv(cls, root, split):
        path = local_path(Path(root) / (split + ".tsv"))
        return tsv_load.__func__(cls, str(path.parent), split)
    def load_yaml(path):
        value = yaml_load(local_path(path))
        for settings in value.values():
            if isinstance(settings, dict) and "dict" in settings and "data" in settings:
                settings["dict"] = str(local_path(settings["dict"]))
                settings["data"] = str(local_path(Path(settings["data"]) / "train.tsv").parent)
        return value
    Dictionary.load = classmethod(load_dictionary)
    creator._load_samples_from_tsv = classmethod(load_tsv)
    data_cfg.get_config_from_yaml = load_yaml
    audio.get_features_or_waveform = lambda path, *a, **kw: audio_load(str(local_path(path)), *a, **kw)
    try:
        yield
    finally:
        Dictionary.load, creator._load_samples_from_tsv = dict_load, tsv_load
        data_cfg.get_config_from_yaml, audio.get_features_or_waveform = yaml_load, audio_load


def stage_inputs(benchmark, local, create_audio=False):
    """Use benchmark WAV archives verbatim; GPU sessions never build missing packs."""
    runtime()
    from direct_s2st.drive_staging import stage_audio, ensure_local, copy_verified, register, active_map
    from direct_s2st.preparation import file_stamp
    benchmark, local = Path(benchmark).resolve(), ensure_local(local)
    common = benchmark / "data/common"
    rows = stage_audio(common, common.parent / ".drive-audio-packs", local / "audio",
                       splits=("train", "dev"), languages=("ja", "en"), create=create_audio,
                       materialize=not create_audio)
    if create_audio:
        return {"status": "PACKED"}
    files = [*sorted((benchmark / "data/s2ut/fairseq").rglob("*")),
             common / "dataset-lock.json", common / "train.jsonl", common / "dev.jsonl",
             *sorted((benchmark / "data/s2ut/units").glob("*.json*"))]
    for source in files:
        if not source.is_file():
            continue
        # Do not fetch a large obsolete per-WAV unit tree. Imported inline units
        # are the benchmark's existing storage migration, not a new unit recipe.
        stamp = file_stamp(source)
        target = local / "metadata" / source.relative_to(benchmark)
        copy_verified(source, target)
        register(rows, source, target, stamp)
    with active_map(rows, local) as name:
        return {"input_map": str(name), "files": len(rows)}


def inspect_checkpoint(path, kind):
    from .benchmark import load_native
    state = load_native(path)
    if "optimizer_history" in state:
        if not state.get("last_optimizer_state") or not state.get("extra_state", {}).get("alignment_contract_hash"):
            raise ValueError("missing native optimizer/contract in recovery checkpoint")
        del state
        from direct_s2st.s2ut.checkpoint import validate_checkpoint
        return validate_checkpoint(Path(path))["num_updates"]
    if not state.get("optimizer") or not state.get("model") or int(state.get("step", 0)) < 1:
        raise ValueError("missing T2TT recovery state")
    return int(state["step"])


def session_paths(work, backup):
    from direct_s2st.drive_staging import ensure_local
    work, backup = ensure_local(work), Path(backup).resolve()
    if work.is_relative_to(backup) or backup.is_relative_to(work) or backup == Path(backup.anchor):
        raise ValueError("local work and backup must be separate dedicated directories")
    corpus = os.environ.get("CORPUS_ROOT")
    if corpus and (backup.is_relative_to(Path(corpus).resolve()) or Path(corpus).resolve().is_relative_to(backup)):
        raise ValueError("outputs must be outside CORPUS_ROOT")
    return work, backup


def identity_for(command, output):
    # Bind file inputs before restoring or skipping a completed stage. Directory
    # artifacts carry their own per-pair contracts and the workflow's data lock.
    output = Path(output).resolve()
    files = {}
    for item in command:
        path = Path(item)
        if path.is_file() and not path.resolve().is_relative_to(output):
            files[str(path.resolve())] = file_hash(path)
    return {"command": list(map(str, command)), "inputs": files, "work": str(output),
            "benchmark_revision": BENCHMARK_REVISION}


def train_session(command, work, backup):
    bench = runtime()
    from direct_s2st.session_runtime import run_continuous
    work, backup = session_paths(work, backup)
    identity = identity_for(command, work)
    previous = bench.latest(backup, identity)
    if previous:
        bench.restore(previous, work)
    elif any(backup.glob("*/snapshot.json")):
        raise ValueError("no verified training snapshot; refusing to restart")
    elif any(work.iterdir()):
        retained = work.with_name(work.name + ".interrupted-" + uuid.uuid4().hex)
        work.rename(retained)
        work.mkdir()
        progress("no durable snapshot; retained unpublished local files at " + str(retained) + "; starting at update 0")
    native = "alignment_transfer_s2ut.benchmark" in command
    relative = "checkpoints/checkpoint_last.pt" if native else "checkpoint_last.pt"
    last = work / relative
    completed = inspect_checkpoint(last, "s2ut") if last.is_file() else 0
    if previous and completed != previous[1]["updates"]:
        raise ValueError("snapshot update count differs from checkpoint")
    if native:
        parser_args = command[command.index("train") + 1:]
        # The same native CLI creates the scientific contract. No chunk-based
        # max_update rewrite: scheduler/budget remain those of the baseline.
        plan_command = command[:command.index("train")] + ["plan"] + parser_args
        plan = json.loads(subprocess.check_output(plan_command, text=True))
        total = plan["training_budget"]
        readers = plan["config"]["dataset"]["num_workers"]
    else:
        total = int(command[command.index("--max-updates") + 1])
        readers = 2
    config = {"kind": "s2ut" if native else "tt2", "checkpoint": relative,
              "command": list(command), "resume_args": [],
              "runtime": {"cache_gb": 8, "readers": max(1, readers), "precision": "default", "adaptive_batch": False}}
    from direct_s2st.translatotron2.monitor import absolute_update_limit
    if absolute_update_limit(config["kind"], total) != total:
        raise ValueError("inherited benchmark monitor environment would change the registered update budget")
    os.environ.setdefault("S2ST_DRIVE_SAFE", "1")
    os.environ.setdefault("S2ST_BACKUP_MIN_SECONDS", "600")
    os.environ.setdefault("S2ST_BACKUP_MAX_GB", "1024")
    with recovery_files():
        return run_continuous(config, work=work, backup=backup, identity=identity, completed=completed,
                              total=total, seconds=0, inspect_checkpoint=inspect_checkpoint, publish=bench.publish_packed)


def stage_session(command, work, backup, interval=600):
    """Publish only changed preprocessing files; unchanged pairs are never recopied.

    The benchmark's archive publisher/verifier/restorer are used unchanged.
    Incremental packs are needed because alignment adds per-pair artifacts.
    """
    bench = runtime()
    work, backup = session_paths(work, backup)
    identity = identity_for(command, work)
    receipts = sorted(backup.glob("*/snapshot.json"))
    versions, stamps, generation = {}, {}, 0
    for marker in receipts:
        state = read_json(marker)
        if state["identity"] != identity:
            raise ValueError("stage input/configuration changed; use a new backup directory")
        # Use the benchmark verifier on this individual incremental recovery point.
        with tempfile.TemporaryDirectory(dir=work.parent, prefix=".stage-restore-") as temp:
            restored = Path(temp) / "files"
            # latest() on all packs could hide a corrupt earlier pack; verify
            # each ZIP explicitly before delegating the unchanged extractor.
            from direct_s2st.drive_staging import raw_sha
            if raw_sha(marker.parent / "snapshot.zip") != state["zip_sha256"]:
                raise ValueError("stage archive checksum mismatch")
            bench.restore((marker.parent, state), restored)
            for name, sha in state["files"].items():
                source, target = restored / name, work / name
                if target.is_symlink() or not target.resolve().is_relative_to(work):
                    raise ValueError("stage restore escapes output")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                versions[name] = sha
                stat = target.stat()
                stamps[name] = (stat.st_size, stat.st_mtime_ns)
        generation = max(generation, state["updates"])
    complete = work / "stage-complete.json"
    if complete.is_file():
        if (read_json(complete).get("identity") == digest(identity) and
                versions.get("stage-complete.json") == file_hash(complete)):
            return {"status": "COMPLETE", "reused": True}
        # A failed upload may leave a local success marker. It is not proof of
        # durable progress. Resume/verify the stage and publish it again.
        complete.unlink()
    def publish(final=False):
        nonlocal generation
        changed = []
        output = Path(command[command.index("--output") + 1]).resolve() if "--output" in command else work
        file_output = output.suffix in (".json", ".jsonl", ".png")
        for path in sorted(work.rglob("*")):
            if path.is_symlink():
                raise ValueError("stage outputs must not be symlinks")
            if not path.is_file() or any(p.startswith(".") for p in path.relative_to(work).parts):
                continue
            if path.suffix == ".tmp" or path.name.endswith(".native.pt"):
                continue
            name = path.relative_to(work).as_posix()
            if file_output and path != output and path != complete and not name.startswith(output.stem + "-"):
                continue
            if path.suffix == ".npz" and not path.with_suffix(".sha256.json").is_file():
                continue
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
            if stamps.get(name) == stamp:
                continue
            sha = file_hash(path)
            if versions.get(name) != sha:
                changed.append((path, name, sha, stamp))
            else:
                stamps[name] = stamp
        if not changed:
            return
        with tempfile.TemporaryDirectory(dir=work.parent, prefix=".stage-freeze-") as temp:
            frozen = Path(temp)
            selected = {}
            for path, name, sha, stamp in changed:
                target = frozen / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if file_hash(target) != sha or file_hash(path) != sha:
                    if final:
                        raise ValueError("stage changed while publishing")
                    return  # Writer advanced; retry at the next boundary.
                selected[name] = sha
            generation += 1
            bench.publish_packed(frozen, backup, identity, generation)
            versions.update(selected)
            stamps.update({name: stamp for _, name, _, stamp in changed})
    process = subprocess.Popen(command)
    last = time.monotonic()
    try:
        while process.poll() is None:
            if time.monotonic() - last >= interval:
                publish()
                last = time.monotonic()
            time.sleep(.25)
        if process.returncode == 0:
            write_json(complete, {"identity": digest(identity)}, overwrite=True)
        publish(final=True)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return {"status": "COMPLETE", "packs": generation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    child = sub.add_parser("stage-inputs")
    child.add_argument("--benchmark", required=True)
    child.add_argument("--local", required=True)
    child.add_argument("--create-audio", action="store_true")
    child = sub.add_parser("run")
    child.add_argument("--work", required=True)
    child.add_argument("--backup", required=True)
    child.add_argument("--training", action="store_true")
    child.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "stage-inputs":
        result = stage_inputs(args.benchmark, args.local, args.create_audio)
    else:
        command = args.command[1:] if args.command and args.command[0] == "--" else args.command
        if not command:
            parser.error("a stage command is required")
        work, backup = Path(args.work).resolve(), Path(args.backup).resolve()
        if work.is_relative_to(backup) or backup.is_relative_to(work):
            raise ValueError("local work and backup must be separate")
        result = (train_session if args.training else stage_session)(command, work, backup)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
