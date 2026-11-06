"""Adapters to the benchmark's pinned Drive staging and recovery implementation."""
import argparse
from contextlib import contextmanager
import errno
import importlib
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

from .io import digest, file_hash, read_json, write_json
from .progress import progress

# Artifact/recipe lineage stays unchanged for this I/O-only runtime patch.
BENCHMARK_REVISION = "4361e55a99c7f4ea3a00c5c138f6820c34a07d06"
BENCHMARK_RUNTIME_REVISION = "0c0024ea888206149a3119bf096b86ba66d63b66"
EXTRA_RECOVERY = ("native-run.json", "native-dev.jsonl", "alignment-train.jsonl", "checkpoint_best.pt", "experiment.json", "metrics.jsonl",
                  "tokenization.json", "tokenizers/ja.model", "tokenizers/ja.vocab", "tokenizers/en.model", "tokenizers/en.vocab")


@lru_cache(maxsize=1)
def runtime():
    import direct_s2st
    root = Path(direct_s2st.__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != BENCHMARK_RUNTIME_REVISION:
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


def s2ut_implementation_files(package):
    """Map the exact reviewed storage-only patch to its prior S2UT identity."""
    package = Path(package)
    sources = {p.relative_to(package).as_posix(): file_hash(p) for p in package.rglob("*.py")}
    compatibility = package / "streaming-compatibility.json"
    if compatibility.is_file():
        approved = read_json(compatibility)
        for variant in approved:
            if all(sources.get(name) == value for name, value in variant["current"].items()):
                sources.update(variant["previous"])
                break
    return sources


def stream_audio_mapping():
    if not os.environ.get("S2ST_INPUT_MAP"):
        return None
    from direct_s2st.drive_staging import mapping, stream_document
    _, stream = stream_document()
    return mapping() if stream is not None and stream["body"].get("s2ut_audio") else None


@contextmanager
def deferred_s2ut_preflight():
    """Validate the complete prepared index while deferring verified WAV reads."""
    rows = stream_audio_mapping()
    if rows is None:
        yield
        return
    from direct_s2st.s2ut import preflight
    original = preflight.validate_prepared

    def validate(root, *, clusters=100, check_audio=True):
        if check_audio:
            columns = ("id", "src_audio", "src_n_frames", "tgt_audio", "tgt_n_frames")
            for split in ("train", "dev", "test"):
                prepared = preflight.read_tsv(root / (split + ".tsv"), columns)
                for row in prepared.values():
                    if os.path.abspath(row["src_audio"]) not in rows:
                        raise ValueError("S2UT source audio is absent from the ZIP index: " + row["src_audio"])
        return original(root, clusters=clusters, check_audio=False)

    preflight.validate_prepared = validate
    try:
        yield
    finally:
        preflight.validate_prepared = original


@contextmanager
def limit_speech_transfers():
    """Share two Drive transfer slots across the forked fairseq loader workers."""
    if stream_audio_mapping() is None:
        yield
        return
    from direct_s2st.stream_inputs import Downloader
    from filelock import FileLock, Timeout
    original = Downloader.transfer

    @contextmanager
    def transfer(manager, background):
        with original(manager, background):
            while True:
                for slot in range(2):
                    lock = FileLock(str(manager.root / f"transfer-{slot}.lock"))
                    try:
                        lock.acquire(timeout=0)
                    except Timeout:
                        continue
                    try:
                        yield
                    finally:
                        lock.release()
                    return
                time.sleep(.05)

    Downloader.transfer = transfer
    try:
        yield
    finally:
        Downloader.transfer = original


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
        with limit_speech_transfers():
            yield
    finally:
        Dictionary.load, creator._load_samples_from_tsv = dict_load, tsv_load
        data_cfg.get_config_from_yaml, audio.get_features_or_waveform = yaml_load, audio_load


def stage_inputs(benchmark, local, create_audio=False, audio_pack_workers=2):
    """Use benchmark WAV archives verbatim; GPU sessions never build missing packs."""
    runtime()
    from direct_s2st.drive_staging import stage_audio, ensure_local, copy_verified, register, active_map
    from direct_s2st.preparation import file_stamp
    benchmark, local = Path(benchmark).resolve(), ensure_local(local)
    common = benchmark / "data/common"
    rows = stage_audio(common, common.parent / ".drive-audio-packs", local / "audio",
                       splits=("train", "dev"), languages=("ja", "en"), create=create_audio,
                       materialize=not create_audio, workers=audio_pack_workers if create_audio else 1)
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


def stage_kokoro(common, corpus_root, packs, local, create_audio=False, audio_pack_workers=2):
    return stage_corpus(common, corpus_root, packs, local, create_audio, audio_pack_workers)


def index_corpus_audio(common, corpus_root, packs, local):
    """Build the benchmark Downloader's audio index without opening Drive ZIPs."""
    from direct_s2st.drive_staging import audio_archive_path, raw_sha
    from direct_s2st.io import read_jsonl
    from direct_s2st.journal import digest as bench_digest
    from direct_s2st.stream_inputs import checksum
    from direct_s2st.translatotron2.mel_storage import guarded_root
    packs = guarded_root(packs)
    rows, jobs = {}, {}
    for split in ("train", "dev"):
        manifest = Path(common) / (split + ".jsonl")
        items = list(read_jsonl(manifest))
        manifest_sha = raw_sha(manifest)
        for language in ("ja", "en"):
            key = language + "_audio"
            for start in range(0, len(items), 128):
                selected = items[start:start + 128]
                archive = audio_archive_path(packs, manifest_sha, split, language, start, 128)
                job = bench_digest(["audio", str(archive), manifest_sha])[:24]
                directory = local / "audio" / job
                pairs, members = {}, {}
                for row in selected:
                    pair = row["pair_id"]
                    sha = checksum(row[language + "_sha256"])
                    if pair in pairs:
                        raise ValueError("duplicate streaming audio ID: " + pair)
                    pairs[pair] = sha
                    member = bench_digest([pair, language])[:32] + ".wav"
                    members[member] = sha
                    aliases = [Path(row[key])]
                    if relative := row.get(key + "_corpus_relative"):
                        relative = Path(relative)
                        if relative.is_absolute() or relative.drive or ".." in relative.parts:
                            raise ValueError("unsafe corpus-relative audio path")
                        aliases.append(corpus_root / relative)
                    for alias in aliases:
                        if not alias.is_absolute() or ".." in alias.parts:
                            raise ValueError("corpus audio index requires safe absolute paths")
                        name = os.path.abspath(alias)
                        entry = dict(local=str(directory / member), job=job, member=member, wav_sha256=sha)
                        previous = rows.setdefault(name, entry)
                        if previous["wav_sha256"] != sha:
                            raise ValueError("conflicting ZIP hashes for " + name)
                jobs[job] = dict(kind="audio", source=str(archive), target=str(directory),
                                 pairs=pairs, members=members)
    if not jobs:
        raise ValueError("empty corpus audio index")
    body = dict(root=str(local), jobs=jobs, s2ut_audio=True)
    progress("corpus WAV index", detail=f"packs={len(jobs)} payloads_downloaded=0")
    return rows, dict(body=body, sha256=bench_digest(body))


def stage_corpus(common, corpus_root, packs, local, create_audio=False, audio_pack_workers=2, *, stream_audio=False):
    """Reuse the pinned benchmark's WAV ZIP receipt, staging and input map."""
    from .benchmark_data import derived_output
    runtime()
    from direct_s2st.drive_staging import active_map, capacity, ensure_local, stage_audio
    corpus_root = Path(corpus_root).resolve()
    packs = derived_output(packs, corpus_root, common)
    local = ensure_local(derived_output(local, corpus_root, common))
    lock = read_json(Path(common) / "dataset-lock.json")
    if lock.get("audio_path_policy") not in {"kokoro-audio-16k-v1", "corpus-relative-v1"}:
        raise ValueError("corpus staging requires imported common manifests")
    old_root = os.environ.get("CORPUS_ROOT")
    if old_root is not None and Path(old_root).resolve() != corpus_root:
        raise ValueError("CORPUS_ROOT differs from the staging root")
    if stream_audio:
        if create_audio:
            raise ValueError("streaming requires existing Drive WAV packs")
        rows, stream = index_corpus_audio(common, corpus_root, packs, local)
        with active_map(rows, local, stream=stream) as name:
            return {"input_map": str(name), "files": len(rows), "mode": "on-demand"}
    if not create_audio:
        wav_bytes = lock["wav_bytes_estimate"]
        progress("corpus audio staging", detail=(
            f"WAV payload={wav_bytes / 2**30:.1f} GiB, VM free={shutil.disk_usage(local).free / 2**30:.1f} GiB. "
            "checkpoints and derived artifacts require additional space"))
        if not (local / "input-map.json").is_file():
            capacity(local, wav_bytes)
    os.environ["CORPUS_ROOT"] = str(corpus_root)
    try:
        rows = stage_audio(common, packs, local / "audio", splits=("train", "dev"),
                           languages=("ja", "en"), create=create_audio, materialize=not create_audio, workers=audio_pack_workers if create_audio else 1)
    finally:
        if old_root is None:
            os.environ.pop("CORPUS_ROOT")
    if create_audio:
        return {"status": "PACKED"}
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


def restore_native_training(work, backup, method):
    """Restore a completed native comparison run without launching a trainer."""
    from .benchmark import load_native, read_record

    bench = runtime()
    work, backup = session_paths(work, backup)
    markers = sorted(backup.glob("*/snapshot.json"), reverse=True)
    if not markers:
        raise FileNotFoundError("no completed training snapshot at " + str(backup))
    identity = read_json(markers[0])["identity"]
    command = identity.get("command", [])
    def option(name):
        return command[command.index(name) + 1] if name in command else None
    if (identity.get("benchmark_revision") != BENCHMARK_REVISION or identity.get("work") != str(work)
            or "alignment_transfer_s2ut.benchmark" not in command or "train" not in command
            or option("--method") != method or option("--output") != str(work)):
        raise ValueError("training snapshot identity differs from the requested comparison run")
    snapshot = bench.latest(backup, identity)
    if snapshot is None:
        raise ValueError("no verified native training snapshot at " + str(backup))
    required = {"native-run.json", "checkpoints/checkpoint_last.pt", "checkpoints/checkpoint_best.pt"}
    if not required <= snapshot[1]["files"].keys():
        raise ValueError("native snapshot lacks plan or last/best checkpoint")
    bench.restore(snapshot, work)
    plan = read_json(work / "native-run.json")
    if (plan.get("contract_hash") != digest({k: v for k, v in plan.items() if k != "contract_hash"})
            or plan["transfer"]["method"] != method or snapshot[1]["updates"] != plan["training_budget"]):
        raise ValueError("native training budget is incomplete or its contract changed")
    baseline = read_record(plan["baseline_record"])
    if baseline["record_hash"] != plan["baseline_record_hash"]:
        raise ValueError("native training baseline registration differs")
    for name in ("checkpoint_last.pt", "checkpoint_best.pt"):
        state = load_native(work / "checkpoints" / name)
        updates = int(state["optimizer_history"][-1]["num_updates"])
        if (state.get("extra_state", {}).get("alignment_contract_hash") != plan["contract_hash"]
                or not 0 < updates <= plan["training_budget"]
                or name == "checkpoint_last.pt" and updates != plan["training_budget"]):
            raise ValueError("restored native checkpoint differs from its training contract")
    return {"status": "RESTORED", "updates": plan["training_budget"], "method": method}


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
    inputs = list(command)
    if any(module in command for module in ("alignment_transfer_s2ut.kokoro_runtime",
                                           "alignment_transfer_s2ut.corpus_runtime")) and "--common" in command:
        common = Path(command[command.index("--common") + 1])
        inputs += [str(common / name) for name in ("dataset-lock.json", "train.jsonl", "dev.jsonl", "test.jsonl")]
        if "--units" in command:
            units = Path(command[command.index("--units") + 1])
            inputs += [str(units / "unit-lock.json"), *map(str, sorted(units.glob("*.receipt.json")))]
    for item in inputs:
        path = Path(item)
        try:
            is_file = path.is_file()
        except OSError as error:
            # Arguments can be inline code or other long text. Linux raises
            # ENAMETOOLONG here, while some platforms return False. The entire
            # argument remains bound by identity['command']; it is not a file.
            if error.errno != errno.ENAMETOOLONG:
                raise
            is_file = False
        if is_file and not path.resolve().is_relative_to(output):
            files[str(path.resolve())] = file_hash(path)
    return {"command": list(map(str, command)), "inputs": files, "work": str(output),
            "benchmark_revision": BENCHMARK_REVISION}


def train_session(command, work, backup, *, popen=None, publish=None):
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
                              total=total, seconds=0, inspect_checkpoint=inspect_checkpoint, publish=publish or bench.publish_packed,
                              **({"popen":popen} if popen is not None else {}))


def stage_session(command, work, backup, interval=600, *, restore_only=False):
    """Publish only changed preprocessing files; unchanged pairs are never recopied.

    The benchmark's archive publisher/verifier/restorer are used unchanged.
    Incremental packs are needed because alignment adds per-pair artifacts.
    """
    bench = runtime()
    work, backup = session_paths(work, backup)
    identity = identity_for(command, work)
    pair_outputs = any(module in command for module in (
        "alignment_transfer_s2ut.extract_t2tt_attention", "alignment_transfer_s2ut.align_speech_text",
        "alignment_transfer_s2ut.extract_units", "alignment_transfer_s2ut.build_teacher_alignment")) or any(
        list(command[i:i + 2]) == ["alignment_transfer_s2ut.benchmark_data", "units"] for i in range(len(command) - 1))
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
        state = read_json(complete)
        if (state.get("identity") == digest(identity) and
                (not pair_outputs or state.get("artifact_receipts_verified") is True) and
                versions.get("stage-complete.json") == file_hash(complete)):
            return {"status": "COMPLETE", "reused": True}
        if restore_only:
            raise ValueError(f"no completed, verified stage at {backup}")
        if pair_outputs and state.get("artifact_receipts_verified") is not True:
            progress("stage resume: revalidate legacy pair outputs before trusting completion")
        # A failed upload may leave a local success marker. It is not proof of
        # durable progress. Resume/verify the stage and publish it again.
        complete.unlink()
    if restore_only:
        raise ValueError(f"no completed, verified stage at {backup}")
    def publish(final=False, require_complete=False):
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
                if require_complete:
                    raise ValueError("missing artifact checksum at successful stage completion: " + name)
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
            write_json(complete, {"identity": digest(identity), "artifact_receipts_verified": True}, overwrite=True)
        publish(final=True, require_complete=process.returncode == 0)
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
    child.add_argument("--audio-pack-workers", type=int, choices=range(1, 5), default=2)
    child = sub.add_parser("restore-training")
    for name in ("work", "backup"):
        child.add_argument("--" + name, required=True)
    child.add_argument("--method", required=True,
                       choices=("distillation", "annealed_distillation", "mask_distillation"))
    child = sub.add_parser("stage-corpus", aliases=["stage-kokoro"])
    for name in ("common", "corpus-root", "packs", "local"):
        child.add_argument("--" + name, required=True)
    child.add_argument("--create-audio", action="store_true")
    child.add_argument("--audio-pack-workers", type=int, choices=range(1, 5), default=2)
    child.add_argument("--stream-audio", action="store_true", help="index existing WAV ZIPs for on-demand reads")
    child = sub.add_parser("run")
    child.add_argument("--work", required=True)
    child.add_argument("--backup", required=True)
    child.add_argument("--training", action="store_true")
    child.add_argument("--restore-only", action="store_true", help="restore a completed preprocessing stage without running its command")
    child.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "stage-inputs":
        result = stage_inputs(args.benchmark, args.local, args.create_audio, args.audio_pack_workers)
    elif args.action == "restore-training":
        result = restore_native_training(args.work, args.backup, args.method)
    elif args.action in {"stage-corpus", "stage-kokoro"}:
        result = stage_corpus(args.common, args.corpus_root, args.packs, args.local, args.create_audio,
                              args.audio_pack_workers, stream_audio=args.stream_audio)
    else:
        command = args.command[1:] if args.command and args.command[0] == "--" else args.command
        if not command:
            parser.error("a stage command is required")
        work, backup = Path(args.work).resolve(), Path(args.backup).resolve()
        if work.is_relative_to(backup) or backup.is_relative_to(work):
            raise ValueError("local work and backup must be separate")
        if args.training and args.restore_only:
            parser.error("--restore-only is for preprocessing stages")
        result = (train_session(command, work, backup) if args.training else
                  stage_session(command, work, backup, restore_only=args.restore_only))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
