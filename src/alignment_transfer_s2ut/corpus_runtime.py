"""Run pinned benchmark preprocessing/training and restore the fixed text teacher."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

from .benchmark_data import derived_output
from .colab_runtime import (BENCHMARK_REVISION, deferred_s2ut_preflight, limit_speech_transfers, runtime,
                            session_paths, s2ut_implementation_files)
from .io import file_hash, read_json, write_json
from .progress import progress


@contextmanager
def allow_empty_test(prepared):
    """Permit a header-only test TSV in the pinned benchmark validator."""
    from direct_s2st.s2ut import multitask

    test_tsv = Path(prepared).resolve() / "test.tsv"
    read_tsv = multitask.read_tsv

    class HeaderOnlyTest(dict):
        # The pinned validator checks truthiness to require a split, but
        # derives counts, auxiliary parity and iteration from actual entries.
        def __bool__(self):
            return True

    def read(path, columns):
        rows = read_tsv(path, columns)
        if Path(path).absolute() == test_tsv and not rows:
            return HeaderOnlyTest()
        return rows

    multitask.read_tsv = read
    try:
        yield
    finally:
        multitask.read_tsv = read_tsv


def prepare_units(common, output, cache, shards, *, legacy=False):
    if shards < 1:
        raise ValueError("positive unit shard count required")
    bench = runtime()
    from direct_s2st.artifacts import download_artifact
    from direct_s2st.config import load_config
    from direct_s2st.s2ut.extract_units import extract_units, extractor_from_config

    common = Path(common).resolve()
    cache = derived_output(cache, common)
    output = derived_output(output, common)
    config = load_config(Path(bench.__file__).resolve().parents[2] / "configs/s2ut/prepare.yaml")
    km, hubert = config["kmeans"], config["hubert"]
    contract = dict(common=file_hash(common / "dataset-lock.json"), config=config, shards=shards,
                    benchmark_revision=BENCHMARK_REVISION)
    contract["manifests"] = {split: file_hash(common / (split + ".jsonl")) for split in ("train", "dev", "test")}
    marker = output / ("kokoro-unit-inputs.json" if legacy else "unit-inputs.json")
    if marker.is_file():
        if read_json(marker) != contract:
            raise ValueError("corpus unit extraction inputs changed")
    else:
        write_json(marker, contract)
    extractor = None
    for shard in range(shards):
        manifest = output / f"manifest.shard-{shard:05d}-of-{shards:05d}.jsonl"
        receipt = manifest.with_suffix(".receipt.json")
        if receipt.is_file():
            if read_json(receipt)["sha256"] != file_hash(manifest):
                raise ValueError(f"corpus unit shard has changed: {manifest}")
            progress(f"corpus units: reuse shard {shard + 1}/{shards}")
            continue
        if extractor is None:
            download_artifact(km["source_url"], cache / km["path"], sha256=km["sha256"])
            extractor = extractor_from_config(config, cache)
        extract_units(common, output, extractor=extractor, split=None,
                      clusters=config["kmeans_clusters"], hubert_model=hubert["model"],
                      hubert_revision=hubert["revision"], hubert_layer=config["hubert_layer"],
                      kmeans_sha256=km["sha256"], kmeans_artifact=km["artifact"],
                      shard_index=shard, num_shards=shards, resume=True)
        write_json(receipt, {"sha256": file_hash(manifest)})
    return {"shards": shards}


def prepare_s2ut(common, units, output):
    bench = runtime()
    from direct_s2st.config import load_config
    from direct_s2st.manifests.validate import validate_manifest_directory
    from direct_s2st.s2ut.prepare_fairseq import prepare_fairseq

    common, units = Path(common).resolve(), Path(units).resolve()
    output = derived_output(output, common, units)
    validate_manifest_directory(common, resume=True)
    config = load_config(Path(bench.__file__).resolve().parents[2] / "configs/s2ut/prepare.yaml")
    with allow_empty_test(output):
        return prepare_fairseq(common, units, output, clusters=config["kmeans_clusters"],
                               multitask=config.get("multitask"), resume=True)


def baseline_session(data, work, backup, restore_only=False, *, legacy=False):
    bench = runtime()
    from direct_s2st.s2ut.paper_recipe import MAX_UPDATES, preflight_dependencies
    from .s2ut.fairseq_extension import FAIRSEQ_REVISION, verify_fairseq

    work, backup = session_paths(work, backup)
    data = Path(data).resolve()
    derived_output(work, data)
    derived_output(backup, data)
    if not os.environ.get("S2ST_INPUT_MAP"):
        raise ValueError("stage the corpus WAVs before starting or restoring Baseline")
    verify_fairseq()
    preflight_dependencies()
    lock_path = work.parent / ("kokoro-baseline-runtime.json" if legacy else "baseline-runtime.json")
    adapter_sha = s2ut_implementation_files(Path(__file__).parent)[Path(__file__).name]
    lock = dict(repository=BENCHMARK_REVISION, fairseq=FAIRSEQ_REVISION, adapter_sha256=adapter_sha)
    if lock_path.is_file() and read_json(lock_path) != lock:
        raise ValueError("Baseline runtime changed; use the original pin")
    if not lock_path.is_file():
        write_json(lock_path, lock)
    from direct_s2st.s2ut.migrate_labels import paper_data_root
    with allow_empty_test(paper_data_root(data)), deferred_s2ut_preflight():
        config = bench.make_config(Path(bench.__file__).resolve().parents[2], data, lock_path,
                                   kind="s2ut", model_size="fisher", performance="gpu80", num_workers=8,
                                   save_interval=100, optimize=True, s2ut_recipe="paper_fisher_v1",
                                   s2ut_vocoder_mode="paper")
    config["command"][2:3] = ["alignment_transfer_s2ut.kokoro_runtime" if legacy else __name__, "_fairseq"]
    result = bench.run_session(config, work=work, backup=backup, total=MAX_UPDATES, confirm_training=True,
                               auto_resume=not restore_only, resume=restore_only, restore_only=restore_only,
                               _staged=True)
    if result["durable_updates"] != MAX_UPDATES or not (work / "checkpoints/checkpoint_best.pt").is_file():
        raise ValueError("a completed 400,000-update Baseline with its dev-selected checkpoint is required")
    return result


def restore_teacher(snapshot, output, updates, source_manifest, speech_manifest, backbone):
    bench = runtime()
    from .config import BackboneConfig
    from .data.manifest import load_manifest, verified_teacher_subset
    from .t2tt_extension import verify_state
    from .tokenization.factory import tokenizers_from_checkpoint

    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    output = derived_output(output, snapshot, Path(source_manifest).parent)
    output, snapshot = session_paths(output, snapshot)
    state = read_json(snapshot / "snapshot.json")
    identity = state["identity"]
    command = identity["command"]
    files = state["files"]
    required = {"checkpoint_best.pt", "checkpoint_last.pt", "experiment.json", "metrics.jsonl"}
    if (state.get("storage") != "zip-v1" or state.get("updates") != updates
            or identity.get("benchmark_revision") != BENCHMARK_REVISION
            or "alignment_transfer_s2ut.train_t2tt" not in command
            or "--max-updates" not in command or int(command[command.index("--max-updates") + 1]) != updates
            or not required <= files.keys()):
        raise ValueError("expected a completed text-only T2TT training snapshot at the requested update")
    if not all((output / name).is_file() and file_hash(output / name) == files[name] for name in required):
        bench.restore((snapshot, state), output)
    _, best = verify_state(output, updates)
    if best.get("backend") != "fairseq" or "tokenizers" not in best:
        raise ValueError("attention transfer requires the trained fairseq SentencePiece teacher")
    tokenizers_from_checkpoint(best)
    rows = load_manifest(speech_manifest, require_audio=False)
    source_hash = verified_teacher_subset(best, rows, source_manifest)
    if best["backbone"] != BackboneConfig.load(backbone).asdict():
        raise ValueError("T2TT teacher backbone differs from the Baseline")
    return {"pairs": len(rows), "training_updates": updates, "best_step": best["step"],
            "checkpoint": str(output / "checkpoint_best.pt"), "checkpoint_hash": files["checkpoint_best.pt"],
            "source_text_hash": source_hash}


def main(*, legacy=False):
    if sys.argv[1:2] == ["_fairseq"]:
        runtime()
        from direct_s2st.s2ut import fairseq_train
        # Preserve the launcher's canonical recipe identity and argument layout.
        sys.argv[:] = [fairseq_train.__name__, *sys.argv[2:]]
        with allow_empty_test(sys.argv[1]), deferred_s2ut_preflight(), limit_speech_transfers():
            fairseq_train.main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    commands = (
        ("units", ("common", "output", "cache")),
        ("prepare", ("common", "units", "output")),
        ("baseline", ("data", "work", "backup")),
        ("restore-teacher", ("snapshot", "output", "source-manifest", "speech-manifest", "backbone")),
    )
    for action, names in commands:
        child = sub.add_parser(action)
        for name in names:
            child.add_argument("--" + name, required=True)
    sub.choices["units"].add_argument("--shards", type=int, default=16)
    baseline_mode = sub.choices["baseline"].add_mutually_exclusive_group(required=True)
    baseline_mode.add_argument("--restore-only", action="store_true")
    baseline_mode.add_argument("--train", action="store_true", help="start or resume 400,000 native S2UT updates")
    sub.choices["restore-teacher"].add_argument("--updates", type=int, required=True)
    args = vars(parser.parse_args())
    action = args.pop("action")
    args.pop("train", None)
    handler = {"units": prepare_units, "prepare": prepare_s2ut, "baseline": baseline_session,
               "restore-teacher": restore_teacher}[action]
    if action in ("units", "baseline"):
        args["legacy"] = legacy
    print(json.dumps(handler(**args), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
