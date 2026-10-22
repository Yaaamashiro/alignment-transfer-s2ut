"""Read-only checkpoint snapshot discovery and fail-closed CLI contracts."""
from pathlib import Path
import zipfile
import pytest
from alignment_transfer_s2ut.benchmark import restore_snapshot
from alignment_transfer_s2ut.io import digest, file_hash, write_json, read_json
from alignment_transfer_s2ut.cli import parser_for, run


def snapshot(root, updates):
    root.mkdir(parents=True)
    names = ["checkpoints/checkpoint_best.pt", "checkpoints/checkpoint_last.pt",
             "research-metadata.json", "s2ut-recipe-lock.json"]
    hashes = {}
    import hashlib
    with zipfile.ZipFile(root / "snapshot.zip", "w") as pack:
        for name in names:
            value = (name + str(updates)).encode()
            hashes[name] = hashlib.sha256(value).hexdigest()
            pack.writestr(name, value)
    write_json(root / "snapshot.json", {"storage": "zip-v1", "updates": updates,
        "zip_sha256": file_hash(root / "snapshot.zip"), "files": hashes})


def test_snapshot_restoration_is_verified_and_frozen_on_resume(tmp_path):
    backup = tmp_path / "backup"
    snapshot(backup / "first", 10)
    archive_hash = file_hash(backup / "first/snapshot.zip")
    value = restore_snapshot(backup, tmp_path / "restored")
    assert value["updates"] == 10 and (tmp_path / "restored/checkpoints/checkpoint_best.pt").is_file()
    snapshot(backup / "newer", 20)
    assert restore_snapshot(backup, tmp_path / "restored", resume=True) == value
    assert file_hash(backup / "first/snapshot.zip") == archive_hash
    with pytest.raises(ValueError, match="separate"):
        restore_snapshot(backup, backup / "derived")


def test_corrupt_or_unsafe_snapshots_are_rejected(tmp_path):
    snapshot(tmp_path / "source", 10)
    path = tmp_path / "source/snapshot.json"
    value = read_json(path)
    value["files"]["../escape"] = "0" * 64
    write_json(path, value, overwrite=True)
    with pytest.raises(ValueError, match="unsafe"):
        restore_snapshot(tmp_path / "source", tmp_path / "unsafe")
    value["files"].pop("../escape")
    write_json(path, value, overwrite=True)
    with (tmp_path / "source/snapshot.zip").open("ab") as stream:
        stream.write(b"corrupt archive")
    with pytest.raises(ValueError, match="checksum"):
        restore_snapshot(tmp_path / "source", tmp_path / "corrupt")


def test_speech_training_requires_registered_baseline_by_default():
    args = parser_for("train_s2ut").parse_args(["--manifest", "manifest", "--units", "units", "--output", "run"])
    assert args.trainer == "fairseq"
    with pytest.raises(ValueError, match="baseline-record"):
        run("train_s2ut", args)
    args = parser_for("train_s2ut").parse_args(["--manifest", "manifest", "--units", "units", "--output", "run",
        "--baseline-record", "baseline.json", "--warmup-updates", "400"])
    with pytest.raises(ValueError, match="unsupported overrides"):
        run("train_s2ut", args)


def test_native_notebook_keeps_baseline_and_binds_backbone():
    import ast
    import json
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           "notebooks/colab/benchmark_comparison.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    ast.parse(source)
    assert "RUN_TRANSFER = False" in source
    assert "s2ut-training-experiment-001-paper-fisher-v1" in source
    assert "baseline['backbone']" in source
    assert "test_native_fp16_trainer_and_fp32_prior" in source
    assert "--finetune-from-model" not in source
    assert "--max-updates" not in source[source.index("TRANSFER_ARGS ="):]
    assert "'compare', *records, '--output', Path(ROOT) / 'comparison/training-comparison.json', '--resume'" in source
