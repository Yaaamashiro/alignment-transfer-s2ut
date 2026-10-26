"""Use the real benchmark archive/staging code; no alternate Drive implementation."""
import json
import importlib.util
import errno
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import soundfile as sf

from alignment_transfer_s2ut import colab_runtime as c
from alignment_transfer_s2ut.io import digest, file_hash, read_json, write_json, write_jsonl


@pytest.fixture
def benchmark_runtime(monkeypatch):
    # Colab installs the immutable runtime separately. An unrelated sibling
    # checkout may have advanced and must not shadow that installed version.
    configured = os.environ.get("S2ST_BENCHMARK_TEST_ROOT")
    source = (Path(configured) / "src" if configured else
              Path(__file__).resolve().parents[2] / "ja-en-direct-s2st-benchmark/src")
    if configured or importlib.util.find_spec("direct_s2st") is None:
        if not source.is_dir():
            if configured:
                pytest.fail("S2ST_BENCHMARK_TEST_ROOT must contain the pinned benchmark src directory")
        else:
            monkeypatch.syspath_prepend(str(source))
            monkeypatch.setenv("PYTHONPATH", str(source) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    pytest.importorskip("direct_s2st")
    c.runtime.cache_clear()
    value = c.runtime()
    monkeypatch.setenv("S2ST_BACKUP_MAX_GB", "1")
    yield value
    c.runtime.cache_clear()


def test_command_identity_binds_long_text_and_external_file_inputs(tmp_path, monkeypatch):
    code = "print('inline code')\n" * 100
    config = tmp_path / "config.json"
    write_json(config, {"seed": 1})
    work = tmp_path / "work"
    write_json(work / "result.json", {"complete": True})
    original = Path.is_file
    def is_file(path):
        if str(path) == code:
            raise OSError(errno.ENAMETOOLONG, "File name too long")
        return original(path)
    monkeypatch.setattr(Path, "is_file", is_file)
    command = [sys.executable, "-c", code, str(config), str(work / "result.json")]
    identity = c.identity_for(command, work)
    assert identity["command"] == command
    assert identity["inputs"][str(config.resolve())] == file_hash(config)
    assert str((work / "result.json").resolve()) not in identity["inputs"]
    changed = c.identity_for([*command[:2], code + "# changed", *command[3:]], work)
    assert digest(changed) != digest(identity)


def test_command_identity_does_not_hide_file_access_errors(tmp_path, monkeypatch):
    def denied(path):
        raise PermissionError(errno.EACCES, "Permission denied")
    monkeypatch.setattr(Path, "is_file", denied)
    with pytest.raises(PermissionError):
        c.identity_for(["config.json"], tmp_path / "work")


def test_stage_backup_recovers_partial_pairs_and_skips_completed_stage(tmp_path, benchmark_runtime, monkeypatch):
    work, backup = tmp_path / "work", tmp_path / "drive"
    script = """import os,sys,numpy as np
from pathlib import Path
from alignment_transfer_s2ut.io import ArtifactStore
s=ArtifactStore(sys.argv[1])
for key in ['a','b']:
    if not s.path(key).is_file(): s.put(key, {'stage':'test'}, {'array':np.ones(3)})
if os.environ.get('TEST_INTERRUPT')=='1': raise SystemExit(7)
"""
    command = [sys.executable, "-c", script, str(work)]
    monkeypatch.setenv("TEST_INTERRUPT", "1")
    with pytest.raises(subprocess.CalledProcessError):
        c.stage_session(command, work, backup)
    markers = list(backup.glob("*/snapshot.json"))
    assert len(markers) == 1
    first = read_json(markers[0])
    assert len(first["files"]) == 4 and first["storage"] == "zip-v1"
    hashes = dict(first["files"])
    work.rename(tmp_path / "lost-vm")
    monkeypatch.setenv("TEST_INTERRUPT", "0")
    assert c.stage_session(command, work, backup)["status"] == "COMPLETE"
    assert all(file_hash(work / name) == sha for name, sha in hashes.items())
    markers = sorted(backup.glob("*/snapshot.json"))
    assert len(markers) == 2
    assert set(read_json(markers[1])["files"]) == {"stage-complete.json"}
    work.rename(tmp_path / "second-vm")
    assert c.stage_session(command, work, backup)["reused"]
    assert len(list(backup.glob("*/snapshot.json"))) == 2
    with (markers[0].parent / "snapshot.zip").open("ab") as stream:
        stream.write(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        c.stage_session(command, work, backup)


def test_benchmark_audio_packs_work_without_original_wavs(tmp_path, benchmark_runtime, monkeypatch):
    from direct_s2st.drive_staging import stage_audio, active_map
    from alignment_transfer_s2ut.data.audio import waveform
    common, packs = tmp_path / "common", tmp_path / "packs"
    originals = tmp_path / "originals"
    originals.mkdir()
    rows = []
    for i in range(2):
        path = originals / f"{i}.wav"
        sf.write(path, np.zeros(800), 16000)
        rows.append({"pair_id": str(i), "ja_audio": str(path), "ja_sha256": file_hash(path)})
    write_jsonl(common / "train.jsonl", rows)
    stage_audio(common, packs, tmp_path / "cpu", splits=("train",), languages=("ja",), materialize=False)
    originals.rename(tmp_path / "originals-offline")
    mapping = stage_audio(common, packs, tmp_path / "gpu", splits=("train",), languages=("ja",), create=False)
    with active_map(mapping, tmp_path / "map"):
        assert waveform(rows[0]["ja_audio"])[0].numel() == 800
    assert len(list(packs.rglob("*.zip"))) == 1


def test_pair_backup_capacity_failure_preserves_recovery_point(tmp_path, benchmark_runtime, monkeypatch):
    work, backup = tmp_path / "work", tmp_path / "drive"
    command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(exist_ok=True); (Path(sys.argv[1])/'result').write_text('complete')", str(work)]
    c.stage_session(command, work, backup)
    before = {str(p): file_hash(p) for p in backup.rglob("*") if p.is_file()}
    monkeypatch.setenv("S2ST_BACKUP_MAX_GB", "0.000000001")
    with pytest.raises(RuntimeError, match="budget"):
        benchmark_runtime.publish_packed(work, backup, {}, 2)
    assert all(file_hash(path) == sha for path, sha in before.items())
    monkeypatch.setenv("CORPUS_ROOT", str(tmp_path / "corpus"))
    with pytest.raises(ValueError, match="CORPUS_ROOT"):
        c.session_paths(tmp_path / "local", tmp_path / "corpus/derived")


def test_colab_uses_local_outputs_and_benchmark_pin():
    import ast
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/colab/benchmark_comparison.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    ast.parse(source)
    assert "ROOT = '/content/alignment-transfer-work/" in source
    assert "PERSISTENT = '/content/drive/" in source
    assert c.BENCHMARK_REVISION in source
    assert "alignment_transfer_s2ut.colab_runtime" in source
    assert "S2ST_INPUT_MAP" in source and "'600'" in source and "'1024'" in source


def test_resident_t2tt_upload_and_restore_uses_benchmark_session(tmp_path, benchmark_runtime, monkeypatch):
    from alignment_transfer_s2ut.smoke_pipeline import TINY_CONFIG
    work, backup = tmp_path / "teacher", tmp_path / "drive"
    manifest, config = tmp_path / "manifest.jsonl", tmp_path / "tiny.json"
    write_json(config, TINY_CONFIG)
    write_jsonl(manifest, [dict(pair_id=str(i), split="train" if i < 2 else "dev",
        ja_text="日本語" + str(i), en_text="text" + str(i),
        ja_audio=str(tmp_path / f"ja{i}.wav"), en_audio=str(tmp_path / f"en{i}.wav")) for i in range(3)])
    command = [sys.executable, "-m", "alignment_transfer_s2ut.train_t2tt", "--manifest", str(manifest),
        "--config", str(config), "--output", str(work), "--backend", "smoke", "--max-updates", "2",
        # One final immutable snapshot makes the Windows integration test avoid
        # replacing a progress file while the upstream supervisor reads it.
        # Native checkpoint hooks independently cover periodic publication.
        "--validation-interval", "2", "--save-interval-updates", "2", "--resume-if-exists"]
    result = c.train_session(command, work, backup)
    assert result == {"status": "COMPLETE", "durable_updates": 2}
    assert len(list(backup.glob("*/snapshot.json"))) >= 1
    checkpoint_hash = file_hash(work / "checkpoint_last.pt")
    work.rename(tmp_path / "lost-training-vm")
    assert c.train_session(command, work, backup)["durable_updates"] == 2
    assert file_hash(work / "checkpoint_last.pt") == checkpoint_hash
    assert (work / "checkpoint_best.pt").is_file() and (work / "experiment.json").is_file()


def test_failed_final_upload_is_not_reported_as_durable_completion(tmp_path, benchmark_runtime, monkeypatch):
    work, backup = tmp_path / "work", tmp_path / "drive"
    command = [sys.executable, "-c", "from pathlib import Path; import sys; (Path(sys.argv[1])/'result').write_text('complete')", str(work)]
    publisher = benchmark_runtime.publish_packed
    def failed(*args, **kwargs):
        raise OSError("Drive unavailable")
    monkeypatch.setattr(benchmark_runtime, "publish_packed", failed)
    with pytest.raises(OSError, match="Drive unavailable"):
        c.stage_session(command, work, backup)
    assert (work / "stage-complete.json").is_file() and not list(backup.glob("*/snapshot.json"))
    monkeypatch.setattr(benchmark_runtime, "publish_packed", publisher)
    assert not c.stage_session(command, work, backup).get("reused", False)
    assert len(list(backup.glob("*/snapshot.json"))) == 1
