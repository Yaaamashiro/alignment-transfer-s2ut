"""Real tiny continuations through the pinned ZIP/runtime and original trainer."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

from alignment_transfer_s2ut import colab_runtime as c, t2tt_extension as e
from alignment_transfer_s2ut.io import file_hash, read_json, write_json, write_jsonl
from alignment_transfer_s2ut.smoke_pipeline import TINY_CONFIG, invoke
from alignment_transfer_s2ut.training import load_checkpoint
from test_benchmark_colab import benchmark_runtime


@pytest.fixture
def source_run(tmp_path_factory, benchmark_runtime, monkeypatch, request):
    # In-process reference and fresh CLI workers must use the same CPU reduction
    # order. torch.set_num_threads alone does not configure spawned interpreters.
    previous_threads = torch.get_num_threads()
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    root = tmp_path_factory.mktemp("ext")
    manifest, config = root / "manifest.jsonl", root / "backbone.yaml"
    write_jsonl(manifest, [dict(pair_id=str(i), split="train" if i < 3 else "dev", ja_text="日本語" + str(i),
        en_text="text" + str(i), ja_audio=str(root / f"ja{i}.wav"), en_audio=str(root / f"en{i}.wav")) for i in range(4)])
    config.write_text(yaml.safe_dump(TINY_CONFIG), encoding="utf-8")
    source_work, source_backup = root / "base", root / "drive/base"
    arguments = ["--manifest", str(manifest), "--config", str(config), "--backend", "smoke",
                 "--batch-size", "1", "--update-frequency", "1", "--loader-workers", "0",
                 "--validation-interval", "2", "--save-interval-updates", "1", "--resume-if-exists"]
    invoke("train_t2tt", *arguments, "--output", source_work, "--max-updates", 2)
    command = [sys.executable, "-m", "alignment_transfer_s2ut.train_t2tt", *arguments,
               "--output", str(source_work), "--max-updates", "2"]
    identity = c.identity_for(command, source_work)
    snapshot = benchmark_runtime.publish_packed(source_work, source_backup, identity, 2)
    monkeypatch.setenv("S2ST_DRIVE_SAFE", "1")
    monkeypatch.setenv("S2ST_BACKUP_MIN_SECONDS", "0")
    return dict(root=root, source_work=source_work, source_backup=source_backup, manifest=manifest, config=config,
                work=root / "extended", backup=root / "drive/extended", arguments=arguments, snapshot=snapshot)


def call(source, **extra):
    arguments = {k: source[k] for k in ("source_work", "source_backup", "work", "backup", "manifest", "config")}
    arguments.update(source_updates=2, target_updates=4, snapshot_updates=2, milestones=[4])
    arguments.update(extra)
    return e.extend_train_session(**arguments)


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_state(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_state(x, y)
    else:
        assert a == b


def test_reference_and_fresh_worker_use_identical_cpu_threads(source_run):
    from alignment_transfer_s2ut.process_logging import run_process
    result = run_process([sys.executable, "-c",
        "import json,os,torch; print(json.dumps(dict(threads=torch.get_num_threads(), "
        "omp=os.environ.get('OMP_NUM_THREADS'), mkl=os.environ.get('MKL_NUM_THREADS'))))"],
        source_run["root"] / "test-logs", name="test-cpu-thread-policy", capture_json=True)
    assert json.loads(result.stdout) == dict(threads=torch.get_num_threads(), omp="1", mkl="1")
    assert torch.get_num_threads() == 1


def test_verified_seed_continuation_is_identical_to_uninterrupted_training(source_run):
    s = source_run
    before = {p: file_hash(p) for d in (s["source_work"], s["source_backup"]) for p in d.rglob("*") if p.is_file()}
    full = s["root"] / "full"
    invoke("train_t2tt", *s["arguments"], "--output", full, "--max-updates", 4)
    result = call(s)
    assert result["durable_updates"] == 4
    actual, expected = [load_checkpoint(p / "checkpoint_last.pt") for p in (s["work"], full)]
    for key in ("model", "optimizer", "step", "epoch", "cursor", "best_main_loss", "bad_validations",
                "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "contract_hash"):
        equal_state(actual[key], expected[key])
    assert read_json(s["work"] / "experiment.json")["planned_max_updates"] == 4
    lineage = read_json(s["work"] / "extension-lineage.json")
    assert lineage["source_contract_hash"] == lineage["target_contract_hash"] == actual["contract_hash"]
    assert {p: file_hash(p) for d in (s["source_work"], s["source_backup"]) for p in d.rglob("*") if p.is_file()} == before
    seed = next(read_json(p) for p in s["backup"].glob("*/snapshot.json") if read_json(p)["updates"] == 2)
    source_files = read_json(s["snapshot"] / "snapshot.json")["files"]
    assert all(seed["files"][name] == source_files[name] for name in ("checkpoint_last.pt", "checkpoint_best.pt"))
    equal_state(load_checkpoint(s["work"] / "checkpoint_best.pt")["model"], load_checkpoint(full / "checkpoint_best.pt")["model"])
    rows, gaps = e.metric_rows(s["work"] / "metrics.jsonl")
    assert [r["step"] for r in rows] == [1, 2, 3, 4] and not gaps
    assert e.report(s["work"], s["work"] / "diagnostics")["continuity_warnings"] == []
    # Lose VM and remove access to the old source snapshot: target must suffice.
    s["work"].rename(s["root"] / "lost-vm")
    s["source_backup"].rename(s["root"] / "offline-source")
    assert call(s)["durable_updates"] == 4


def test_target_interruption_resumes_its_own_optimizer_state(source_run, monkeypatch, benchmark_runtime):
    s = source_run
    publish = benchmark_runtime.publish_packed
    def interrupted(files, backup, identity, updates):
        result = publish(files, backup, identity, updates)
        if updates == 3:
            raise OSError("simulated interruption after verified target upload")
        return result
    monkeypatch.setattr(benchmark_runtime, "publish_packed", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        call(s, snapshot_updates=1)
    monkeypatch.setattr(benchmark_runtime, "publish_packed", publish)
    assert e.latest_snapshot(s["backup"])[1]["updates"] == 3
    s["work"].rename(s["root"] / "lost-partial-vm")
    s["source_backup"].rename(s["root"] / "offline-source")
    assert call(s)["durable_updates"] == 4
    full = s["root"] / "full"
    invoke("train_t2tt", *s["arguments"], "--output", full, "--max-updates", 4)
    a, b = [load_checkpoint(x / "checkpoint_last.pt") for x in (s["work"], full)]
    for key in ("model", "optimizer", "torch_rng", "epoch", "cursor"):
        equal_state(a[key], b[key])


@pytest.mark.parametrize("change", ["missing_last", "missing_best", "wrong_actual_step", "broken_zip", "wrong_hash",
                                   "wrong_revision", "wrong_manifest", "wrong_config"])
def test_invalid_source_snapshot_is_rejected(source_run, benchmark_runtime, change):
    import shutil
    s = dict(source_run)
    forged = s["root"] / "forged"
    shutil.copytree(s["source_work"], forged)
    if change in ("missing_last", "missing_best"):
        (forged / ("checkpoint_last.pt" if change == "missing_last" else "checkpoint_best.pt")).unlink()
    elif change == "wrong_actual_step":
        state = load_checkpoint(forged / "checkpoint_last.pt")
        state["step"] = 1
        torch.save(state, forged / "checkpoint_last.pt")
    original = read_json(s["snapshot"] / "snapshot.json")
    s["source_backup"] = s["root"] / "drive/forged"
    point = benchmark_runtime.publish_packed(forged, s["source_backup"], original["identity"], 2)
    receipt = read_json(point / "snapshot.json")
    if change == "broken_zip":
        with (point / "snapshot.zip").open("ab") as stream:
            stream.write(b"broken")
    elif change == "wrong_hash":
        receipt["zip_sha256"] = "0" * 64
    elif change == "wrong_revision":
        receipt["identity"]["benchmark_revision"] = "0" * 40
    elif change in ("wrong_manifest", "wrong_config"):
        receipt["identity"]["inputs"][str(s[change.removeprefix("wrong_")])] = "0" * 64
    write_json(point / "snapshot.json", receipt, overwrite=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        call(s)
    assert not list(s["backup"].glob("*/snapshot.json"))


def test_original_source_revision_is_recovered_without_contract_rewrite(source_run):
    import hashlib
    import io
    import subprocess
    import tarfile
    from alignment_transfer_s2ut.io import digest
    s = source_run
    repo = Path(__file__).resolve().parents[1]
    revision = "703b7e8e6027083c253dffadf51b8081c1c6d516"
    raw = subprocess.check_output(["git", "-C", str(repo), "archive", revision, "src/alignment_transfer_s2ut"])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        files = {str(Path(x.name).relative_to("src/alignment_transfer_s2ut")): hashlib.sha256(archive.extractfile(x).read()).hexdigest()
                 for x in archive if x.isfile() and x.name.endswith(".py")}
    expected_hash = digest(files)
    producer = s["root"] / "old-code-producer"
    producer.mkdir()
    runtime, _ = e.recover_implementation(producer, expected_hash, repo, revision)
    env = dict(os.environ, PYTHONPATH=runtime + os.pathsep + os.environ.get("PYTHONPATH", ""))
    old_source = s["root"] / "old-base"
    from alignment_transfer_s2ut.process_logging import run_process
    source_command = [sys.executable, "-m", "alignment_transfer_s2ut.train_t2tt", *s["arguments"],
                      "--output", str(old_source), "--max-updates", "2"]
    run_process(source_command, s["root"] / "test-logs", name="old-source-training", env=env)
    s["source_work"], s["source_backup"] = old_source, s["root"] / "drive/old-base"
    c.runtime().publish_packed(old_source, s["source_backup"], c.identity_for(source_command, old_source), 2)
    assert call(s)["durable_updates"] == 4
    plan = read_json(s["work"] / "extension-plan.json")
    assert plan["training_revision"] == revision and plan["implementation_hash"] == expected_hash
    assert load_checkpoint(s["work"] / "checkpoint_last.pt")["implementation_hash"] == expected_hash


def test_preflight_rejects_recipe_not_matching_actual_checkpoint(source_run):
    import subprocess
    s = source_run
    path = s["snapshot"] / "snapshot.json"
    doc = read_json(path)
    doc["identity"]["command"] += ["--learning-rate", "0.005"]
    write_json(path, doc, overwrite=True)
    with pytest.raises(subprocess.CalledProcessError):
        call(s)
    assert not list(s["backup"].glob("*/snapshot.json"))


def test_metrics_continuity_warns_about_gaps_and_duplicates(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_jsonl(path, [dict(step=x, **{"train/main_loss": 1.}) for x in (1, 2, 2, 4)])
    _, warnings = e.metric_rows(path)
    assert warnings == [dict(expected=3, actual=2), dict(expected=3, actual=4)]


def test_native_fairseq_model_continuation_is_identical(source_run, benchmark_runtime):
    pytest.importorskip("fairseq")
    s = dict(source_run)
    arguments = list(s["arguments"])
    arguments[arguments.index("--backend") + 1] = "fairseq"
    s["arguments"] = arguments
    s["source_work"], s["source_backup"] = s["root"] / "native-base", s["root"] / "drive/native-base"
    command = [sys.executable, "-m", "alignment_transfer_s2ut.train_t2tt", *arguments,
               "--output", str(s["source_work"]), "--max-updates", "2"]
    invoke("train_t2tt", *arguments, "--output", s["source_work"], "--max-updates", 2)
    benchmark_runtime.publish_packed(s["source_work"], s["source_backup"], c.identity_for(command, s["source_work"]), 2)
    full = s["root"] / "native-full"
    invoke("train_t2tt", *arguments, "--output", full, "--max-updates", 4)
    assert call(s)["durable_updates"] == 4
    a, b = [load_checkpoint(p / "checkpoint_last.pt") for p in (s["work"], full)]
    for key in ("model", "optimizer", "step", "epoch", "cursor", "best_main_loss", "torch_rng", "contract_hash"):
        equal_state(a[key], b[key])


def test_durable_frequency_is_independent_of_validation(source_run):
    s = source_run
    assert call(s, target_updates=8, snapshot_updates=6, keep_last=10, milestones=[])["durable_updates"] == 8
    snapshots = sorted(read_json(p)["updates"] for p in s["backup"].glob("*/snapshot.json"))
    rows, gaps = e.metric_rows(s["work"] / "metrics.jsonl")
    assert [r["step"] for r in rows if "dev/main_loss" in r] == [2, 4, 6, 8]
    assert snapshots == [2, 6, 8] and not gaps


def test_only_transient_progress_file_lock_is_retried(tmp_path, monkeypatch):
    import errno
    from types import SimpleNamespace
    monkeypatch.setattr(e, "os", SimpleNamespace(name="nt"))
    target = tmp_path / "training-local-progress.json"
    calls = []
    def writer():
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(errno.EACCES, "reader lock", "temporary", None, str(target))
        return "saved"
    assert e.progress_update(writer, target) == "saved" and len(calls) == 3
    def other_error():
        raise PermissionError(errno.EACCES, "checkpoint denied", "temporary", None, str(tmp_path / "checkpoint.pt"))
    with pytest.raises(PermissionError, match="checkpoint denied"):
        e.progress_update(other_error, target)


def test_retention_failure_preserves_new_snapshot(source_run, monkeypatch):
    s = source_run
    identity = {"fixture": "target"}
    for step in (2, 3, 4):
        folder = s["backup"] / str(step)
        write_json(folder / "snapshot.json", dict(identity=identity, updates=step))
        (folder / "snapshot.zip").write_bytes(b"fixture")
    original = Path.unlink
    def fail_old(path, *a, **kw):
        if path == s["backup"] / "2/snapshot.zip":
            raise PermissionError("fixture deletion denied")
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, "unlink", fail_old)
    e.prune_target(s["backup"], identity, 2, [])
    assert (s["backup"] / "4/snapshot.zip").is_file()
    assert (s["backup"] / "2/snapshot.zip").is_file()


@pytest.mark.parametrize("changed", ["manifest", "config", "learning_rate", "warmup_updates", "batch_size",
                                    "update_frequency", "label_smoothing"])
def test_source_settings_cannot_change(source_run, changed):
    s = source_run
    if changed in ("manifest", "config"):
        with s[changed].open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with pytest.raises(ValueError, match="differs"):
            call(s)
    else:
        state = read_json(s["snapshot"] / "snapshot.json")
        command = e.changed_option(e.changed_option(state["identity"]["command"], "--output", s["work"]), "--max-updates", 4)
        flag = "--" + changed.replace("_", "-")
        command = e.changed_option(command, flag, "3") if flag in command else [*command, flag, "3"]
        with pytest.raises(ValueError, match="only output"):
            call(s, command=command)


def test_retention_is_target_only_and_keeps_milestones(source_run, monkeypatch):
    s = source_run
    identity = {"test": "owned target"}
    for step in (2, 3, 4, 5, 6):
        directory = s["backup"] / str(step)
        write_json(directory / "snapshot.json", dict(identity=identity, updates=step))
        (directory / "snapshot.zip").write_bytes(b"owned fixture")
    before = {p: file_hash(p) for p in s["source_backup"].rglob("*") if p.is_file()}
    e.prune_target(s["backup"], identity, 2, [3])
    assert {p.name for p in s["backup"].iterdir()} == {"3", "5", "6"}
    assert all(file_hash(p) == h for p, h in before.items())
