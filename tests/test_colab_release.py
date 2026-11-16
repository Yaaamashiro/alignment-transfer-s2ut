"""Exercise the primary notebooks' release bootstrap without Colab or Drive."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_training_notebook import cells


ROOT = Path(__file__).resolve().parents[1]
RELEASE = runpy.run_path(str(ROOT / "tools/colab_release.py"))
PINS = ("a" * 40, "b" * 40)


def test_release_manifest_and_notebook_forms():
    RELEASE["sync_colab_release"](ROOT, check=True)
    release = json.loads((ROOT / "notebooks/colab/release.json").read_text(encoding="utf-8"))
    label = "Last updated " + RELEASE["notebook_update_label"](release["NOTEBOOK_UPDATED_AT"])
    settings = cells("training")["settings"]
    headings = [line.split("### ", 1)[1] for line in settings.splitlines() if line.startswith("#@markdown ### ")]
    assert headings == ["Experiment", "Data and audio preparation", "T2TT teacher model",
                        "S2UT Baseline", "Teacher artifacts", "Top-p diagnostics and calibration",
                        "Transfer training conditions", "Pilot and checkpoint recovery", "Evaluation and diagnostics"]
    for name in ("training", "evaluate"):
        notebook = json.loads((ROOT / f"notebooks/colab/{name}.ipynb").read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"][0]["source"][2].strip() == label
        assert "Last updated" not in cells(name)["settings"]
        assert "NOTEBOOK_UPDATED_AT" not in cells(name)["settings"]
        assert f"NOTEBOOK_UPDATED_AT = {release['NOTEBOOK_UPDATED_AT']!r}" in cells(name)["paths"]
        for identity, code in cells(name).items():
            compile(code, f"{name}:{identity}", "exec")
        order = list(cells(name))
        assert order.index("paths") < order.index("release-check")
        assert next(i for i, code in enumerate(cells(name).values()) if "name='install-uv'" in code) > order.index("release-check")


def test_notebook_update_labels_distinguish_two_updates_on_the_same_day():
    label = RELEASE["notebook_update_label"]
    assert label("2026-11-16T09:01+09:00") == "2026-11-16 09:01 JST"
    assert label("2026-11-16T09:02+09:00") == "2026-11-16 09:02 JST"
    for value in ("2026-11-16", "2026-11-16T09:01", "2026-11-16T09:01:00+09:00"):
        with pytest.raises(ValueError, match="NOTEBOOK_UPDATED_AT"):
            label(value)


@pytest.mark.parametrize("issue", [None, "alignment-head", "benchmark-head", "dirty", "checkout", "short-pin"])
def test_bootstrap_records_the_observed_state_and_fails_before_install(tmp_path, issue):
    observed = [dict(actual_head=pin, worktree="clean", status_entries=[]) for pin in PINS]
    expected = list(PINS)
    if issue == "alignment-head":
        observed[0]["actual_head"] = "c" * 40
    if issue == "benchmark-head":
        observed[1]["actual_head"] = "c" * 40
    if issue == "dirty":
        observed[1].update(worktree="dirty", status_entries=["?? relative-override.py"])
    if issue == "short-pin":
        expected[0] = expected[0][:8]
    calls = []

    def execute(*args, name, **kwargs):
        calls.append((name, args))
        if issue == "checkout" and name == "checkout-alignment":
            raise RuntimeError("checkout failure")
        if name == "inspect-colab-revisions":
            assert kwargs["capture_json"]
            return SimpleNamespace(stdout=json.dumps(observed))

    events = []
    logs, drive = tmp_path / "logs", tmp_path / "drive"
    call = lambda: RELEASE["prepare_colab_release"](
        execute, lambda *event: events.append(event), notebook="training",
        notebook_updated_at="2026-11-16T09:02+09:00",
        experiment_id="experiment-001", transfer_id="", logs=logs, persistent=drive,
        alignment=(tmp_path / "repo", "https://example.com/repo.git", expected[0]),
        benchmark=(tmp_path / "benchmark", "https://example.com/benchmark.git", expected[1]))
    if issue:
        with pytest.raises(RuntimeError, match="revision check failed"):
            call()
    else:
        assert call()["status"] == "PASS"
    record, = logs.glob("colab-release-*.json")
    saved = (drive / "execution-provenance" / record.name).read_bytes()
    assert saved == record.read_bytes()
    payload = json.loads(saved)
    assert events[0][1] == payload["status"] == ("FAIL" if issue else "PASS")
    assert payload["schema"] == "colab-release-v2"
    assert payload["notebook_updated_at"] == "2026-11-16T09:02+09:00"
    for index, key in enumerate(("alignment", "benchmark_runtime")):
        assert payload["repositories"][key] == {"pinned_revision": expected[index], **observed[index]}
    for _, command in calls:
        assert "main" not in command
    if issue == "short-pin":
        assert len(calls) == 1


def test_git_checkout_probe_detects_real_tracked_and_untracked_changes(tmp_path):
    repo = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    tracked = repo / "config"
    tracked.write_text("revision one", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "config"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "test: create checkout fixture"], check=True)
    probe = [sys.executable, "-c", RELEASE["GIT_CHECKOUT_STATE"], str(repo)]
    clean, = json.loads(subprocess.check_output(probe, text=True))
    assert len(clean["actual_head"]) == 40 and clean["worktree"] == "clean"
    tracked.write_text("changed", encoding="utf-8")
    (repo / "other").touch()
    dirty, = json.loads(subprocess.check_output(probe, text=True))
    assert dirty["actual_head"] == clean["actual_head"] and dirty["worktree"] == "dirty"
    assert dirty["status_entries"] == [" M config", "?? other"]


def test_embedded_command_guard_and_release_receipt(tmp_path):
    ns = {}
    exec(cells("training")["process-logging"], ns)
    ns.update(LOGS=tmp_path, RELEASE_REQUIRED=True, RELEASE_RECORD=None)
    with pytest.raises(RuntimeError, match="revision check before"):
        ns["execute"](sys.executable, "-c", "raise SystemExit(99)")
    release = {"status": "PASS", "run_id": "fixture"}
    ns["RELEASE_RECORD"] = release
    result = ns["execute"](sys.executable, "-c", "print('verified')", name="fixture")
    assert result.record["colab_release"] == release
    receipt, = tmp_path.glob("*.json")
    assert json.loads(receipt.read_text(encoding="utf-8"))["colab_release"] == release
