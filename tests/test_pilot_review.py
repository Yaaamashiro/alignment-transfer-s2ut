from pathlib import Path
import json
import runpy
import sys

import pytest
import torch

from alignment_transfer_s2ut.colab_runtime import BENCHMARK_REVISION, BENCHMARK_RUNTIME_REVISION
from alignment_transfer_s2ut.io import digest, file_hash, read_json, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
REVIEW = runpy.run_path(str(ROOT / "tools/review_transfer_pilot.py"))
REVISION = "fcfbbfc662421c71df3bce24f2c6c69fddd1d869"
ADAPTER = "c6e6f85c6ba617d594e55be4431705e926f6342f87d9cc01a2fafca475209e35"


def test_original_revision_matches_historical_baseline_adapter():
    REVIEW["verify_adapter"](ROOT, REVISION, ADAPTER)
    with pytest.raises(ValueError, match="does not match"):
        REVIEW["verify_adapter"](ROOT, REVISION, "0" * 64)


def test_completed_baseline_binds_snapshot_checkpoint_recipe_and_launch(tmp_path, monkeypatch):
    pytest.importorskip("omegaconf")
    from types import ModuleType

    paper = ModuleType("direct_s2st.s2ut.paper_recipe")
    paper.MAX_UPDATES = 400000
    monkeypatch.setitem(sys.modules, paper.__name__, paper)
    b = REVIEW["benchmark"]
    work = tmp_path / "baseline"
    backup = tmp_path / "backups"
    checkpoint = work / "checkpoints/checkpoint_best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"registered best")
    data = tmp_path / "prepared"
    cfg = {"task": {}, "model": {}, "criterion": {}, "common": {"seed": 1, "fp16": True},
           "dataset": {"num_workers": 8}, "optimization": {"max_update": 400000}, "lr_scheduler": {"warmup_updates": 10000}}
    recipe = work / "s2ut-recipe-lock.json"
    write_json(recipe, {"fixture": True})
    meta = work / "research-metadata.json"
    metadata = {"repository_revision": BENCHMARK_RUNTIME_REVISION}
    write_json(meta, metadata)
    last = work / "checkpoints/checkpoint_last.pt"
    torch.save({"cfg": cfg, "optimizer_history": [{"num_updates": 400000}],
                "extra_state": {"s2st_recipe_fingerprint": "fixture"}}, last)
    lock = work.parent / "baseline-runtime.json"
    write_json(lock, {"repository": BENCHMARK_REVISION, "fairseq": b.REVISION, "adapter_sha256": ADAPTER})
    files = {path.relative_to(work).as_posix(): file_hash(path) for path in (last, checkpoint, recipe, meta)}
    write_json(backup / "final/snapshot.json", {"updates": 400000, "files": files, "identity": {
        "work": str(work), "config": {"identity_files": [str(lock)], "kind": "s2ut",
                                      "command": ["alignment_transfer_s2ut.corpus_runtime", "_fairseq"]},
        "inputs": {str(lock): file_hash(lock)}}})
    record = {"method": "baseline", "checkpoint": str(checkpoint), "data": str(data),
              "checkpoint_hash": file_hash(checkpoint), "training_budget": 400000, "baseline_fingerprint": "fixture",
              "scientific_config": b.scientific_config(cfg), "prepared": {}, "fairseq_revision": b.REVISION,
              "metadata": metadata, "record_hash": "registered", "provenance": {
                  "metadata_hash": file_hash(meta), "recipe_lock_hash": file_hash(recipe)}}
    monkeypatch.setattr(b, "read_record", lambda _: record)
    verify = REVIEW["verify_baseline"]
    report = verify("fixture", backup, tmp_path / "reports", ROOT)
    assert report["recipe_verified"] is True
    assert report["executed_revision_verified"] is False and report["alignment_revision_source"] == "unavailable"
    assert "No alignment commit SHA" in report["warnings"][0]
    candidate = verify("fixture", backup, tmp_path / "reports", ROOT, revision=REVISION)
    assert candidate["recipe_verified"] is True
    assert candidate["executed_revision_verified"] is False and candidate["alignment_revision_source"] == "notebook_pin"
    assert "notebook pin matches" in candidate["warnings"][0]
    assert candidate["report_path"] != report["report_path"]

    logs = tmp_path / "logs"
    write_json(logs / "checkout.json", {"name": "checkout-alignment", "status": "succeeded", "returncode": 0,
        "started_utc": "2026-11-12T00:00:00+00:00", "elapsed_seconds": 1.,
        "command": ["git", "-C", str(ROOT), "checkout", "--detach", REVISION]})
    write_json(logs / "launch.json", {"name": "baseline", "pid": 2, "cwd": str(ROOT),
        "started_utc": "2026-11-12T00:00:02+00:00", "command": ["python", "-m", "alignment_transfer_s2ut.corpus_runtime",
            "baseline", "--train", "--work", str(work), "--backup", str(backup)]})
    result = verify("fixture", backup, tmp_path / "reports", ROOT, logs=logs)
    assert result["alignment_revision_source"] == "launch_log" and result["alignment_revision"] == REVISION
    assert result["executed_revision_verified"] is True and result["warnings"] == []
    with pytest.raises(ValueError, match="sources disagree"):
        verify("fixture", backup, tmp_path / "reports", ROOT, revision="0" * 40, logs=logs)

    metadata["alignment_revision"] = REVISION
    write_json(meta, metadata, overwrite=True)
    record["provenance"]["metadata_hash"] = file_hash(meta)
    receipt_path = backup / "final/snapshot.json"
    receipt = read_json(receipt_path)
    receipt["files"]["research-metadata.json"] = file_hash(meta)
    write_json(receipt_path, receipt, overwrite=True)
    result = verify("fixture", backup, tmp_path / "reports", ROOT)
    assert result["alignment_revision_source"] == "metadata" and result["alignment_revision"] == REVISION
    assert result["executed_revision_verified"] is True and result["warnings"] == []

    last.write_bytes(b"changed")
    with pytest.raises(ValueError, match="no completed Baseline snapshot"):
        verify("fixture", backup, tmp_path / "reports", ROOT)


def test_baseline_cli_emits_warning_to_stderr_and_json_to_stdout(monkeypatch, capsys):
    main = REVIEW["main"]
    report = {"recipe_verified": True, "executed_revision_verified": False, "warnings": ["Baseline revision unverified"]}
    monkeypatch.setitem(main.__globals__, "verify_baseline", lambda **kwargs: report)
    monkeypatch.setattr(sys, "argv", ["review_transfer_pilot.py", "baseline", "--baseline", "record",
                                    "--backup", "backup", "--output", "report", "--repo", "."])
    main()
    output = capsys.readouterr()
    assert json.loads(output.out) == report
    assert output.err.strip() == "WARNING: Baseline revision unverified"


def test_short_recovery_requires_cold_directory_and_optimizer_history(tmp_path):
    pytest.importorskip("omegaconf")
    check = REVIEW["recovery_check"]
    work, backup = tmp_path / "work", tmp_path / "backup"
    method = "distillation"
    assert check(work, backup, method, "save")["target"] == 2
    with pytest.raises(ValueError, match="publish the update-2"):
        check(work, backup, method, "resume")
    write_json(backup / "step2/snapshot.json", {"updates": 2})
    assert check(work, backup, method, "resume")["target"] == 4
    cfg = {"common": {"fp16": True}, "dataset": {"num_workers": 8}, "lr_scheduler": {"warmup_updates": 10000}}
    plan = {"config": cfg, "transfer": {"method": method}, "training_budget": 400000}
    plan["contract_hash"] = digest(plan)
    write_json(work / "native-run.json", plan)
    with pytest.raises(ValueError, match="fresh Colab runtime"):
        check(work, backup, method, "resume")
    write_json(work / "native-session.json", {"status": "PILOT_PAUSED", "durable_updates": 4})
    write_jsonl(work / "alignment-train.jsonl", [{"step": step} for step in range(1, 5)])
    checkpoint = work / "checkpoints/checkpoint_last.pt"
    checkpoint.parent.mkdir()
    state = {"cfg": cfg, "optimizer_history": [{"num_updates": 2}, {"num_updates": 4}],
             "extra_state": {"alignment_contract_hash": plan["contract_hash"], "train_iterator": {"epoch": 1},
                             "alignment_fp16_scaler": {"loss_scale": 64}, "alignment_fp32_master": [torch.ones(1)]}}
    receipt = backup / "step4/snapshot.json"

    def publish():
        torch.save(state, checkpoint)
        write_json(receipt, {"updates": 4, "files": {"checkpoints/checkpoint_last.pt": file_hash(checkpoint)}}, overwrite=True)

    publish()
    output = tmp_path / "reports/resume.json"
    report = check(work, backup, method, "resume", report=True, output=output)
    assert report["status"] == "RECOVERY_CHECK_PASSED" and report["updates"] == [1, 2, 3, 4]
    assert output.is_file() and report["checkpoint_sha256"] == file_hash(checkpoint)
    state["optimizer_history"] = [{"num_updates": 4}]
    publish()
    with pytest.raises(ValueError, match="optimizer history"):
        check(work, backup, method, "resume", report=True)
