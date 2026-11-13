import math
import os
import random
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from test_benchmark_colab import benchmark_runtime
from test_benchmark_native import native_data
from test_distillation_native import calibrated
from alignment_transfer_s2ut import benchmark, colab_runtime, transfer_pilot
from alignment_transfer_s2ut.alignment.calibration import calibrate_lambda, gradient_diagnostics
from alignment_transfer_s2ut.io import file_hash, read_json, read_jsonl, write_json


def test_gradient_diagnostic_norm_ratio_cosine_and_invalid_values():
    task = [torch.tensor([2., 0.]), None]
    align = [torch.tensor([-3., 0.]), None]
    observed = gradient_diagnostics(task, align, 2., 3., .5)
    assert observed["gradient_ratio"] == .75 and observed["gradient_cosine"] == -1.
    assert gradient_diagnostics(task, align, 2., 3., 0.)["gradient_ratio"] == 0.
    zero = gradient_diagnostics(task, [None, None], 2., 0., 1.)
    assert zero["gradient_ratio"] == 0 and zero["gradient_cosine"] is None
    assert zero["cosine_undefined_reason"] == "zero alignment gradient"
    assert gradient_diagnostics([None, None], align, 0., 3., 1.)["gradient_ratio"] is None
    for nonfinite in (math.nan, math.inf):
        with pytest.raises(ValueError, match="nonfinite"):
            gradient_diagnostics(task, align, nonfinite, 3., .5)


def test_baseline_gate_reads_only_complete_receipts(tmp_path, benchmark_runtime):
    receipt = tmp_path / "snapshot/snapshot.json"
    files = {name: "hash" for name in ("checkpoints/checkpoint_last.pt", "checkpoints/checkpoint_best.pt",
                                       "research-metadata.json", "s2ut-recipe-lock.json")}
    write_json(receipt, {"updates": 10000, "files": files})
    before = file_hash(receipt)
    with pytest.raises(ValueError, match="published 400,000-update"):
        transfer_pilot.baseline_ready(tmp_path)
    assert file_hash(receipt) == before
    write_json(receipt, {"updates": 400000, "files": files}, overwrite=True)
    assert transfer_pilot.baseline_ready(tmp_path)["updates"] == 400000


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.fairseq
@pytest.mark.parametrize("method,native_data", [
    pytest.param("distillation", (2, "cpu", 1, None), id="kl-mid-epoch"),
    pytest.param("annealed_distillation", (2, "cpu", 2, 2), id="annealed-report-only"),
    pytest.param("mask_distillation", 2, id="mask-native"),
    pytest.param("annealed_distillation", (2, "cuda", 2, 2, 5), marks=pytest.mark.gpu, id="annealed-fp16"),
], indirect=["native_data"])
def test_native_pilot_restore_and_continue_matches_resident_run(calibrated, native_data, benchmark_runtime, method, monkeypatch):
    root, calibration = calibrated
    shim = root / "sampler"
    shim.mkdir()
    startup = "import random\nrandom.seed(1)\n"
    try:
        __import__("fairseq.data.data_utils_fast")
    except ImportError:
        # Give the real child the same singleton sampler as the synthetic fixture.
        startup += (
            "from fairseq.data.fairseq_dataset import FairseqDataset\n"
            "FairseqDataset.batch_by_size = lambda self, indices, **kw: [[int(i)] for i in indices]\n")
    (shim / "sitecustomize.py").write_text(startup, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(shim) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    baseline = root / "baseline.json"
    protected = {path: file_hash(path) for path in (baseline, root / "metadata.json", root / "recipe.json",
                                                    root / "baseline/checkpoints/checkpoint_best.pt")}
    calibration_path = root / "analysis/alignment-calibration.json"
    for kind in ("kl", "mask"):
        norms = [(r["gradient_norms"]["task"], r["gradient_norms"][kind]) for r in calibration["per_batch"]]
        coefficient = calibrate_lambda(norms)
        assert calibration["lambda_" + kind] == coefficient
        for row, (task, alignment) in zip(calibration["per_batch"], norms, strict=True):
            assert row["gradient_diagnostics"][kind]["gradient_ratio"] == coefficient * alignment / task
        assert sum(calibration["diagnostic_summary"][kind]["layer_weights"]["mean_weights"]) == pytest.approx(1.)
    for candidate in (.8, .9, .95):
        row = calibration["top_p_diagnostics"]["candidates"][str(candidate)]
        assert row["invalid_rows"] > 0 and row["retained_teacher_mass"]["min"] >= candidate - 1e-6

    def command(work):
        return [sys.executable, "-m", "alignment_transfer_s2ut.benchmark", "train", "--resume",
                "--baseline", str(baseline), "--manifest", str(root / "manifest.jsonl"),
                "--units", str(root / "units"), "--teacher", str(root / "teacher"), "--method", method,
                "--calibration", str(calibration_path), "--output", str(work)]

    split, continuous = root / "split", root / "continuous"
    backup = root / "split-backup"
    budget = native_data.record["training_budget"]
    paused = 2 if budget == 5 else 1
    check = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/review_transfer_pilot.py"))["recovery_check"]
    if budget == 5:
        assert check(split, backup, method, "save")["target"] == paused
    state = colab_runtime.train_session([*command(split), "--stop-after-update", str(paused)], split, backup)
    assert state == {"status": "PILOT_PAUSED", "durable_updates": paused, "training_budget": budget}
    plan = read_json(split / "native-run.json")
    assert plan["config"]["optimization"]["max_update"] == budget
    last = split / "checkpoints/checkpoint_last.pt"
    checkpoint = benchmark.load_native(last)
    assert checkpoint["last_optimizer_state"] and checkpoint["optimizer_history"][-1]["lr_scheduler_state"] is not None
    if plan["config"]["common"]["fp16"]:
        assert checkpoint["last_optimizer_state"]["loss_scale"] > 0
        assert checkpoint["extra_state"]["alignment_fp32_master"][0].dtype == torch.float32
    assert {"train_iterator", "alignment_rng", "alignment_selection", "alignment_contract_hash"} <= checkpoint["extra_state"].keys()
    with pytest.raises(ValueError, match="completed training budget"):
        benchmark.register_transfer(split, last, root / "early.json")
    with pytest.raises(ValueError, match="incomplete|lacks plan or last/best"):
        colab_runtime.restore_native_training(split, backup, method)

    before = file_hash(last)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    probe = transfer_pilot.probe(split, paused, calibration_path, root / "probe/probe.json")
    assert_nested_equal(rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
    assert probe["step"] == paused and probe["status"] == "PILOT_PAUSED" and probe["dev"]["split"] == "dev"
    if plan["config"]["common"]["fp16"]:
        assert probe["peak_cuda_bytes"] > 0 and probe["loss_scale"] == checkpoint["last_optimizer_state"]["loss_scale"]
    assert probe["lambda_t"] == pytest.approx(calibration["lambda_" + ("mask" if method.startswith("mask") else "kl")] *
        ((1 + math.cos(math.pi * paused / (.8 * budget))) / 2 if method == "annealed_distillation" else 1))
    assert file_hash(last) == before
    if method == "distillation":
        scores = transfer_pilot.translate(root / "pilot-score/result.json", root / "manifest.jsonl", root / "units", run=split)
        assert scores["split"] == "dev" and scores["samples"] == 1
        assert scores["checkpoint_updates"] == 1 and scores["unit_error_rate"] >= 0
    split.rename(root / "cold-vm")
    changed = command(split)
    changed.extend(["--mask-top-p", ".95"])
    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero exit status"):
        colab_runtime.train_session(changed, split, backup)
    if budget == 5:
        target = check(split, backup, method, "resume")["target"]
        colab_runtime.train_session([*command(split), "--stop-after-update", str(target)], split, backup)
        assert check(split, backup, method, "resume", report=True)["status"] == "RECOVERY_CHECK_PASSED"
    resumed = colab_runtime.train_session(command(split), split, backup)
    whole = colab_runtime.train_session(command(continuous), continuous, root / "continuous-backup")
    assert resumed == whole == {"status": "COMPLETED", "durable_updates": budget, "training_budget": budget}
    final = benchmark.load_native(last)
    reference = benchmark.load_native(continuous / "checkpoints/checkpoint_last.pt")
    for key in ("model", "last_optimizer_state"):
        assert_nested_equal(final[key], reference[key])
    # fairseq appends a history entry on each process resume.
    assert_nested_equal(final["optimizer_history"][-1], reference["optimizer_history"][-1])
    for key in ("alignment_rng", "train_iterator", "alignment_selection", "best", "alignment_fp16_scaler", "alignment_fp32_master"):
        assert_nested_equal(final["extra_state"].get(key), reference["extra_state"].get(key))
    assert [row["step"] for row in read_jsonl(split / "alignment-train.jsonl")] == list(range(1, budget + 1))
    record = benchmark.register_transfer(split, last.with_name("checkpoint_best.pt"), root / "final.json")
    assert record["scientific_config"] == read_json(baseline)["scientific_config"]
    assert record["training_budget"] == budget and record["method"] == method
    assert {path: file_hash(path) for path in protected} == protected
