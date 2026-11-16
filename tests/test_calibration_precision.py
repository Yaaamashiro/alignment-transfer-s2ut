import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from alignment_transfer_s2ut import alignment_calibration as calibration, colab_runtime
from alignment_transfer_s2ut.alignment import precision
from alignment_transfer_s2ut.io import digest


def test_fp16_gradients_are_widened_before_unscale_and_added_in_fp32(monkeypatch):
    parameter = torch.tensor([0.], dtype=torch.float16, requires_grad=True)
    calls = []
    monkeypatch.setattr(calibration, "encoder_padding_mask", lambda *_: None)
    monkeypatch.setattr(calibration, "layer_statistics", lambda *_: {"losses": parameter, "counts": None})
    monkeypatch.setattr(calibration, "worst_layer", lambda loss, *_: (loss.float().sum() * 2**-28, loss))

    def components(*_):
        calls.append(1)
        return parameter.float().sum() * 2**-28, 1, None, None

    criterion = SimpleNamespace(sentence_avg=False, components=components)
    samples = [{"alignment_prior": None, "ntokens": 1}] * 2
    gradients, _ = calibration.scaled_gradients(
        None, criterion, samples, [parameter], {"kl": 1., "mask": 1.}, .8, (1, 128, 1024))
    assert len(calls) == 2
    for kind in ("task", "kl", "mask"):
        assert gradients[1][kind][0].item() == 0.
        for scale in (128, 1024):
            assert gradients[scale][kind][0].dtype == torch.float32
            assert gradients[scale][kind][0].item() == 2**-28
    assert parameter.grad is None


@pytest.mark.parametrize("variant, selected", [("stable", 1024), ("overflow", 128),
                                                 ("different_direction", None), ("no_reference", None)])
def test_scale_qualification_checks_every_batch_and_objective(variant, selected):
    reference = torch.tensor([1., 0.])
    param = [reference]
    objectives = {kind: [reference] for kind in ("task", "kl", "mask")}
    scales = {scale: objectives.copy() for scale in precision.SCALES}
    first = precision.compare_scales(scales, param, objectives)
    if variant == "overflow":
        scales[1024]["mask"] = [torch.full_like(reference, torch.inf)]
    elif variant == "different_direction":
        for scale in scales:
            scales[scale]["kl"] = [torch.tensor([0., 1.])]
    last = precision.compare_scales(scales, param, None if variant == "no_reference" else objectives)
    scale, receipts = precision.qualify([first, last])
    assert scale == selected
    if variant == "overflow":
        assert last["metrics"]["1024"]["mask"]["nonfinite"] == 2
        assert receipts["1024"]["fp32_failed_batches"] == [2]
    if variant == "different_direction":
        assert last["comparisons"]["1024/fp32"]["kl"]["relative_norm"] == 0
        assert receipts["1024"]["fp32_failed_batches"] == [2]


def test_legacy_top_p_compatibility_binds_unchanged_sources(monkeypatch):
    projection = "alignment/projection.py"
    files = {projection: "projection hash", "alignment_calibration.py": "new",
             "alignment/calibration.py": "new", "alignment/precision.py": "new"}
    old = {str(Path(k)): v for k, v in (calibration.PREVIOUS_CALIBRATION_SOURCES |
                                       {projection: files[projection]}).items()}
    old_hash = digest(old)
    plan = {"implementation_hash": digest(files)}
    monkeypatch.setattr(colab_runtime, "s2ut_implementation_files", lambda _: files.copy())
    assert calibration.compatible_top_p_hashes(plan) == {old_hash, plan["implementation_hash"]}
    files[projection] = "changed projection"
    assert old_hash not in calibration.compatible_top_p_hashes(plan)
    files[projection] = "projection hash"
    monkeypatch.setattr(calibration, "read_json", lambda _: {"alignment_calibration.py": "different code hash"})
    assert calibration.compatible_top_p_hashes(plan) == {plan["implementation_hash"]}


def test_colab_restores_previous_calibration_and_publishes_separately(tmp_path, monkeypatch):
    notebook = Path(__file__).resolve().parents[1] / "notebooks/colab/training.ipynb"
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]}
    output = tmp_path / "new/alignment-calibration.json"
    previous = tmp_path / "old/alignment-calibration.json"
    top_p = tmp_path / "top-p/top-p.json"
    top_p.parent.mkdir()
    top_p.write_text('{"candidates": {}}', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace(DataFrame=list))
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(display=lambda _: None, Image=dict))
    root, persistent, suffix = tmp_path / "work", tmp_path / "drive", "-all-layers"
    baseline, manifest, teacher = [root / name for name in ("baseline.json", "manifest.jsonl", "teacher")]
    module = "alignment_transfer_s2ut.alignment_calibration"
    old_args = ["calibrate", "--baseline", baseline, "--manifest", manifest, "--units", root / "units",
                "--teacher", teacher, "--batches", 16, "--diagnostics", top_p, "--mask-top-p", .8,
                "--output", previous]
    marker = persistent / "stages" / ("research-v5-calibration" + suffix) / "pack/snapshot.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"files": {"stage-complete.json": "hash"},
                                   "identity": {"command": ["python", "-m", module, *map(str, old_args)]}}))
    calls = []
    context = dict(Path=Path, json=json, ROOT=root, PERSISTENT=persistent, SCOPE_SUFFIX=suffix,
                   PREPARE_AUDIO_PACKS=False, NEED_TEACHER=True, BASELINE=baseline, MANIFEST=manifest,
                   TEACHER_ROOT=teacher, CALIBRATION_BATCHES=16, RUN_MASK_TOPP_DIAGNOSTICS=False,
                   RUN_ALIGNMENT_CALIBRATION=True, TOPP_REPORT=top_p, TOPP_ROOT=top_p.parent, PYTHON="python",
                   PREVIOUS_CALIBRATION=previous, CALIBRATION=output, MASK_TOP_P=.8, LOGS=tmp_path,
                   run=lambda *args, **kw: calls.append((args, kw)), log_event=lambda *args: None)
    exec(cells["calibration"], context)
    assert calls[0][0][0] == "research-v5-calibration" + suffix
    assert calls[0][1] == {"restore_only": True}
    assert calls[1][0][0] == "research-v6-calibration-fp16" + suffix
    assert calls[1][0][-4:] == ("--previous-calibration", previous, "--output", output)
    assert calls[1][1] == {"gpu": True}
