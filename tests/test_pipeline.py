from pathlib import Path
import shutil
import numpy as np
import pytest
import torch
from alignment_transfer_s2ut.smoke_pipeline import smoke, invoke
from alignment_transfer_s2ut.training import load_checkpoint
from alignment_transfer_s2ut.io import read_json, read_jsonl, write_jsonl


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    result = smoke(root)
    assert result["status"] == "passed"
    return root


def test_end_to_end_all_methods_teacher_free_waveform_inference(completed):
    root = completed
    for name in ("t2tt", "mt", "ctc", "teacher"):
        assert not (root / name).exists()
    initializations = set()
    recipes = set()
    for method in ("baseline", "distillation", "guidance", "annealed_guidance"):
        experiment = read_json(root / method / "experiment.json")
        initializations.add(experiment["initialization_hash"])
        recipes.add(str(experiment["recipe"]))
        predictions = read_jsonl(root / method / "predictions.jsonl")
        assert predictions[0]["inference_beta"] == 0
        assert Path(predictions[0]["en_audio"]).is_file()
        assert all(0 <= unit < 100 for unit in predictions[0]["units"])
    assert len(initializations) == len(recipes) == 1
    log = read_jsonl(root / "distillation/metrics.jsonl")
    assert log[0]["train/alignment_loss"] > 0
    log = read_jsonl(root / "annealed_guidance/metrics.jsonl")
    assert log[-1]["guidance/beta"] == 0
    from alignment_transfer_s2ut.compare_experiments import compare_experiments
    assert compare_experiments([root / m for m in ("baseline", "distillation", "guidance", "annealed_guidance")])["fair_comparison"]


def test_resumed_training_matches_uninterrupted(completed, tmp_path):
    torch.set_num_threads(1)
    common = ["--manifest", completed / "manifest.jsonl", "--config", completed / "tiny.yaml",
              "--backend", "smoke", "--batch-size", 2, "--update-frequency", 1, "--warmup-updates", 0]
    invoke("train_t2tt", *common, "--max-updates", 3, "--output", tmp_path / "full")
    invoke("train_t2tt", *common, "--max-updates", 1, "--output", tmp_path / "resume")
    invoke("train_t2tt", *common, "--max-updates", 3, "--output", tmp_path / "resume", "--resume")
    full = load_checkpoint(tmp_path / "full/checkpoint_last.pt")
    resumed = load_checkpoint(tmp_path / "resume/checkpoint_last.pt")
    assert full["step"] == resumed["step"] == 3
    for key, value in full["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
    with pytest.raises(ValueError, match="contract"):
        invoke("train_t2tt", *common, "--max-updates", 4, "--output", tmp_path / "resume", "--resume", "--seed", 7)


def test_inference_deploys_only_checkpoint_and_source_audio(completed, tmp_path):
    checkpoint = tmp_path / "model.pt"
    audio = tmp_path / "source.wav"
    shutil.copyfile(completed / "guidance/checkpoint_best.pt", checkpoint)
    shutil.copyfile(completed / "test3-ja.wav", audio)
    prediction = invoke("infer", "--checkpoint", checkpoint, "--audio", audio,
                        "--output", tmp_path / "prediction.jsonl", "--units-only", "--max-units", 4, "--beam-size", 1)
    assert prediction[0]["units"]
    resumed = invoke("infer", "--checkpoint", checkpoint, "--audio", audio,
                     "--output", tmp_path / "prediction.jsonl", "--units-only", "--max-units", 4,
                     "--beam-size", 1, "--resume")
    assert prediction == resumed
    with pytest.raises(ValueError, match="contract"):
        invoke("infer", "--checkpoint", checkpoint, "--audio", audio,
               "--output", tmp_path / "prediction.jsonl", "--units-only", "--max-units", 5,
               "--beam-size", 1, "--resume")
    forbidden = tmp_path / "forbidden.jsonl"
    write_jsonl(forbidden, [{"pair_id": "source", "ja_audio": str(audio), "en_text": "leak"}])
    with pytest.raises(ValueError, match="only"):
        invoke("infer", "--checkpoint", checkpoint, "--input", forbidden,
               "--output", tmp_path / "bad.jsonl", "--units-only")


def test_evaluation_and_manifest_idempotent_resume(completed):
    previous = read_json(completed / "baseline/evaluation.json")
    resumed = invoke("evaluate", "--manifest", completed / "manifest.jsonl", "--split", "test",
                     "--predictions", completed / "baseline/predictions.jsonl", "--units", completed / "units",
                     "--output", completed / "baseline/evaluation.json", "--resume")
    assert previous == resumed
    invoke("prepare_manifest", "--input", completed / "input.jsonl", "--output", completed / "manifest.jsonl", "--resume")


def test_export_inference_input_has_only_source_speech(completed, tmp_path):
    from alignment_transfer_s2ut.prepare_inference_input import prepare_inference_input
    rows = prepare_inference_input(completed / "manifest.jsonl", "test", tmp_path / "source.jsonl")
    assert len(rows) == 1 and set(rows[0]) == {"pair_id", "ja_audio"}
    assert prepare_inference_input(completed / "manifest.jsonl", "test", tmp_path / "source.jsonl", resume=True) == rows


def test_no_reference_repository_runtime_imports():
    root = Path(__file__).resolve().parents[1] / "src"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import direct_s2st" not in text and "from direct_s2st" not in text
