"""Opt-in real-model pipeline test. No model downloads occur in the default test run."""
import os
from pathlib import Path
import shutil
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not os.environ.get("AT_S2UT_GPU_MANIFEST") or not torch.cuda.is_available(),
    reason="set AT_S2UT_GPU_MANIFEST to a small real train/dev/test manifest on a CUDA machine")]


def test_real_ctc_hubert_native_s2ut_vocoder_teacher_free(tmp_path):
    from alignment_transfer_s2ut.smoke_pipeline import invoke
    from alignment_transfer_s2ut.data.manifest import load_manifest
    from alignment_transfer_s2ut.io import write_jsonl
    root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.jsonl"
    invoke("prepare_manifest", "--input", os.environ["AT_S2UT_GPU_MANIFEST"], "--output", manifest)
    invoke("fetch_assets", "--config", root / "configs/units/hubert_km100.yaml")
    invoke("fetch_assets", "--config", root / "configs/s2ut/vocoder.yaml")
    common = ["--manifest", manifest, "--config", root / "configs/s2ut/fisher.yaml",
              "--device", "cuda", "--batch-size", 1, "--update-frequency", 1, "--max-updates", 2]
    invoke("train_t2tt", *common, "--output", tmp_path / "t2tt")
    invoke("extract_t2tt_attention", "--manifest", manifest, "--checkpoint", tmp_path / "t2tt/checkpoint_best.pt",
           "--device", "cuda", "--output", tmp_path / "mt")
    invoke("align_speech_text", "--manifest", manifest, "--config", root / "configs/alignment/ctc.yaml",
           "--device", "cuda", "--output", tmp_path / "ctc")
    invoke("extract_units", "--manifest", manifest, "--config", root / "configs/units/hubert_km100.yaml",
           "--device", "cuda", "--output", tmp_path / "units")
    invoke("build_teacher_alignment", "--manifest", manifest, "--mt-attention", tmp_path / "mt",
           "--alignment", tmp_path / "ctc", "--units", tmp_path / "units", "--output", tmp_path / "teacher")
    methods = ["baseline", "distillation", "guidance", "annealed_guidance"]
    for method in methods:
        invoke("train_s2ut", *common, "--units", tmp_path / "units", "--teacher", tmp_path / "teacher",
               "--method", method, "--output", tmp_path / method)
    for name in ("t2tt", "mt", "ctc", "teacher"):
        shutil.rmtree(tmp_path / name)
    rows = load_manifest(manifest, "test")
    write_jsonl(tmp_path / "speech_only.jsonl", [{"pair_id": row["pair_id"], "ja_audio": row["ja_audio"]} for row in rows])
    for method in methods:
        result = invoke("infer", "--input", tmp_path / "speech_only.jsonl", "--checkpoint", tmp_path / method / "checkpoint_best.pt",
                        "--vocoder-config", root / "configs/s2ut/vocoder.yaml", "--device", "cuda",
                        "--max-units", 32, "--beam-size", 1, "--output", tmp_path / method / "predictions.jsonl")
        assert all(Path(row["en_audio"]).is_file() for row in result)
