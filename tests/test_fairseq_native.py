"""Run these in the supported Python 3.10 fairseq environment, without external assets."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch

pytestmark = [pytest.mark.fairseq,
              pytest.mark.skipif(importlib.util.find_spec("fairseq") is None, reason="pinned fairseq not installed")]


def test_native_baseline_beta_zero_and_guidance_context():
    from alignment_transfer_s2ut.config import BackboneConfig
    from alignment_transfer_s2ut.s2ut.model import build_model
    from alignment_transfer_s2ut.s2ut.fairseq_extension import TransferAttention
    from alignment_transfer_s2ut.training import UNIT_VOCABULARY
    from fairseq.modules import MultiheadAttention
    config = BackboneConfig(encoder_layers=1, decoder_layers=1, encoder_embed_dim=16,
                            decoder_embed_dim=16, encoder_ffn_embed_dim=32, decoder_ffn_embed_dim=32,
                            encoder_attention_heads=2, decoder_attention_heads=2,
                            dropout=0, attention_dropout=0, activation_dropout=0, conv_channels=32,
                            input_feat_per_channel=8)
    model = build_model(config, "speech", [], UNIT_VOCABULARY).eval()
    source, lengths, previous = torch.randn(2, 40, 8), torch.tensor([40, 31]), torch.tensor([[2, 11, 12], [2, 5, 6]])
    stock = copy.deepcopy(model)
    stock.network.decoder.layers[-1].encoder_attn.__class__ = MultiheadAttention
    with torch.no_grad():
        expected = stock.network(source, lengths, previous)[0]
        actual, _, _, _ = model(source, lengths, previous)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        encoded = model.encode(source, lengths)
        size = encoded["encoder_out"][0].size(0)
        uniform = lambda lengths, target: torch.ones(2, target, size) / size
        zero, _, _, _ = model(source, lengths, previous, uniform, beta=0)
        torch.testing.assert_close(zero, expected, atol=0, rtol=0)
        guided_uniform, _, _, _ = model(source, lengths, previous, uniform, beta=1)
        torch.testing.assert_close(guided_uniform, expected, atol=2e-6, rtol=2e-6)
    def biased(lengths, target):
        prior = torch.full((2, target, size), 0.001)
        prior[:, :, 0] = 1
        return prior / prior.sum(-1, keepdim=True)
    logits, raw, guided, prior = model(source, lengths, previous, biased, beta=1)
    assert not torch.allclose(raw, guided)
    assert not torch.allclose(logits, expected)
    logits.sum().backward()
    assert model.network.decoder.layers[-1].encoder_attn.q_proj.weight.grad.abs().sum() > 0


def test_native_text_and_speech_use_same_layer_types_and_dimensions():
    from alignment_transfer_s2ut.config import BackboneConfig
    from alignment_transfer_s2ut.s2ut.model import build_model
    from alignment_transfer_s2ut.tokenization.character import SPECIAL
    from alignment_transfer_s2ut.training import UNIT_VOCABULARY
    config = BackboneConfig(encoder_layers=1, decoder_layers=1, conv_channels=32)
    speech = build_model(config, "speech", [], UNIT_VOCABULARY)
    text = build_model(config, "text", SPECIAL + ["a", "あ"], SPECIAL + ["b"])
    assert type(speech.network.encoder.transformer_layers[0]) is type(text.network.encoder.transformer_layers[0])
    assert type(speech.network.decoder.layers[0]) is type(text.network.decoder.layers[0])
    assert type(speech.network.decoder.layers[0].encoder_attn) is type(text.network.decoder.layers[0].encoder_attn)
    speech_shapes = {k: tuple(v.shape) for k, v in speech.network.encoder.transformer_layers.state_dict().items()}
    text_shapes = {k: tuple(v.shape) for k, v in text.network.encoder.transformer_layers.state_dict().items()}
    assert speech_shapes == text_shapes


def test_native_training_cli_and_visualization_with_synthetic_supervision(tmp_path):
    """Use real native layers with deliberately synthetic audio/alignment/units, no downloaded weights."""
    import numpy as np
    import soundfile as sf
    import yaml
    from alignment_transfer_s2ut.smoke_pipeline import TINY_CONFIG, invoke, synthetic_frontends
    from alignment_transfer_s2ut.io import write_jsonl, read_json
    from alignment_transfer_s2ut.compare_experiments import compare_experiments
    torch.set_num_threads(1)
    rows = []
    for i, (split, ja, en) in enumerate([("train", "あい", "ab"), ("dev", "いあ", "ba"), ("test", "ああ", "aa")]):
        row = dict(pair_id=split, split=split, ja_text=ja, en_text=en)
        for lang in ("ja", "en"):
            path = tmp_path / f"{i}-{lang}.wav"
            sf.write(path, 0.1 * np.sin(2 * np.pi * (200 + i * 50) * np.arange(8000) / 16000), 16000)
            row[lang + "_audio"] = str(path)
        rows.append(row)
    write_jsonl(tmp_path / "input.jsonl", rows)
    manifest = tmp_path / "manifest.jsonl"
    invoke("prepare_manifest", "--input", tmp_path / "input.jsonl", "--output", manifest)
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY_CONFIG), encoding="utf-8")
    common = ["--manifest", manifest, "--config", config, "--batch-size", 1,
              "--update-frequency", 1, "--max-updates", 2, "--warmup-updates", 0]
    invoke("train_t2tt", *common, "--output", tmp_path / "t2tt")
    bleu = invoke("evaluate_t2tt", "--manifest", manifest, "--checkpoint", tmp_path / "t2tt/checkpoint_best.pt",
                  "--beam-size", 2, "--max-chars", 8, "--output", tmp_path / "t2tt-dev-text.json")
    assert bleu["backend"] == "fairseq" and bleu["samples"] == 1 and 0 <= bleu["text_bleu"] <= 100
    invoke("extract_t2tt_attention", "--manifest", manifest, "--checkpoint", tmp_path / "t2tt/checkpoint_best.pt",
           "--output", tmp_path / "mt")
    synthetic_frontends(manifest, tmp_path / "ctc", tmp_path / "units")
    invoke("build_teacher_alignment", "--manifest", manifest, "--mt-attention", tmp_path / "mt",
           "--alignment", tmp_path / "ctc", "--units", tmp_path / "units", "--config", config,
           "--output", tmp_path / "teacher")
    methods = ["baseline", "distillation", "guidance", "annealed_guidance"]
    for method in methods:
        invoke("train_s2ut", "--trainer", "legacy", *common, "--units", tmp_path / "units", "--teacher", tmp_path / "teacher",
               "--method", method, "--anneal-steps", 1, "--output", tmp_path / method)
    assert compare_experiments([tmp_path / m for m in methods])["fair_comparison"]
    figure = invoke("visualize_attention", "--manifest", manifest, "--pair-id", "train",
                    "--checkpoint", tmp_path / "guidance/checkpoint_best.pt", "--teacher", tmp_path / "teacher",
                    "--units", tmp_path / "units", "--output", tmp_path / "figures")
    assert Path(figure).stat().st_size > 1000
