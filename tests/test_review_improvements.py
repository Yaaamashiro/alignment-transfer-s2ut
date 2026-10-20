"""Regression coverage for the six implementation-review changes."""
from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from alignment_transfer_s2ut.config import BackboneConfig
from alignment_transfer_s2ut.guidance.attention import smooth_prior, attention, alignment_kl
from alignment_transfer_s2ut.alignment.timing import encoder_time_spans, output_lengths, monotonic_prior
from alignment_transfer_s2ut.io import ArtifactStore, read_json, read_jsonl, write_jsonl
from alignment_transfer_s2ut.smoke_pipeline import TINY_CONFIG, invoke, synthetic_frontends
from alignment_transfer_s2ut.training import load_checkpoint


def test_smoothing_preserves_missing_rows_padding_and_beta_zero():
    prior = torch.tensor([[[1., 0, 0, 9], [0., 0, 0, 0], [0., 1, 0, 0]]])
    mask = torch.tensor([[False, False, False, True]])
    effective = smooth_prior(prior, mask, .1)
    torch.testing.assert_close(effective[0, 0], torch.tensor([.9 + .1 / 3, .1 / 3, .1 / 3, 0]))
    assert effective[0, 1].eq(0).all()
    assert effective[..., -1].eq(0).all()
    scores = torch.randn(1, 2, 3, 4)
    raw, unchanged = attention(scores, prior, 0, mask, prior_smoothing=.1)
    assert torch.equal(raw, unchanged)
    _, guided = attention(scores, prior, 1, mask, prior_smoothing=.1)
    torch.testing.assert_close(guided[:, :, 1], raw[:, :, 1])
    assert (guided[:, :, 0, :3] > 0).all()


def test_kl_accumulation_matches_union_for_unequal_valid_rows():
    teacher = torch.tensor([[[1., 0], [0., 0], [0., 0]], [[.1, .9], [.5, .5], [.8, .2]]])
    scores = torch.randn(2, 3, 2, requires_grad=True)
    reference = alignment_kl(teacher, scores.softmax(-1))
    expected = torch.autograd.grad(reference, scores)[0]
    accumulated = sum(alignment_kl(teacher[i:i+1], scores[i:i+1].softmax(-1), reduction="sum")
                      for i in range(2)) / 4
    actual = torch.autograd.grad(accumulated, scores)[0]
    torch.testing.assert_close(reference, accumulated)
    torch.testing.assert_close(expected, actual)


def test_frontend_timing_matches_convolution_support_and_padding():
    cfg = BackboneConfig()
    signal = torch.ones(1, 1, 100, requires_grad=True)
    value = signal
    for _ in range(2):
        value = torch.nn.functional.conv1d(value, torch.ones(1, 1, 5), stride=2, padding=2)
    assert value.size(-1) == output_lengths(100, cfg) == 25
    value[0, 0, 10].backward()
    supported = signal.grad[0, 0].nonzero().ravel().numpy()
    spans = encoder_time_spans(100, 25, cfg, "fairseq", 1.015)
    np.testing.assert_allclose(spans[10], [supported.min() * .01, supported.max() * .01 + .025])
    np.testing.assert_allclose(spans[0], [0, .085])
    np.testing.assert_allclose(spans[-1], [.90, 1.015])
    with pytest.raises(ValueError, match="mismatch"):
        encoder_time_spans(100, 24, cfg, "fairseq", 1.015)


def test_monotonic_prior_uses_time_and_contains_no_zero_rows():
    spans = np.array([[0, .1], [.45, .55], [.9, 1.]])
    prior = monotonic_prior(spans, 1, spans, 1, .1)
    np.testing.assert_array_equal(prior.argmax(-1), [0, 1, 2])
    np.testing.assert_allclose(prior.sum(-1), 1)


@pytest.fixture(scope="module")
def review_data(tmp_path_factory):
    import soundfile as sf
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("review")
    rows = []
    for i, (split, ja, en) in enumerate([("train", "あい", "ab"), ("train", "いあ", "ba"),
                                       ("dev", "あう", "az"), ("test", "ああ", "aa")]):
        row = dict(pair_id=str(i), split=split, ja_text=ja, en_text=en)
        for language in ("ja", "en"):
            path = root / f"{i}-{language}.wav"
            sf.write(path, .1 * np.sin(2 * np.pi * (200 + i * 30) * np.arange(8000) / 16000), 16000)
            row[language + "_audio"] = str(path)
        rows.append(row)
    write_jsonl(root / "input.jsonl", rows)
    invoke("prepare_manifest", "--input", root / "input.jsonl", "--output", root / "manifest.jsonl")
    (root / "tiny.yaml").write_text(yaml.safe_dump(TINY_CONFIG), encoding="utf-8")
    common = ["--manifest", root / "manifest.jsonl", "--config", root / "tiny.yaml", "--backend", "smoke",
              "--batch-size", 1, "--update-frequency", 1, "--max-updates", 1, "--warmup-updates", 0]
    invoke("train_t2tt", *common, "--output", root / "t2tt")
    invoke("extract_t2tt_attention", "--manifest", root / "manifest.jsonl", "--checkpoint",
           root / "t2tt/checkpoint_best.pt", "--output", root / "mt")
    synthetic_frontends(root / "manifest.jsonl", root / "ctc", root / "units")
    invoke("build_teacher_alignment", "--manifest", root / "manifest.jsonl", "--config", root / "tiny.yaml",
           "--mt-attention", root / "mt", "--alignment", root / "ctc", "--units", root / "units",
           "--output", root / "teacher")
    # Deliberately make one microbatch contain fewer valid rows than the other.
    store = ArtifactStore(root / "teacher")
    meta, packet = store.get("1")
    packet["target_spans"][0] = [-1, -1]
    store.put("1", meta, packet, overwrite=True)
    return root


def _train_args(root):
    return ["--manifest", root / "manifest.jsonl", "--config", root / "tiny.yaml", "--units", root / "units",
            "--max-updates", 1, "--warmup-updates", 0]


def test_trainer_kl_global_rows_matches_large_batch(review_data, tmp_path, monkeypatch):
    root = review_data
    common = [*_train_args(root), "--backend", "smoke", "--teacher", root / "teacher", "--method", "distillation"]
    gradients = []
    original_step = torch.optim.Adam.step
    def capture_step(optimizer, *args, **kwargs):
        gradients.append([p.grad.detach().clone() for group in optimizer.param_groups for p in group["params"]])
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", capture_step)
    invoke("train_s2ut", *common, "--batch-size", 2, "--update-frequency", 1, "--output", tmp_path / "large")
    invoke("train_s2ut", *common, "--batch-size", 1, "--update-frequency", 2, "--output", tmp_path / "micro")
    # Compare update gradients directly: Adam can amplify rounding in near-zero biases.
    for large, micro in zip(*gradients):
        torch.testing.assert_close(large, micro, atol=1e-6, rtol=2e-5)
    a = read_jsonl(tmp_path / "large/metrics.jsonl")[0]
    b = read_jsonl(tmp_path / "micro/metrics.jsonl")[0]
    assert a["train/alignment_loss"] == pytest.approx(b["train/alignment_loss"], abs=1e-6)


def test_monotonic_training_and_audit(review_data, tmp_path):
    root = review_data
    invoke("train_s2ut", *_train_args(root), "--backend", "smoke", "--batch-size", 2, "--update-frequency", 1,
           "--prior-type", "monotonic", "--method", "annealed_guidance", "--output", tmp_path / "mono")
    saved = load_checkpoint(tmp_path / "mono/checkpoint_last.pt")
    assert saved["teacher_signature"] is None
    assert read_jsonl(tmp_path / "mono/metrics.jsonl")[0]["alignment/teacher_entropy"] > 0
    from alignment_transfer_s2ut.diagnose_alignment import diagnose
    summary = diagnose(Namespace(manifest=root / "manifest.jsonl", config=root / "tiny.yaml", units=root / "units",
        teacher=root / "teacher", output=tmp_path / "audit", split="dev", max_samples=500, heatmaps=20,
        seed=1, source_mapping="frontend", prior_smoothing=.05, overwrite=False))
    assert summary["audited_pairs"] == summary["heatmaps"] == 1
    assert summary["valid_source_tokens_ratio"] < 1
    assert summary["valid_target_tokens_ratio"] < 1
    assert summary["smoothed_entropy"] >= summary["prior_entropy"]
    assert list((tmp_path / "audit/heatmaps").glob("*.png"))


@pytest.mark.fairseq
def test_strong_baseline_resume_and_teacher_free_inference(review_data, tmp_path):
    pytest.importorskip("fairseq")
    pytest.importorskip("sentencepiece")
    from alignment_transfer_s2ut.recipes import load_recipe
    root = review_data
    recipe = load_recipe("strong")
    recipe.pop("name")
    recipe["specaugment"]["freq_mask_F"] = 3
    recipe["specaugment"]["time_mask_T"] = 5
    cfg = recipe["auxiliary"]
    cfg.update(source_layer=1, target_layer=1, decoder_ctc_layer=1, ctc_vocab_size=16)
    cfg["decoder_args"].update(decoder_layers=1, decoder_embed_dim=16, decoder_ffn_embed_dim=32,
                               decoder_attention_heads=2, dropout=.1)
    recipe_path = tmp_path / "strong-tiny.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    common = [*_train_args(root), "--recipe", "strong", "--recipe-config", recipe_path,
              "--batch-size", 1, "--update-frequency", 2]
    invoke("train_s2ut", *common, "--max-updates", 2, "--output", tmp_path / "full")
    invoke("train_s2ut", *common, "--output", tmp_path / "resume")
    invoke("train_s2ut", *common, "--max-updates", 2, "--output", tmp_path / "resume", "--resume")
    full = load_checkpoint(tmp_path / "full/checkpoint_last.pt")
    resumed = load_checkpoint(tmp_path / "resume/checkpoint_last.pt")
    assert "z" not in full["auxiliary"]["target_vocabulary"]
    assert "う" not in full["auxiliary"]["source_vocabulary"]
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key]), key
    metrics = read_jsonl(tmp_path / "full/metrics.jsonl")[0]
    for name in ("source_letter", "target_letter", "decoder_target_ctc"):
        assert metrics["train/auxiliary_" + name] > 0
        assert any("network." + name + "_decoder" in key for key in full["model"])
    _check_native_auxiliary_criterion_parity(full, root)
    invoke("train_t2tt", "--manifest", root / "manifest.jsonl", "--config", root / "tiny.yaml",
           "--batch-size", 1, "--update-frequency", 1, "--max-updates", 1, "--output", tmp_path / "native-mt")
    invoke("extract_t2tt_attention", "--manifest", root / "manifest.jsonl", "--checkpoint",
           tmp_path / "native-mt/checkpoint_best.pt", "--output", tmp_path / "native-attention")
    invoke("build_teacher_alignment", "--manifest", root / "manifest.jsonl", "--config", root / "tiny.yaml",
           "--mt-attention", tmp_path / "native-attention", "--alignment", root / "ctc", "--units", root / "units",
           "--output", tmp_path / "native-teacher")
    # Same strong recipe across T2TT/monotonic annealed controls; no recipe mixing.
    for prior in ("monotonic", "t2tt"):
        invoke("train_s2ut", *common, "--max-updates", 2, "--method", "annealed_guidance", "--prior-type", prior,
               "--teacher", tmp_path / "native-teacher", "--output", tmp_path / prior)
    from alignment_transfer_s2ut.compare_experiments import compare_experiments
    roots = [tmp_path / name for name in ("full", "monotonic", "t2tt")]
    assert compare_experiments(roots)["fair_comparison"]
    from alignment_transfer_s2ut.io import write_json
    changed = read_json(tmp_path / "monotonic/experiment.json")
    changed["transfer"]["source_mapping"] = "uniform"
    write_json(tmp_path / "monotonic/experiment.json", changed, overwrite=True)
    assert "transfer/source_mapping" in compare_experiments(roots)["differences"]
    invoke("infer", "--checkpoint", tmp_path / "full/checkpoint_best.pt", "--audio", root / "3-ja.wav",
           "--output", tmp_path / "prediction.jsonl", "--units-only", "--beam-size", 1, "--max-units", 3)
    assert read_jsonl(tmp_path / "prediction.jsonl")[0]["inference_beta"] == 0


def _check_native_auxiliary_criterion_parity(saved, root):
    """Compare against the upstream multitask criterion, including its index rules."""
    import base64
    import sentencepiece as spm
    from fairseq.criterions.ctc import CtcCriterionConfig
    from fairseq.criterions.speech_to_speech_criterion import MultitaskCriterion
    from alignment_transfer_s2ut.recipes import auxiliary_losses, _targets, transform_features
    from alignment_transfer_s2ut.training import checkpoint_model, collate
    from alignment_transfer_s2ut.data.manifest import load_manifest
    from alignment_transfer_s2ut.tokenization.character import CharacterTokenizer
    from alignment_transfer_s2ut.data.audio import waveform
    from fairseq.data.audio.audio_utils import _get_torchaudio_fbank
    rows = [row for row in load_manifest(root / "manifest.jsonl") if row["split"] == "train"]
    model = checkpoint_model(saved).train()
    source, lengths, previous, target = collate(rows, "speech", None, None, "fairseq", ArtifactStore(root / "units"),
        8, "cpu", saved["recipe"]["speech_recipe"])
    # Verify the strong FBANK scale and native CMVN, separately from the trainer.
    wave, rate = waveform(rows[0]["ja_audio"])
    native = torch.from_numpy(_get_torchaudio_fbank((wave[None] * 32768).numpy(), rate, 8))
    torch.testing.assert_close(source[0, :lengths[0]], transform_features(native, {"cmvn": True}))
    model(source, lengths, previous)
    encoded, extra = model.auxiliary_states
    state = torch.get_rng_state()
    actual = sum(auxiliary_losses(model, rows, target).values())
    torch.set_rng_state(state)
    meta, tasks, multitask = saved["auxiliary"], {}, {}
    cfg = meta["config"]
    processor = spm.SentencePieceProcessor(model_proto=base64.b64decode(meta["sentencepiece_model"]))
    for name, field, layer, weight in (("source_letter", "ja_text", cfg["source_layer"], cfg["source_weight"]),
            ("target_letter", "en_text", cfg["target_layer"], cfg["target_weight"]),
            ("decoder_target_ctc", None, cfg["decoder_ctc_layer"], cfg["ctc_weight"])):
        if field:
            tokenizer = CharacterTokenizer(meta["source_vocabulary" if field == "ja_text" else "target_vocabulary"])
            sequences = [tokenizer.encode(row[field]) for row in rows]
            criterion_cfg = Namespace(sentence_avg=False, label_smoothing=cfg["label_smoothing"])
        else:
            sequences = [[i + 3 if i else 3 for i in processor.encode(row["en_text"])] for row in rows]
            criterion_cfg = CtcCriterionConfig(zero_infinity=False, sentence_avg=False)
        aux_target, aux_previous = _targets(sequences, "cpu")
        tasks[name] = Namespace(target_dictionary=getattr(model.network, name + "_decoder").dictionary,
            args=Namespace(decoder_type="transformer" if field else "ctc", input_layer=layer - 1,
                input_from="encoder" if field else "decoder", rdrop_alpha=0.0, criterion_cfg=criterion_cfg,
                get_loss_weight=lambda step, value=weight: value))
        multitask[name] = {"net_input": {"prev_output_tokens": aux_previous}, "target": aux_target,
                          "target_lengths": torch.tensor([len(seq) + (1 if field else 0) for seq in sequences]),
                          "ntokens": int(aux_target.ne(1).sum())}
    criterion = MultitaskCriterion(tasks)
    for name, task in tasks.items():
        criterion.set_multitask_loss_weight(name, task.args.get_loss_weight(0))
    expected, _ = criterion.get_multitask_loss(model.network, {"multitask": multitask,
        "target_lengths": target.ne(1).sum(-1), "id": torch.arange(len(rows))}, {**encoded, **extra})
    torch.testing.assert_close(actual, expected)
