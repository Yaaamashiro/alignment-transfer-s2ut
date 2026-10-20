import numpy as np
import pytest
import torch
from alignment_transfer_s2ut.alignment import source_mapping, target_mapping, compose, viterbi_spans
from alignment_transfer_s2ut.guidance.attention import attention, alignment_kl, beta_at
from alignment_transfer_s2ut.units import reduce_units, frame_spans, run_spans


def test_teacher_composition_exact_and_invalid_rows():
    mt = np.array([[0.8, 0.2], [0.1, 0.9]])
    src = np.array([[0.5, 0.5, 0], [0, 0, 1]])
    tgt = np.array([[1, 0], [0.25, 0.75], [0, 0]])
    result = compose(mt, src, tgt)
    np.testing.assert_allclose(result, [[0.4, 0.4, 0.2], [0.1375, 0.1375, 0.725], [0, 0, 0]])
    np.testing.assert_allclose(result[:2].sum(-1), 1)
    assert np.isfinite(result).all()


def test_time_mapping_uses_runtime_encoder_length_and_overlap():
    spans = [[0, 0.5], [0.5, 1], [-1, -1]]
    np.testing.assert_allclose(source_mapping(spans, 1, 4),
                               [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0, 0]])
    assert source_mapping(spans, 1, 7).shape == (3, 7)
    np.testing.assert_allclose(target_mapping([[0, 0.4], [0.4, 1]], [[0.2, 0.6], [1.1, 1.2]]),
                               [[0.5, 0.5], [0, 0]])


def test_unit_runs_reconstruct_and_preserve_frame_timing():
    original = np.array([7, 7, 7, 12, 12, 5, 5, 5])
    reduced, runs = reduce_units(original)
    np.testing.assert_array_equal(reduced, [7, 12, 5])
    np.testing.assert_array_equal(runs, [[0, 3], [3, 5], [5, 8]])
    np.testing.assert_array_equal(np.repeat(reduced, np.diff(runs, axis=1).ravel()), original)
    frames = frame_spans(8, 16000, 0.2, [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2])
    assert run_spans(runs, frames)[0, 1] == frames[2, 1]


def test_ctc_repeated_labels_require_blank_and_viterbi_spans():
    probs = np.log(np.array([[0.01, 0.99], [0.99, 0.01], [0.01, 0.99]]))
    spans = viterbi_spans(probs, [1, 1], 0, 0.3)
    np.testing.assert_allclose(spans, [[0, 0.1], [0.2, 0.3]])
    with pytest.raises(ValueError, match="too few"):
        viterbi_spans(probs[:2], [1, 1], 0, 0.2)


def test_beta_zero_uniform_prior_padding_and_gradients():
    torch.manual_seed(42)
    scores = torch.randn(2, 3, 4, 5, requires_grad=True)
    mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
    prior = torch.rand(2, 4, 5).masked_fill(mask[:, None], 0)
    raw, beta_zero = attention(scores, prior, beta=0, padding_mask=mask)
    assert torch.equal(raw, beta_zero)
    _, uniform = attention(scores, torch.ones_like(prior), beta=2, padding_mask=mask)
    torch.testing.assert_close(raw, uniform)
    raw, guided = attention(scores, prior, beta=1, padding_mask=mask)
    assert not torch.allclose(raw, guided)
    assert guided.masked_select(mask[:, None, None].expand_as(guided)).eq(0).all()
    torch.testing.assert_close(guided.sum(-1), torch.ones(2, 3, 4))
    loss = alignment_kl(prior / prior.sum(-1, keepdim=True), raw.mean(1))
    loss.backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


def test_invalid_teacher_rows_do_not_bias_or_contribute_kl():
    logits = torch.randn(1, 2, 3, 4, requires_grad=True)
    teacher = torch.zeros(1, 3, 4)
    raw, guided = attention(logits, teacher, 1)
    assert torch.equal(raw, guided)
    assert alignment_kl(teacher, raw.mean(1)).item() == 0


def test_annealing_and_inference():
    assert beta_at(0, "annealed_guidance", anneal_steps=10) == 1
    assert beta_at(5, "annealed_guidance", anneal_steps=10) == 0.5
    assert beta_at(10, "annealed_guidance", anneal_steps=10) == 0
    assert beta_at(100, "annealed_guidance", anneal_steps=10) == 0
    for method in ("baseline", "distillation", "guidance", "annealed_guidance"):
        assert beta_at(0, method, training=False) == 0
