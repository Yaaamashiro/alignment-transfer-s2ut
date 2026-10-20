"""Local receptive-field supports before the contextual Transformer encoder.

These supports describe the acoustic frontend, not a restriction on self-attention.
Kaldi FBANK uses 25 ms windows, 10 ms shifts and snip_edges=True. Smoke STFT
uses centered windows. Convolution padding contributes no observed audio.
"""
import numpy as np


def output_lengths(lengths, config, backend="fairseq"):
    if backend == "smoke":
        return (lengths + 3) // 4
    for _ in config.conv_kernel_sizes.split(","):
        lengths = (lengths + 1) // 2
    return lengths


def encoder_time_spans(feature_length, encoder_length, config, backend, duration):
    if feature_length < 1 or duration <= 0 or int(output_lengths(feature_length, config, backend)) != encoder_length:
        raise ValueError("frontend length/timing mismatch")
    jump, left, right = 1, 0, 0
    kernels = [5] if backend == "smoke" else [int(k) for k in config.conv_kernel_sizes.split(",")]
    for kernel in kernels:
        # Native Conv1d uses stride=2, padding=k//2 (smoke stride=4).
        left += (kernel // 2) * jump
        right += (kernel - 1 - kernel // 2) * jump
        jump *= 4 if backend == "smoke" else 2
    centers = np.arange(encoder_length) * jump
    first = np.maximum(0, centers - left)
    last = np.minimum(feature_length - 1, centers + right)
    offset = -0.0125 if backend == "smoke" else 0.0
    return np.clip(np.stack([first * .01 + offset, last * .01 + offset + .025], -1), 0, duration)


def monotonic_prior(unit_spans, target_duration, source_spans, source_duration, sigma=0.1):
    """A Gaussian diagonal using observed run times, with no text/alignment input."""
    from .projection import normalize
    if sigma <= 0 or not np.isfinite(sigma) or min(target_duration, source_duration) <= 0:
        raise ValueError("positive monotonic width and durations required")
    target = np.asarray(unit_spans).mean(-1) / target_duration
    source = np.asarray(source_spans).mean(-1) / source_duration
    scores = -.5 * ((target[:, None] - source[None]) / sigma) ** 2
    scores -= scores.max(-1, keepdims=True)
    return normalize(np.exp(scores))
