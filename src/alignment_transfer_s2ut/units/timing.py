import numpy as np


def reduce_units(original):
    original = np.asarray(original, dtype=np.int64)
    if original.ndim != 1 or not len(original) or (original < 0).any() or (original >= 100).any():
        raise ValueError("expected nonempty KM100 unit sequence")
    starts = np.r_[0, np.flatnonzero(original[1:] != original[:-1]) + 1]
    runs = np.stack([starts, np.r_[starts[1:], len(original)]], axis=-1)
    return original[starts], runs


def frame_spans(count, sample_rate, duration, kernels, strides):
    receptive, jump = 1, 1
    for kernel, stride in zip(kernels, strides):
        receptive += (kernel - 1) * jump
        jump *= stride
    centers = ((receptive - 1) / 2 + np.arange(count) * jump) / sample_rate
    return np.clip(np.stack([centers - jump / sample_rate / 2,
                             centers + jump / sample_rate / 2], axis=-1), 0, duration)


def run_spans(runs, frames):
    runs = np.asarray(runs, dtype=np.int64)
    return np.stack([frames[runs[:, 0], 0], frames[runs[:, 1] - 1, 1]], axis=-1)
