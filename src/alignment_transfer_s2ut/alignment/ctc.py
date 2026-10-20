"""CTC forced Viterbi paths, including blank states and repeated labels."""
import numpy as np


def viterbi_spans(log_probs, labels, blank, duration):
    log_probs = np.asarray(log_probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if log_probs.ndim != 2 or not np.isfinite(log_probs).all() or duration <= 0:
        raise ValueError("finite [frames,vocabulary] CTC log probabilities required")
    if not len(labels) or (labels == blank).any() or (labels < 0).any() or (labels >= log_probs.shape[1]).any():
        raise ValueError("invalid CTC transcript labels")
    frames = len(log_probs)
    if frames < len(labels) + int((labels[1:] == labels[:-1]).sum()):
        raise ValueError("impossible CTC alignment: too few acoustic frames")
    states = np.full(2 * len(labels) + 1, blank, dtype=np.int64)
    states[1::2] = labels
    prev = np.full(len(states), -np.inf)
    prev[:2] = log_probs[0, states[:2]]
    back = np.zeros((frames, len(states)), dtype=np.int8)
    for t in range(1, frames):
        candidates = np.stack([prev, np.r_[-np.inf, prev[:-1]],
                               np.r_[-np.inf, -np.inf, prev[:-2]]])
        skip = (states != blank) & (states != np.r_[blank, blank, states[:-2]])
        candidates[2, ~skip] = -np.inf
        best = candidates.argmax(0)
        back[t] = best
        prev = candidates[best, np.arange(len(states))] + log_probs[t, states]
    state = len(states) - 2 + int(prev[-1] > prev[-2])
    if not np.isfinite(prev[state]):
        raise ValueError("no valid forced CTC path")
    path = np.zeros(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    spans = []
    for index in range(len(labels)):
        selected = np.flatnonzero(path == 2 * index + 1)
        if not len(selected):
            raise ValueError("CTC label has no Viterbi span")
        spans.append([selected[0] * duration / frames, (selected[-1] + 1) * duration / frames])
    return np.asarray(spans, dtype=np.float64)
