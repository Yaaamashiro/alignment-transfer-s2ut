import numpy as np


def normalize(matrix, eps=1e-12):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("expected a finite nonnegative matrix")
    sums = matrix.sum(-1, keepdims=True)
    return np.divide(matrix, np.maximum(sums, eps), out=np.zeros_like(matrix), where=sums > eps)


def overlap(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != (len(left), 2) or right.shape != (len(right), 2):
        raise ValueError("spans must be [N,2]")
    valid = (left[:, 0] >= 0) & (left[:, 1] > left[:, 0])
    valid_right = (right[:, 0] >= 0) & (right[:, 1] > right[:, 0])
    result = np.maximum(0, np.minimum(left[:, None, 1], right[None, :, 1]) -
                        np.maximum(left[:, None, 0], right[None, :, 0]))
    return result * valid[:, None] * valid_right[None, :]


def source_mapping(spans, duration, encoder_length):
    if duration <= 0 or encoder_length < 1:
        raise ValueError("positive speech duration and encoder length required")
    boundaries = np.linspace(0, duration, encoder_length + 1)
    return normalize(overlap(spans, np.stack([boundaries[:-1], boundaries[1:]], axis=-1)))


def target_mapping(token_spans, unit_spans):
    return normalize(overlap(unit_spans, token_spans))


def compose(mt, src, tgt):
    mt, src, tgt = map(normalize, (mt, src, tgt))
    if mt.shape[1] != src.shape[0] or tgt.shape[1] != mt.shape[0]:
        raise ValueError("teacher mapping dimensions disagree")
    return normalize(tgt @ mt @ src)


def project_packet(packet, encoder_length):
    return compose(packet["mt_attention"],
                   source_mapping(packet["source_spans"], float(packet["source_duration"]), encoder_length),
                   target_mapping(packet["target_spans"], packet["unit_spans"]))
