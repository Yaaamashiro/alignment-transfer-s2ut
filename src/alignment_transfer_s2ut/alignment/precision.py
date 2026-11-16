"""No-update calibration precision checks on the registered logical batches."""

from contextlib import contextmanager
from itertools import combinations
import math

import torch


SCALES = (128, 1024, 2048)
PREFERENCE = (1024, 128, 2048)
# Diagnostic acceptance limits, independent of the scientific calibration parameters.
LIMITS = {
    "same_forward": {"min_cosine": .995, "max_relative_l2": .10, "max_zero_fraction_delta": .01},
    "fp32_reference": {"min_cosine": .95, "max_relative_norm": .05, "max_zero_fraction_delta": .02},
}


@contextmanager
def fp32_reference(model, criterion):
    """Widen the registered initial values and restore the original storage afterward."""
    tensors = [(tensor, tensor.data) for module in (model, criterion)
               for tensor in (*module.parameters(), *module.buffers()) if tensor.is_floating_point()]
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        for tensor, saved in tensors:
            tensor.data = saved.to(dtype=torch.float32, copy=True)
        yield
    finally:
        for tensor, saved in tensors:
            tensor.data = saved
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def vectorize(gradients, parameters):
    return torch.cat([g.reshape(-1) if g is not None else p.new_zeros(p.numel(), dtype=torch.float32)
                      for g, p in zip(gradients, parameters, strict=True)])


def gradient_metrics(vector):
    if vector.dtype != torch.float32:
        raise TypeError("the accumulated calibration gradient must be FP32")
    nonfinite = int((~torch.isfinite(vector)).sum())
    norm = None if nonfinite else float(torch.linalg.vector_norm(vector))
    return dict(l2=norm if norm is not None and math.isfinite(norm) else None,
                nonfinite=nonfinite, zero_fraction=float((vector == 0).float().mean()))


def difference(left, right, a, b):
    if not a["l2"] or not b["l2"]:
        return dict(cosine=None, relative_l2=None, relative_norm=None, zero_fraction_delta=None)
    norm = b["l2"]
    cosine = float(torch.dot(left / a["l2"], right / norm))
    delta = float(torch.linalg.vector_norm(left - right)) / norm
    if not math.isfinite(cosine) or not math.isfinite(delta):
        return dict(cosine=None, relative_l2=None, relative_norm=None, zero_fraction_delta=None)
    return dict(cosine=max(-1., min(1., cosine)),
                relative_l2=delta,
                relative_norm=abs(a["l2"] - norm) / norm,
                zero_fraction_delta=abs(a["zero_fraction"] - b["zero_fraction"]))


def compare_scales(gradients, parameters, reference):
    vectors = {str(scale): {kind: vectorize(value, parameters) for kind, value in objectives.items()}
               for scale, objectives in gradients.items()}
    if reference is not None:
        vectors["fp32"] = {kind: vectorize(value, parameters) for kind, value in reference.items()}
    metrics = {key: {kind: gradient_metrics(vector) for kind, vector in objectives.items()}
               for key, objectives in vectors.items()}
    pairs = {}
    for left, right in combinations(vectors, 2):
        pairs[left + "/" + right] = {
            kind: difference(vector, vectors[right][kind], metrics[left][kind], metrics[right][kind])
            for kind, vector in vectors[left].items()
        }
    return dict(metrics=metrics, comparisons=pairs)


def passes(comparison, limits):
    return comparison["cosine"] is not None and comparison["cosine"] >= limits["min_cosine"] and all(
        comparison[key[4:]] <= value for key, value in limits.items() if key.startswith("max_"))


def qualify(batches):
    """A candidate needs a stable FP16 neighbour and FP32 agreement on every batch."""
    receipts = {}
    for scale in PREFERENCE:
        key = str(scale)
        reference_failures = [
            index for index, batch in enumerate(batches, 1)
            if not batch["comparisons"].get(key + "/fp32") or not all(
                passes(result, LIMITS["fp32_reference"])
                for result in batch["comparisons"][key + "/fp32"].values())
        ]
        neighbour_failures = {}
        for neighbour in SCALES:
            if neighbour == scale:
                continue
            left, right = sorted((neighbour, scale))
            pair = f"{left}/{right}"
            neighbour_failures[str(neighbour)] = [
                index for index, batch in enumerate(batches, 1)
                if not all(passes(result, LIMITS["same_forward"])
                           for result in batch["comparisons"][pair].values())
            ]
        receipts[key] = dict(fp32_failed_batches=reference_failures, neighbour_failed_batches=neighbour_failures,
                             qualified=not reference_failures and any(not failures for failures in neighbour_failures.values()))
    selected = next((scale for scale in PREFERENCE if receipts[str(scale)]["qualified"]), None)
    return selected, receipts
