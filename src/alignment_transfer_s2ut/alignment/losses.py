"""Loss-only layer-wise supervision. Priors never enter the model forward."""
import math
import torch

METHODS = ("baseline", "distillation", "annealed_distillation", "mask_distillation")
METHOD_LABELS = {
    "baseline": "Baseline",
    "distillation": "KL-based Attention Distillation",
    "annealed_distillation": "Annealed KL-based Attention Distillation",
    "mask_distillation": "Attention-level Knowledge Distillation",
}


def distributions(teacher, student, source_padding=None):
    if teacher.ndim == 3:
        teacher, student = teacher[:, None], student[:, None]
    if teacher.ndim != 4 or teacher.shape != student.shape:
        raise ValueError("alignment requires matching [B,L,U,F] tensors")
    teacher, student = teacher.detach().float(), student.float()
    if not torch.isfinite(teacher).all() or (teacher < 0).any() or not torch.isfinite(student).all() or (student < 0).any():
        raise ValueError("nonfinite/negative alignment probabilities")
    allowed = torch.ones((teacher.size(0),1,1,teacher.size(-1)), dtype=torch.bool, device=teacher.device)
    if source_padding is not None:
        allowed = allowed & ~source_padding[:, None, None, :]
    teacher = teacher.masked_fill(~allowed, 0)
    mass = teacher.sum(-1, keepdim=True)
    valid = mass.squeeze(-1) > 0
    teacher = teacher / torch.where(mass > 0, mass, torch.ones_like(mass))
    return teacher, student, valid, allowed


def top_p_mask(teacher, p, source_padding=None):
    if not math.isfinite(p) or not 0 < p <= 1:
        raise ValueError("top-p must be explicitly selected in (0,1]")
    original_ndim = teacher.ndim
    prior, _, valid, allowed = distributions(teacher, teacher, source_padding)
    # Stable descending sort resolves ties by ascending original source index.
    values, indices = torch.sort(prior, dim=-1, descending=True, stable=True)
    before = values.cumsum(-1) - values
    selected = (before < p) & (values > 0)
    mask = torch.zeros_like(selected).scatter(-1, indices, selected)
    mask = mask & allowed & valid[..., None]
    return mask[:, 0] if original_ndim == 3 else mask


def layer_statistics(teacher, student, kind="kl", top_p=None, source_padding=None):
    prior, raw, valid, allowed = distributions(teacher, student, source_padding)
    counts = valid.sum((0, 2))
    if not counts.any():
        raise ValueError("no valid teacher rows in training microbatch")
    kl_rows = torch.where(prior > 0, prior * (prior.clamp_min(1e-30).log() - raw.clamp_min(1e-30).log()), 0).sum(-1)
    def means(rows):
        return (rows * valid).sum((0, 2)) / counts.clamp_min(1)
    teacher_entropy = means(-(prior * prior.clamp_min(1e-30).log()).sum(-1))
    student_entropy = means(-(raw * raw.clamp_min(1e-30).log()).sum(-1))
    result = dict(kl=means(kl_rows), counts=counts, teacher_entropy=teacher_entropy, student_entropy=student_entropy)
    if kind == "mask":
        mask = top_p_mask(prior, top_p, source_padding)
        mass = (mask * raw).sum(-1)
        result.update(losses=means(1 - mass), region_mass=means(mass),
            coverage=means(mask.sum(-1) / allowed.sum(-1).clamp_min(1)), selected_frames=(mask * valid[..., None]).sum((0, 2, 3)))
    elif kind == "kl":
        result["losses"] = result["kl"]
    else:
        raise ValueError("unknown alignment loss")
    return result


def worst_layer(losses, tau, counts=None):
    if not math.isfinite(tau) or tau <= 0 or losses.ndim != 1 or not torch.isfinite(losses).all():
        raise ValueError("finite positive calibrated tau and finite layer losses required")
    valid = torch.ones_like(losses, dtype=torch.bool) if counts is None else counts > 0
    if not valid.any():
        raise ValueError("no supervised decoder layers")
    center = losses.masked_fill(~valid, -torch.inf).max().detach()
    logits = ((losses - center) / tau).masked_fill(~valid, -torch.inf)
    loss = center + tau * (torch.logsumexp(logits, 0) - math.log(int(valid.sum())))
    return loss, torch.softmax(logits, 0)


def lambda_at(step, method, initial, total_updates, anneal_fraction, training=True):
    if method not in METHODS or not math.isfinite(initial) or initial < 0 or total_updates < 1 or not 0 < anneal_fraction <= 1:
        raise ValueError("invalid distillation schedule")
    if not training or method == "baseline":
        return 0.
    if method != "annealed_distillation":
        return initial
    end = anneal_fraction * total_updates
    if step >= end:
        return 0.
    return initial * (1 + math.cos(math.pi * max(0, step) / end)) / 2


def annealing_policy(policy):
    fraction = policy["anneal_fraction"]
    if policy["anneal_schedule"] != "cosine":
        raise ValueError("anneal_schedule must be cosine")
    if isinstance(fraction, bool) or not isinstance(fraction, (int,float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("anneal_fraction must be finite and in (0,1]")
    return fraction


def loss_and_logs(teacher, student, policy, step, budget, source_padding=None):
    kind = "mask" if policy["method"] == "mask_distillation" else "kl"
    parameters = policy["calibrated"]
    tau = parameters["tau_" + kind]
    fraction = annealing_policy(policy)
    coefficient = lambda_at(step, policy["method"], parameters["lambda_" + kind], budget, fraction)
    stats = layer_statistics(teacher, student, kind, policy.get("mask_top_p"), source_padding)
    loss, weights = worst_layer(stats["losses"], tau, stats["counts"])
    valid_layers = stats["counts"] > 0
    logs = {"alignment/loss": loss.detach(), "alignment/lambda": coefficient, "alignment/tau": tau,
            "alignment/worst_layer": int(stats["losses"].masked_fill(stats["counts"] == 0, -torch.inf).argmax()),
            "alignment/teacher_entropy": stats["teacher_entropy"][valid_layers].mean().detach(),
            "alignment/student_entropy": stats["student_entropy"][valid_layers].mean().detach(),
            "alignment/kl_mean": stats["kl"][valid_layers].mean().detach(), "alignment/kl_worst": stats["kl"][valid_layers].max().detach()}
    for layer in range(len(weights)):
        logs.update({f"alignment/D_{layer}": stats["losses"][layer].detach(),
                     f"alignment/w_{layer}": weights[layer].detach(), f"alignment/kl_layer_{layer}": stats["kl"][layer].detach(),
                     f"alignment/valid_rows_{layer}": stats["counts"][layer].detach()})
        if kind == "mask":
            for key in ("region_mass", "coverage", "selected_frames"):
                logs[f"alignment/{key}_{layer}"] = stats[key][layer].detach()
    if kind == "mask":
        logs["alignment/top_p"] = policy["mask_top_p"]
    if policy["method"] == "annealed_distillation":
        end = fraction * budget
        logs.update({"alignment/annealing_progress": min(1., step / end),
                     "alignment/task_only_start": math.ceil(end) if step >= end else -1,
                     "alignment/task_only_phase": float(step >= end)})
    return coefficient * loss, logs
