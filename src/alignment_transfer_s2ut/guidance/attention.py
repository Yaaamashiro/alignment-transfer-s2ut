import math
import torch


def beta_at(step, method, beta=1.0, beta_start=1.0, beta_end=0.0, anneal_steps=10000, training=True):
    if not training or method in {"baseline", "distillation"}:
        return 0.0
    if method == "guidance":
        return beta
    if method != "annealed_guidance" or anneal_steps < 1:
        raise ValueError("invalid alignment method or anneal_steps")
    fraction = min(1.0, max(0.0, step / anneal_steps))
    return beta_start + fraction * (beta_end - beta_start)


def smooth_prior(prior, padding_mask=None, alpha=0.05):
    """Mix valid rows with uniform acoustic mass; never revive missing/EOS rows."""
    if not 0 <= alpha <= 1:
        raise ValueError("prior smoothing must be in [0, 1]")
    prior = prior.detach().float()
    if not torch.isfinite(prior).all() or (prior < 0).any():
        raise ValueError("invalid guidance prior")
    allowed = torch.ones_like(prior[:, :1], dtype=torch.bool) if padding_mask is None else ~padding_mask[:, None]
    masked = prior * allowed
    mass = masked.sum(-1, keepdim=True)
    uniform = allowed / allowed.sum(-1, keepdim=True).clamp_min(1)
    return torch.where(mass > 0, (1 - alpha) * masked / mass.clamp_min(1e-30) + alpha * uniform,
                       torch.zeros_like(masked))


def attention(logits, prior=None, beta=0.0, padding_mask=None, eps=1e-8, prior_smoothing=0.0):
    scores = logits.float()
    if padding_mask is not None:
        if padding_mask.all(-1).any():
            raise ValueError("all source positions are padded")
        scores = scores.masked_fill(padding_mask[:, None, None, :], -torch.inf)
    raw = torch.softmax(scores, -1)
    if beta == 0 or prior is None:
        return raw, raw
    if not math.isfinite(beta) or not torch.isfinite(prior).all() or (prior < 0).any():
        raise ValueError("invalid guidance prior or coefficient")
    prior = smooth_prior(prior, padding_mask, prior_smoothing)
    valid = prior.sum(-1, keepdim=True) > 0
    bias = torch.where(valid, torch.log(prior + eps), torch.zeros_like(prior))
    guided = torch.softmax(scores + beta * bias[:, None, :, :], -1)
    return raw, guided


def alignment_kl(teacher, student, eps=1e-8, reduction="mean"):
    teacher = teacher.detach().float()
    student = student.float()
    valid = teacher.sum(-1) > 0
    values = torch.where(teacher > 0, teacher * (teacher.clamp_min(eps).log() -
                         student.clamp_min(eps).log()), torch.zeros_like(teacher)).sum(-1)
    if reduction == "none":
        return values[valid]
    if reduction == "sum":
        return values[valid].sum()
    if reduction != "mean":
        raise ValueError("unknown KL reduction")
    return values[valid].mean() if valid.any() else student.sum() * 0


def entropy(matrix):
    return -(matrix * matrix.clamp_min(1e-8).log()).sum(-1)
