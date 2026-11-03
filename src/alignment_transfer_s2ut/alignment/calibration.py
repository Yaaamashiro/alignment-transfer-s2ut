"""Pure calibration estimators; failures are explicit, without fallback constants."""
import math
import torch
from ..io import digest


def runtime_identity():
    import platform
    from importlib.metadata import version
    return dict(python=platform.python_version(),torch=str(torch.__version__),numpy=version("numpy"))


def median(values):
    return float(torch.tensor(values, dtype=torch.float64).median())


def calibrate_tau(layer_batches, ratio=3., minimum_delta=1e-8, valid_counts=None):
    if not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("target worst/median ratio must exceed one")
    estimates, rejected = [], []
    if valid_counts is not None and len(valid_counts) != len(layer_batches):
        raise ValueError("tau layer losses/counts must have matching microbatches")
    for i, losses in enumerate(layer_batches):
        values = torch.as_tensor(losses, dtype=torch.float64)
        if valid_counts is not None:
            counts = torch.as_tensor(valid_counts[i])
            if counts.shape != values.shape:
                raise ValueError("tau layer losses/counts must have matching layers")
            values = values[counts > 0]
        if not values.numel():
            rejected.append(i)
            continue
        delta = float(values.max() - values.median())
        if not torch.isfinite(values).all() or not math.isfinite(delta) or delta <= minimum_delta:
            rejected.append(i)
        else:
            estimates.append(delta / math.log(ratio))
    if not estimates:
        raise ValueError("tau calibration failure: all layer gaps are nonfinite or near zero")
    return dict(tau=median(estimates), estimates=estimates, rejected_microbatches=rejected,
                target_ratio=ratio, minimum_delta=minimum_delta, median_definition="lower order statistic")


def calibrate_lambda(norms, alpha=.5):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("positive gradient target alpha required")
    ratios = []
    for task, alignment in norms:
        if not all(math.isfinite(v) and v > 0 for v in (task, alignment)):
            raise ValueError("gradient calibration failure: zero/nonfinite task or alignment norm")
        ratios.append(alpha * task / alignment)
    if not ratios:
        raise ValueError("gradient calibration failure: no batches")
    return median(ratios)


def gradient_norm(gradients):
    values = [g.detach().double().square().sum() for g in gradients if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def binding(plan):
    return {key: plan[key] for key in ("baseline_record_hash", "scientific_config", "prepared", "initialization")} | {
        "artifacts": plan["transfer"]["artifacts"], "all_layers": plan["transfer"]["all_layers"]}


def validate_artifact(value, plan):
    if value.get("format") != "alignment-calibration-v2" or value.get("tau_calibration_unit") != "microbatch":
        raise ValueError("microbatch tau calibration v2 required; regenerate the calibration artifact")
    if value.get("artifact_hash") != digest({k:v for k,v in value.items() if k != "artifact_hash"}):
        raise ValueError("corrupt alignment calibration artifact")
    if value.get("binding") != binding(plan) or value.get("implementation_hash") != plan["implementation_hash"]:
        raise ValueError("calibration data/teacher/initialization/implementation mismatch")
    if value.get("runtime") != runtime_identity():
        raise ValueError("calibration runtime differs")
    for key in ("lambda_kl", "lambda_mask", "tau_kl", "tau_mask"):
        if not isinstance(value.get(key), (int,float)) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("invalid calibrated coefficient: " + key)
    if not 0 < value.get("selected_top_p", 0) <= 1:
        raise ValueError("calibration requires explicitly selected top-p")
    return {key:value[key] for key in ("lambda_kl", "lambda_mask", "tau_kl", "tau_mask")}
