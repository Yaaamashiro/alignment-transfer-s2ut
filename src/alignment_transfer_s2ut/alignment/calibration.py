"""Pure calibration estimators; failures are explicit, without fallback constants."""
import math
from pathlib import Path
import torch
from ..io import digest, read_json


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


def gradient_diagnostics(task, alignment, task_norm, alignment_norm, coefficient):
    if not all(math.isfinite(value) and value >= 0 for value in (task_norm, alignment_norm, coefficient)):
        raise ValueError("nonfinite or negative gradient norm/coefficient")
    reason = None
    cosine = None
    if task_norm == 0 or alignment_norm == 0:
        reason = "zero task gradient" if task_norm == 0 else "zero alignment gradient"
    else:
        products = [(left.double() * right.double()).sum() for left, right in zip(task, alignment, strict=True)
                    if left is not None and right is not None]
        cosine = float(torch.stack(products).sum()) / task_norm / alignment_norm if products else 0.
        cosine = min(1., max(-1., cosine))
    ratio = coefficient * alignment_norm / task_norm if task_norm else None
    if ratio is not None and not math.isfinite(ratio):
        raise ValueError("nonfinite alignment gradient ratio")
    return {"gradient_ratio": ratio, "gradient_cosine": cosine, "cosine_undefined_reason": reason,
            "ratio_undefined_reason": "zero task gradient" if not task_norm else None}


def distribution_summary(values):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if not tensor.numel():
        return None
    percentiles = tensor.quantile(tensor.new_tensor([.25, .5, .75, .9])).tolist()
    return {"mean": float(tensor.mean()), "median": percentiles[1],
            "min": float(tensor.min()), "max": float(tensor.max()),
            "p25": percentiles[0], "p75": percentiles[2], "p90": percentiles[3]}


def layer_weight_summary(weights, losses, counts):
    values = torch.as_tensor(weights, dtype=torch.float64)
    losses, counts = torch.as_tensor(losses), torch.as_tensor(counts)
    valid = counts > 0
    used = valid.any(dim=1)
    if not used.any():
        return None
    values, losses, valid = values[used], losses[used], valid[used]
    entropy = -(values * values.clamp_min(1e-300).log()).sum(dim=1)
    worst = values.argmax(dim=1)
    # Use the same lower-order median and deterministic first tie as calibration/logging.
    ratios = []
    for row, layer_losses, eligible in zip(values, losses, valid, strict=True):
        selected = row[eligible][layer_losses[eligible].argsort(stable=True)]
        middle = float(selected[(len(selected) - 1) // 2])
        if middle > 0:
            ratios.append(float(selected[-1]) / middle)
    return {"mean_weights": values.mean(dim=0).tolist(), "max_weight": float(values.max()),
            "worst_layer_frequency": (torch.bincount(worst, minlength=values.size(1)) / len(worst)).tolist(),
            "entropy": distribution_summary(entropy.tolist()),
            "effective_layers": distribution_summary(entropy.exp().tolist()),
            "max_to_median_weight_ratio": distribution_summary(ratios),
            "undefined_weight_ratios": len(worst) - len(ratios)}


def binding(plan):
    return {key: plan[key] for key in ("baseline_record_hash", "scientific_config", "prepared", "initialization")} | {
        "artifacts": plan["transfer"]["artifacts"], "all_layers": plan["transfer"]["all_layers"]}


def calibration_implementation_hashes(plan):
    """Accept only the reviewed coefficient-policy transition for existing calibration."""
    from ..colab_runtime import s2ut_implementation_files

    package = Path(__file__).resolve().parents[1]
    compatibility = read_json(package / "coefficient-compatibility.json")
    accepted = {plan["implementation_hash"]}
    sources = s2ut_implementation_files(package)
    if any(all(sources.get(name) == expected for name, expected in reviewed.items())
           for reviewed in (compatibility["current"], compatibility.get("current_crlf", compatibility["current"]))):
        sources.update(compatibility["previous"])
        accepted.add(digest({str(Path(name)): value for name, value in sources.items()}))
    return accepted


def validate_artifact(value, plan):
    if value.get("format") not in {"alignment-calibration-v2", "alignment-calibration-v3"} or value.get("tau_calibration_unit") != "microbatch":
        raise ValueError("microbatch tau calibration v2 or v3 required; regenerate the calibration artifact")
    if value.get("artifact_hash") != digest({k:v for k,v in value.items() if k != "artifact_hash"}):
        raise ValueError("corrupt alignment calibration artifact")
    if value.get("binding") != binding(plan) or value.get("implementation_hash") not in calibration_implementation_hashes(plan):
        raise ValueError("calibration data/teacher/initialization/implementation mismatch")
    if value.get("runtime") != runtime_identity():
        raise ValueError("calibration runtime differs")
    if plan["config"]["common"]["fp16"]:
        from .precision import SCALES

        precision = value.get("gradient_precision") or {}
        receipt = precision.get("precision_artifact_hash", "")
        if (value["format"] != "alignment-calibration-v3" or precision.get("loss_scale") not in SCALES
                or precision.get("gradient_dtype") != "float32"
                or not isinstance(precision.get("qualified_batches"), int)
                or not 0 < precision["qualified_batches"] == value.get("number_of_batches")
                or not isinstance(receipt, str) or len(receipt) != 64
                or any(character not in "0123456789abcdef" for character in receipt)):
            raise ValueError("FP16 transfer requires a precision-qualified calibration v3")
    for key in ("lambda_kl", "lambda_mask", "tau_kl", "tau_mask"):
        if not isinstance(value.get(key), (int,float)) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("invalid calibrated coefficient: " + key)
    if not 0 < value.get("selected_top_p", 0) <= 1:
        raise ValueError("calibration requires explicitly selected top-p")
    return {key:value[key] for key in ("lambda_kl", "lambda_mask", "tau_kl", "tau_mask")}
