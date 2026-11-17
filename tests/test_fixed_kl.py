from copy import deepcopy
from pathlib import Path

import pytest
import torch

from alignment_transfer_s2ut.alignment.losses import coefficient_policy, layer_statistics, loss_and_logs, worst_layer
from alignment_transfer_s2ut.alignment.calibration import calibration_implementation_hashes, binding, runtime_identity, validate_artifact
from alignment_transfer_s2ut import colab_runtime
from alignment_transfer_s2ut.io import digest


def policy(method):
    return dict(method=method, kl_lambda=1., mask_top_p=.8, anneal_fraction=.8, anneal_schedule="cosine",
                calibrated=dict(lambda_kl=12.781413823176955, lambda_mask=145.51251933764968,
                                tau_kl=.5538942450051301, tau_mask=.04495423606858799))


@pytest.mark.parametrize("method,steps", [
    ("distillation", [(0, 1.), (20000, 1.), (400000, 1.)]),
    ("annealed_distillation", [(0, 1.), (160000, .5), (320000, 0.), (400000, 0.)]),
    ("mask_distillation", [(0, 145.51251933764968), (20000, 145.51251933764968), (400000, 145.51251933764968)]),
])
def test_real_loss_and_backward_use_condition_coefficient(method, steps):
    prior = torch.tensor([[[[.8, .2]], [[.2, .8]]]])
    settings = policy(method)
    kind = "mask" if method == "mask_distillation" else "kl"
    for update, expected_coefficient in steps:
        logits = torch.tensor([[[[.1, .9]], [[.5, -.5]]]], requires_grad=True)
        student = logits.softmax(-1)
        alignment, logs = loss_and_logs(prior, student, settings, update, 400000)
        statistics = layer_statistics(prior, student, kind, .8)
        unweighted, _ = worst_layer(statistics["losses"], settings["calibrated"]["tau_" + kind], statistics["counts"])
        task = logits.square().sum()
        actual = task + alignment
        expected = task + expected_coefficient * unweighted
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected, logits)[0])
        assert logs["alignment/lambda"] == pytest.approx(expected_coefficient)
        assert logs["alignment/tau"] == settings["calibrated"]["tau_" + kind]
    info = coefficient_policy(settings, 400000)
    assert info["lambda_source"] == ("calibration" if kind == "mask" else "explicit_fixed")
    assert info["calibration_lambda"] == settings["calibrated"]["lambda_" + kind]
    assert info["anneal_end_update"] == (320000 if method == "annealed_distillation" else None)


def test_mask_and_unspecified_kl_preserve_calibrated_loss():
    prior = torch.tensor([[[[.8, .2]]]])
    student = torch.tensor([[[[.4, .6]]]], requires_grad=True)
    for method in ("mask_distillation", "distillation"):
        original = policy(method)
        original.pop("kl_lambda")
        loss, logs = loss_and_logs(prior, student, original, 20000, 400000)
        changed = deepcopy(original)
        changed["kl_lambda"] = 1.
        after, after_logs = loss_and_logs(prior, student, changed, 20000, 400000)
        if method == "mask_distillation":
            torch.testing.assert_close(loss, after, rtol=0, atol=0)
            assert logs["alignment/lambda"] == after_logs["alignment/lambda"]
        else:
            torch.testing.assert_close(loss, after * original["calibrated"]["lambda_kl"])


@pytest.mark.parametrize("value", [True, -1., float("nan"), float("inf")])
def test_invalid_explicit_coefficient_is_rejected(value):
    settings = policy("distillation")
    settings["kl_lambda"] = value
    with pytest.raises(ValueError, match="kl_lambda"):
        coefficient_policy(settings, 400000)


def test_previous_v3_calibration_requires_reviewed_source_and_same_binding(monkeypatch):
    package = Path(colab_runtime.__file__).parent
    sources = colab_runtime.s2ut_implementation_files(package)
    plan = dict(implementation_hash=digest({str(Path(k)): v for k, v in sources.items()}),
                baseline_record_hash="baseline", scientific_config={"seed": 1}, prepared="prepared",
                initialization="fresh", transfer={"artifacts": {"teacher": "verified"}, "all_layers": True},
                config={"common": {"fp16": True}})
    accepted = calibration_implementation_hashes(plan)
    previous = next(value for value in accepted if value != plan["implementation_hash"])
    artifact = dict(format="alignment-calibration-v3", tau_calibration_unit="microbatch", runtime=runtime_identity(),
                    implementation_hash=previous, binding=binding(plan), **policy("distillation")["calibrated"],
                    selected_top_p=.8, number_of_batches=16,
                    gradient_precision=dict(loss_scale=1024, gradient_dtype="float32",
                                            qualified_batches=16, precision_artifact_hash="a" * 64))
    artifact["artifact_hash"] = digest(artifact)
    snapshot = deepcopy(artifact)
    assert validate_artifact(artifact, plan)["lambda_kl"] == artifact["lambda_kl"]
    assert artifact == snapshot
    changed = deepcopy(plan)
    changed["transfer"]["artifacts"]["teacher"] = "different"
    with pytest.raises(ValueError, match="implementation mismatch"):
        validate_artifact(artifact, changed)
    sources["alignment/projection.py"] = "changed projection"
    monkeypatch.setattr(colab_runtime, "s2ut_implementation_files", lambda _: sources)
    with pytest.raises(ValueError, match="implementation mismatch"):
        validate_artifact(artifact, plan)


def test_baseline_storage_adapter_identity_survives_coefficient_change():
    package = Path(colab_runtime.__file__).parent
    sources = colab_runtime.s2ut_implementation_files(package)
    assert sources["corpus_runtime.py"] in {
        "c6e6f85c6ba617d594e55be4431705e926f6342f87d9cc01a2fafca475209e35",
        "d0a4dce83df76acde966af685d1090c7dd95fcc4a897027239c3bfec31ccc187",
    }
