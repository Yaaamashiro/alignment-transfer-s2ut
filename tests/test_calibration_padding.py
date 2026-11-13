import json
from types import SimpleNamespace

import pytest
import torch

from alignment_transfer_s2ut.alignment.calibration import calibrate_lambda, calibrate_tau, gradient_diagnostics, gradient_norm
from alignment_transfer_s2ut.alignment.losses import layer_statistics, top_p_mask
from alignment_transfer_s2ut import alignment_calibration


@pytest.mark.parametrize("polluted", [False, True], ids=["zero-padding", "nonzero-padding"])
def test_padding_effect_on_calibration_objectives(polluted, monkeypatch):
    lengths = torch.tensor([3, 4])
    padding = torch.arange(4)[None, :] >= lengths[:, None]
    logits = torch.tensor([[[[.1, .6, .3, -.1]], [[1.7, .3, -.2, .5]]],
                           [[[.2, -.3, 1.1, .9]], [[-.2, .8, -.5, 1.4]]]], requires_grad=True)
    teacher = torch.tensor([[[[.84, .13, .03, 0.]], [[.15, .73, .12, 0.]]],
                            [[[.06, .82, .08, .04]], [[.58, .20, .14, .08]]]])
    if polluted:
        teacher[0, :, :, -1] = torch.tensor([[5.], [.25]])
    subsample = SimpleNamespace(get_out_seq_lens_tensor=lambda sizes: sizes)
    model = SimpleNamespace(encoder=SimpleNamespace(subsample=subsample))
    sample = {"alignment_prior": teacher, "net_input": {"src_lengths": lengths}, "ntokens": 2}

    def components(model, sample):
        student = logits.masked_fill(padding[:, None, None], -torch.inf).softmax(-1)
        return (logits - .4).square().sum(), 2, {}, student

    criterion = SimpleNamespace(sentence_avg=False, components=components)
    results = []
    for masked in (False, True):
        _, _, _, student = components(model, sample)
        mask = padding if masked else None
        stats = {kind: layer_statistics(teacher, student, kind, .8, mask) for kind in ("kl", "mask")}
        temperatures = {kind: calibrate_tau([value["losses"].tolist()])["tau"] for kind, value in stats.items()}
        with monkeypatch.context() as patch:
            if not masked:
                patch.setattr(alignment_calibration, "encoder_padding_mask", lambda *_: None)
            gradients, _ = alignment_calibration.logical_gradients(model, criterion, [sample], [logits], temperatures, .8)
        task_norm = gradient_norm(gradients["task"])
        result = {"region": top_p_mask(teacher, .8, mask).int().flatten().tolist(),
                  "region_mass": stats["mask"]["region_mass"].tolist(),
                  "coverage": stats["mask"]["coverage"].tolist(), "task_norm": task_norm}
        for kind in stats:
            norm = gradient_norm(gradients[kind])
            coefficient = calibrate_lambda([(task_norm, norm)])
            diagnostics = gradient_diagnostics(gradients["task"], gradients[kind], task_norm, norm, coefficient)
            result[kind] = {"loss": stats[kind]["losses"].tolist(), "tau": temperatures[kind],
                            "lambda": coefficient, "norm": norm, "cosine": diagnostics["gradient_cosine"]}
        results.append(result)
    unmasked, masked = results
    print(json.dumps({"polluted": polluted, "unmasked": unmasked, "masked": masked}))
    assert masked["task_norm"] == unmasked["task_norm"]
    assert masked["coverage"] != unmasked["coverage"]
    if polluted:
        for field in ("region", "region_mass"):
            assert masked[field] != unmasked[field]
        for kind in ("kl", "mask"):
            for field in masked[kind]:
                assert masked[kind][field] != pytest.approx(unmasked[kind][field])
    else:
        for field in ("region", "region_mass", "kl", "mask"):
            assert masked[field] == unmasked[field]
