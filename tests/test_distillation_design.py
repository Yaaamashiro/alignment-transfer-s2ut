import math
import pytest
import torch
from alignment_transfer_s2ut.alignment.losses import top_p_mask, layer_statistics, worst_layer, lambda_at, loss_and_logs
from alignment_transfer_s2ut.alignment.calibration import calibrate_tau, calibrate_lambda, validate_artifact


def test_kl_detach_padding_invalid_and_layers():
    teacher=torch.tensor([[[[.2,.8,0.],[0.,0.,0.]],[[1.,0.,0.],[.5,.5,0.]]]],requires_grad=True)
    student=teacher.detach().clone().requires_grad_()
    result=layer_statistics(teacher,student)
    torch.testing.assert_close(result["losses"],torch.zeros(2))
    assert result["counts"].tolist()==[1,2]
    result["losses"].sum().backward()
    assert teacher.grad is None and student.grad is not None
    assert not student.grad[0,0,1].any()


def test_top_p_minimal_ties_padding_and_nested_sets():
    teacher=torch.tensor([[[[.4,.3,.2,.1,100.],[1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]]])
    padding=torch.tensor([[False,False,False,False,True]])
    masks=[top_p_mask(teacher,p,padding) for p in (.8,.9,.95)]
    assert [int(mask[0,0,0].sum()) for mask in masks]==[3,3,4]
    assert all((~a|b).all() for a,b in zip(masks,masks[1:]))
    assert all(not mask[...,4].any() and mask[0,0,1].sum()==1 and not mask[0,0,2].any() for mask in masks)
    tie=top_p_mask(torch.ones(1,2,4),.5)
    assert tie[0,0].tolist()==[True,True,False,False]


@pytest.mark.parametrize("mass,expected",[(1.,0.),(.7,.3)])
def test_mask_region_mass(mass,expected):
    prior=torch.tensor([[[[1.,0.]]]])
    student=torch.tensor([[[[mass,1-mass]]]],requires_grad=True)
    stats=layer_statistics(prior,student,"mask",.9)
    assert float(stats["losses"][0])==pytest.approx(expected)
    stats["losses"].sum().backward()
    assert student.grad[0,0,0,0] == -1


def test_stable_worst_layer_and_gradient_weights():
    losses=torch.tensor([10000.,10001.,10002.],requires_grad=True)
    value,weights=worst_layer(losses,.4)
    value.backward()
    assert torch.isfinite(value) and float(weights.sum().detach())==pytest.approx(1.)
    torch.testing.assert_close(losses.grad,weights)
    assert weights[0]<weights[1]<weights[2]
    torch.testing.assert_close(worst_layer(torch.ones(6),.01)[1],torch.ones(6)/6)
    assert worst_layer(torch.tensor([1.,3.]),.01)[0]>worst_layer(torch.tensor([1.,3.]),100.)[0]


def test_cosine_schedule():
    assert lambda_at(0,"annealed_distillation",2.,100,.8)==2.
    assert lambda_at(40,"annealed_distillation",2.,100,.8)==pytest.approx(1.)
    assert lambda_at(80,"annealed_distillation",2.,100,.8)==0.
    assert lambda_at(99,"annealed_distillation",2.,100,.8)==0.
    assert lambda_at(0,"mask_distillation",2.,100,.8,training=False)==0.


def test_calibration_estimators_and_failures():
    result=calibrate_tau([[1.,2.,4.],[1.,2.,4.],[1.,2.,1000.]])
    assert result["tau"]==pytest.approx(2/math.log(3))
    assert calibrate_lambda([(2.,1.),(4.,2.),(100.,1.)])==1.
    with pytest.raises(ValueError,match="calibration failure"):calibrate_tau([[1.,1.],[float("nan"),2.]])
    for pair in ((0.,1.),(1.,0.),(1.,float("inf"))):
        with pytest.raises(ValueError,match="gradient calibration failure"):calibrate_lambda([pair])


def test_microbatch_lse_is_not_global_lse():
    left=torch.tensor([.1,2.],requires_grad=True); right=torch.tensor([3.,.1],requires_grad=True)
    micro=(worst_layer(left,.5)[0]+worst_layer(right,.5)[0])/2
    global_value=worst_layer((left+right)/2,.5)[0]
    assert float(micro.detach())!=pytest.approx(float(global_value.detach()))


def test_tau_candidates_use_microbatches_without_cancelling_layer_gaps():
    losses=torch.tensor([[1.,4.,2.],[4.,1.,2.]],dtype=torch.float64)
    result=calibrate_tau(losses)
    assert result["tau"]==pytest.approx(2/math.log(3))
    # Logical averaging loses the gap even though both training microbatches
    # have an equally strong worst layer, at different layer indices.
    with pytest.raises(ValueError,match="calibration failure"):
        calibrate_tau([losses.mean(0)])
    for microbatch in losses:
        weights=torch.softmax(microbatch/result["tau"],0)
        assert float(weights.max()/weights.median())==pytest.approx(3.)


def test_tau_ignores_unsupervised_layers_and_rejects_old_artifacts():
    result=calibrate_tau([[0.,0.,1.,4.]],valid_counts=[[0,0,3,3]])
    assert result["tau"]==pytest.approx(3/math.log(3))
    with pytest.raises(ValueError,match="microbatch tau calibration v2"):
        validate_artifact({"format":"alignment-calibration-v1"},{})


def test_tau_records_rejected_microbatch_indices_with_the_new_key():
    result=calibrate_tau([[1.,2.,4.],[1.,1.,1.],[float("nan"),1.,2.],
                          [float("inf"),1.,2.],[0.,0.,0.]],
                         valid_counts=[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[0,0,0]])
    assert result["tau"]==pytest.approx(2/math.log(3))
    assert result["rejected_microbatches"]==[1,2,3,4]
    assert "rejected_batches" not in result


def policy(**overrides):
    return dict(method="annealed_distillation",calibrated=dict(lambda_kl=2.,tau_kl=.4),
                anneal_fraction=.7,anneal_schedule="cosine",**overrides)


def test_policy_fraction_controls_both_loss_and_task_only_logs():
    teacher=torch.tensor([[[[1.,0.]]]])
    student=torch.tensor([[[[.5,.5]]]],requires_grad=True)
    _,middle=loss_and_logs(teacher,student,policy(),35,100)
    assert middle["alignment/lambda"]==pytest.approx(1.)
    assert middle["alignment/annealing_progress"]==.5
    before,before_logs=loss_and_logs(teacher,student,policy(),69,100)
    tail,tail_logs=loss_and_logs(teacher,student,policy(),70,100)
    assert float(before.detach())>0 and float(tail.detach())==0.
    assert before_logs["alignment/task_only_start"]==-1
    assert tail_logs["alignment/task_only_start"]==70 and tail_logs["alignment/task_only_phase"]==1
    assert not any("teacher_free" in key for key in tail_logs)


@pytest.mark.parametrize("key,value",[("anneal_fraction",0.),("anneal_fraction",float("nan")),
    ("anneal_fraction",1.1),("anneal_fraction",True),("anneal_schedule","linear")])
def test_invalid_schedule_is_rejected(key,value):
    settings=policy();settings[key]=value
    with pytest.raises(ValueError,match=key):
        loss_and_logs(torch.ones(1,1,1,1),torch.ones(1,1,1,1),settings,0,100)


def test_diagnostic_means_exclude_invalid_layers():
    teacher=torch.tensor([[[[0.,0.]],[[1.,0.]]]])
    student=torch.full_like(teacher,.5)
    _,logs=loss_and_logs(teacher,student,policy(),0,100)
    assert float(logs["alignment/kl_mean"])==pytest.approx(math.log(2))
    assert float(logs["alignment/kl_worst"])==pytest.approx(math.log(2))
    assert float(logs["alignment/student_entropy"])==pytest.approx(math.log(2))
    assert int(logs["alignment/valid_rows_0"])==0
