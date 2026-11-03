from pathlib import Path
from copy import deepcopy
import shutil
import pytest
import torch
from test_benchmark_native import native_data
from test_all_layers_native import layered_teacher
from alignment_transfer_s2ut import benchmark as b, alignment_calibration as c
from alignment_transfer_s2ut.io import write_json, read_json, file_hash, digest, ArtifactStore
from alignment_transfer_s2ut.alignment.calibration import calibrate_tau


@pytest.fixture
def calibrated(native_data,monkeypatch):
    # This Windows installation lacks fairseq's compiled sampler. Only the tiny
    # test dataset uses singleton batches, matching native_data's baseline.
    try:
        __import__("fairseq.data.data_utils_fast")
    except ImportError:
        from fairseq.data.fairseq_dataset import FairseqDataset
        monkeypatch.setattr(FairseqDataset,"batch_by_size",lambda self,indices,**kw:[[int(i)] for i in indices])
    root=native_data.root; layered_teacher(root)
    plan=b.make_plan(root/"baseline.json",root/"calibration",manifest=root/"manifest.jsonl",units=root/"units",teacher=root/"teacher",for_calibration=True)
    def forbidden(*a,**kw):raise AssertionError("calibration called optimizer.step")
    with monkeypatch.context() as protected:
        protected.setattr(torch.optim.Adam,"step",forbidden)
        c.diagnostics(plan,root/"analysis/top-p.json",batches=2)
        result=c.calibrate(plan,root/"analysis/alignment-calibration.json",root/"analysis/top-p.json",.8,batches=2)
    return root,result


@pytest.mark.parametrize("native_data",[2],indirect=True)
def test_native_calibration_is_reproducible_and_binds_plan(calibrated):
    root,result=calibrated
    plan=b.make_plan(root/"baseline.json",root/"calibration",manifest=root/"manifest.jsonl",units=root/"units",teacher=root/"teacher",for_calibration=True)
    repeated=c.calibrate(plan,root/"analysis/again.json",root/"analysis/top-p.json",.8,batches=2)
    assert repeated==result and result["optimizer_updates"]==0
    assert result["tau_calibration_unit"]=="microbatch" and result["number_of_batches"]==2
    assert result["number_of_microbatches"]==len(result["microbatches"])==4
    assert len(result["per_batch"])==2  # Lambda remains a logical-batch gradient calibration.
    for kind in ("kl","mask"):
        assert len(result["per_layer_losses"][kind])==4
        expected=calibrate_tau(result["per_layer_losses"][kind],valid_counts=result["per_layer_valid_rows"][kind])
        assert result["tau_"+kind]==expected["tau"]
    ordinary=b.make_plan(root/"baseline.json",root/"ordinary",manifest=root/"manifest.jsonl",units=root/"units",teacher=root/"teacher",calibration=root/"analysis/again.json")
    annealed=b.make_plan(root/"baseline.json",root/"annealed",method="annealed_distillation",manifest=root/"manifest.jsonl",units=root/"units",teacher=root/"teacher",calibration=root/"analysis/again.json")
    assert ordinary["transfer"]["calibrated"]==annealed["transfer"]["calibrated"]
    assert file_hash(b.read_record(root/"baseline.json")["checkpoint"]) == read_json(root/"baseline.json")["checkpoint_hash"]
    with pytest.raises(ValueError,match="requires --calibration"):
        b.make_plan(root/"baseline.json",root/"missing",manifest=root/"manifest.jsonl",units=root/"units",teacher=root/"teacher")


@pytest.mark.parametrize("native_data",[2],indirect=True)
def test_yaml_fraction_reaches_plan_and_changes_resume_contract(calibrated,monkeypatch):
    import yaml
    from alignment_transfer_s2ut.experiment_conditions import load_conditions
    root,_=calibrated
    arguments=dict(method="annealed_distillation",manifest=root/"manifest.jsonl",units=root/"units",
                   teacher=root/"teacher",calibration=root/"analysis/alignment-calibration.json")
    original=b.make_plan(root/"baseline.json",root/"fraction-run",**arguments)
    write_json(root/"fraction-run/native-run.json",original)
    conditions=load_conditions();conditions["annealed_distillation"]["anneal_fraction"]=.7
    path=root/"conditions.yaml";path.write_text(yaml.safe_dump(conditions),encoding="utf-8")
    monkeypatch.setattr(b,"load_conditions",lambda:load_conditions(path))
    changed=b.make_plan(root/"baseline.json",root/"fraction-run",**arguments)
    assert changed["transfer"]["anneal_fraction"]==.7
    assert changed["transfer"]["anneal_schedule"]=="cosine"
    assert changed["transfer"]["calibrated"]==original["transfer"]["calibrated"]
    assert changed["contract_hash"]!=original["contract_hash"]
    with pytest.raises(ValueError,match="contract exists"):
        b.train_native(changed,root/"fraction-run",resume=True)


@pytest.mark.parametrize("native_data",[2],indirect=True)
@pytest.mark.parametrize("method",["baseline","distillation","annealed_distillation","mask_distillation"])
def test_native_four_methods_update_resume_and_teacher_free(calibrated,native_data,method):
    from fairseq import tasks,utils
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root,result=calibrated
    plan=b.make_plan(root/"baseline.json",root/"run",method,manifest=root/"manifest.jsonl",units=root/"units",teacher=root/"teacher",calibration=root/"analysis/alignment-calibration.json")
    write_json(root/"run/native-run.json",plan)
    cfg=b.native_config(plan["config"]); utils.set_torch_seed(cfg.common.seed)
    task=tasks.setup_task(cfg.task); model=task.build_model(cfg.model); criterion=task.build_criterion(cfg.criterion)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in native_data.initial.items())
    task.load_dataset("train");ds=task.dataset("train");samples=[ds.collater([ds[i]]) for i in range(2)]
    with trainer_hooks(plan["contract_hash"]):
        trainer=train.Trainer(cfg,task,model,criterion)
        trainer.train_step(samples)
        assert trainer.get_num_updates()==1
        file=root/"run/checkpoints/checkpoint_last.pt"
        trainer.consolidate_optimizer();trainer.save_checkpoint(str(file),{})
        resumed=train.Trainer(cfg,task,task.build_model(cfg.model),task.build_criterion(cfg.criterion))
        resumed.load_checkpoint(str(file));assert resumed.get_num_updates()==1
        resumed.train_step(samples);assert resumed.get_num_updates()==2
        for name in ("checkpoint_last.pt","checkpoint_best.pt"):
            resumed.consolidate_optimizer();resumed.save_checkpoint(str(file.parent/name),{})
        registered=b.register_transfer(root/"run",file.parent/"checkpoint_best.pt",root/"registered.json")
        from alignment_transfer_s2ut import alignment_report
        diagnostic_plan=b.make_plan(root/"baseline.json",root/"diagnostic",manifest=root/"manifest.jsonl",
            units=root/"units",teacher=root/"teacher",calibration=root/"analysis/alignment-calibration.json")
        diagnostic=alignment_report._diagnose(diagnostic_plan,registered,root/"analysis/alignment-calibration.json",root/"diagnostics/result.json",1)
        assert len(diagnostic["per_layer_kl"])==2 and not diagnostic["generation_uses_teacher"]
        if method!="baseline":
            changed=deepcopy(result);changed["alpha"]*=.5;changed["lambda_kl"]*=.5;changed["lambda_mask"]*=.5
            changed["artifact_hash"]=digest({k:v for k,v in changed.items() if k!="artifact_hash"})
            write_json(root/"changed-calibration.json",changed)
            altered=b.make_plan(root/"baseline.json",root/"run",method,manifest=root/"manifest.jsonl",units=root/"units",
                teacher=root/"teacher",calibration=root/"changed-calibration.json")
            with pytest.raises(ValueError,match="contract exists"):b.train_native(altered,root/"run",resume=True)
        if method=="mask_distillation":
            from alignment_transfer_s2ut.smoke_pipeline import invoke
            invoke("visualize_attention","--manifest",root/"manifest.jsonl","--checkpoint",file.parent/"checkpoint_best.pt",
                "--teacher",root/"teacher","--units",root/"units","--pair-id","2","--output",root/"visual")
            metadata,arrays=ArtifactStore(root/"visual").get("2")
            assert metadata["beta"]==0 and "teacher_region_mask" in arrays and "guided_s2ut" not in arrays
        task.load_dataset("dev");dev=task.dataset("dev");sample=dev.collater([dev[0]])
        assert "alignment_prior" not in sample
        shutil.rmtree(root/"teacher")
        resumed.valid_step(sample)
        resumed.model.eval()
        with torch.no_grad(): output=resumed.model(**sample["net_input"])[0]
        assert torch.isfinite(output).all()
    assert (root/"run/alignment-train.jsonl").is_file()
    from alignment_transfer_s2ut import alignment_report
    assert Path(alignment_report.curves([root/"run"],root/"curves")["curves"][0]).is_file()


@pytest.mark.parametrize("native_data",[2],indirect=True)
def test_priors_change_only_loss_and_annealed_tail_matches_task(calibrated):
    from fairseq import tasks,utils
    from alignment_transfer_s2ut.alignment.losses import layer_statistics,worst_layer
    root,_=calibrated
    plan=b.make_plan(root/"baseline.json",root/"capture",method="annealed_distillation",manifest=root/"manifest.jsonl",
        units=root/"units",teacher=root/"teacher",calibration=root/"analysis/alignment-calibration.json")
    write_json(root/"capture/native-run.json",plan)
    cfg=b.native_config(plan["config"]);utils.set_torch_seed(cfg.common.seed)
    task=tasks.setup_task(cfg.task);model=task.build_model(cfg.model);criterion=task.build_criterion(cfg.criterion)
    for name in task.multitask_tasks:criterion.set_multitask_loss_weight(name,task.multitask_tasks[name].args.get_loss_weight(0))
    task.load_dataset("train");ds=task.dataset("train");sample=ds.collater([ds[0]])
    logits=[]
    model.register_forward_hook(lambda m,a,output:logits.append(output[0].detach().clone()))
    main,size,_,raw=criterion.components(model,sample)
    losses=layer_statistics(sample["alignment_prior"],raw)["losses"]
    expected=worst_layer(losses,plan["transfer"]["calibrated"]["tau_kl"])[0]
    aligned,_,logs=criterion(model,sample)
    torch.testing.assert_close(aligned-main,size*expected*plan["transfer"]["calibrated"]["lambda_kl"])
    sample["alignment_prior"]=sample["alignment_prior"].flip(-1)
    criterion(model,sample)
    torch.testing.assert_close(logits[0],logits[-1],atol=0,rtol=0)
    model.set_num_updates(2)
    tail,_,logs=criterion(model,sample)
    assert logs["alignment/lambda"]==0
    torch.testing.assert_close(tail,main,atol=0,rtol=0)

__all__ = ['native_data']  # pytest fixture exports
