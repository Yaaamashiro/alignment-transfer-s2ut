from pathlib import Path
from copy import deepcopy
import json
import shutil
import pytest
import torch
from test_benchmark_native import native_data
from test_all_layers_native import layered_teacher
from alignment_transfer_s2ut import benchmark as b, alignment_calibration as c
from alignment_transfer_s2ut.io import write_json, read_json, read_jsonl, file_hash, digest, ArtifactStore
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
    batches = 16 if plan["config"]["common"]["fp16"] else 2
    def forbidden(*a,**kw):raise AssertionError("calibration called optimizer.step")
    with monkeypatch.context() as protected:
        protected.setattr(torch.optim.Adam,"step",forbidden)
        c.diagnostics(plan,root/"analysis/top-p.json",batches=batches)
        result=c.calibrate(plan,root/"analysis/alignment-calibration.json",root/"analysis/top-p.json",.8,batches=batches)
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
        saved=read_json(root/"analysis/alignment-calibration.json")["tau_estimates"][kind]
        assert saved["rejected_microbatches"]==expected["rejected_microbatches"]
        assert "rejected_batches" not in saved
    ordinary=b.make_plan(root/"baseline.json",root/"ordinary",manifest=root/"manifest.jsonl",units=root/"units",teacher=root/"teacher",calibration=root/"analysis/again.json")
    previous = {**result, "implementation_hash": "0" * 64}
    previous["artifact_hash"] = digest({key: value for key, value in previous.items() if key != "artifact_hash"})
    write_json(root / "analysis/previous.json", previous)
    with pytest.raises(ValueError, match="implementation mismatch"):
        b.make_plan(root/"baseline.json",root/"ordinary",manifest=root/"manifest.jsonl",units=root/"units",
                    teacher=root/"teacher",calibration=root/"analysis/previous.json")
    annealed=b.make_plan(root/"baseline.json",root/"annealed",method="annealed_distillation",manifest=root/"manifest.jsonl",units=root/"units",teacher=root/"teacher",calibration=root/"analysis/again.json")
    assert ordinary["transfer"]["calibrated"]==annealed["transfer"]["calibrated"]
    assert file_hash(b.read_record(root/"baseline.json")["checkpoint"]) == read_json(root/"baseline.json")["checkpoint_hash"]
    with pytest.raises(ValueError,match="requires --calibration"):
        b.make_plan(root/"baseline.json",root/"missing",manifest=root/"manifest.jsonl",units=root/"units",teacher=root/"teacher")


@pytest.mark.parametrize("native_data", [2], indirect=True)
def test_fixed_kl_native_loss_checkpoint_and_contract(calibrated, native_data, capsys):
    from fairseq import tasks, utils
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    from alignment_transfer_s2ut.alignment.losses import layer_statistics, worst_layer

    root, result = calibrated
    calibration = root / "analysis/alignment-calibration.json"
    before = file_hash(calibration)
    kwargs = dict(manifest=root / "manifest.jsonl", units=root / "units", teacher=root / "teacher",
                  calibration=calibration)
    old = b.make_plan(root / "baseline.json", root / "old-pilot", **kwargs)
    new = b.make_plan(root / "baseline.json", root / "lambda1", kl_lambda=1., **kwargs)
    write_json(root / "old-pilot/native-run.json", old)
    mixed = b.make_plan(root / "baseline.json", root / "old-pilot", kl_lambda=1., **kwargs)
    assert mixed["contract_hash"] != old["contract_hash"]
    with pytest.raises(ValueError, match="output/contract exists"):
        b.train_native(mixed, root / "old-pilot", resume=True)
    assert new["scientific_config"] == old["scientific_config"]
    assert new["transfer"]["calibrated"] == old["transfer"]["calibrated"]
    write_json(root / "lambda1/native-run.json", new)
    cfg = b.native_config(new["config"])
    utils.set_torch_seed(cfg.common.seed)
    task = tasks.setup_task(cfg.task)
    model, criterion = task.build_model(cfg.model), task.build_criterion(cfg.criterion)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in native_data.initial.items())
    for name, auxiliary in task.multitask_tasks.items():
        criterion.set_multitask_loss_weight(name, auxiliary.args.get_loss_weight(0))
    task.load_dataset("train")
    ds = task.dataset("train")
    sample = ds.collater([ds[0]])
    main, size, _, raw = criterion.components(model, sample)
    unweighted = worst_layer(layer_statistics(sample["alignment_prior"], raw)["losses"], result["tau_kl"])[0]
    total, _, logs = criterion(model, sample)
    torch.testing.assert_close(total, main + size * unweighted)
    assert logs["alignment/lambda"] == 1.
    from fairseq_cli import train
    with trainer_hooks(new["contract_hash"]):
        trainer = train.Trainer(cfg, task, model, criterion)
        trainer.train_step([sample])
        checkpoint = root / "lambda1/checkpoints/checkpoint_last.pt"
        trainer.consolidate_optimizer()
        trainer.save_checkpoint(str(checkpoint), {})
        saved = b.load_native(checkpoint)
        assert saved["extra_state"]["alignment_coefficient_policy"]["lambda_source"] == "explicit_fixed"
        assert saved["extra_state"]["alignment_coefficient_policy"]["initial_lambda"] == 1.
        assert saved["extra_state"]["alignment_coefficient_policy"]["calibration_lambda"] == result["lambda_kl"]
        row = read_jsonl(root / "lambda1/alignment-train.jsonl")[-1]
        assert row["alignment/lambda"] == 1. and row["coefficient_policy"]["lambda_source"] == "explicit_fixed"
    moved = root / "copied-old-pilot"
    copied = b.make_plan(root / "baseline.json", moved, kl_lambda=1., **kwargs)
    write_json(moved / "native-run.json", copied)
    saved["extra_state"]["alignment_contract_hash"] = old["contract_hash"]
    (moved / "checkpoints").mkdir()
    torch.save(saved, moved / "checkpoints/checkpoint_last.pt")
    with pytest.raises(ValueError, match="different training contract"):
        b.train_native(copied, moved, resume=True)
    assert file_hash(calibration) == before
    mask_before = b.make_plan(root / "baseline.json", root / "mask", method="mask_distillation", **kwargs)
    mask_after = b.make_plan(root / "baseline.json", root / "mask", method="mask_distillation", kl_lambda=1., **kwargs)
    assert mask_before == mask_after
    capsys.readouterr()
    b.main(["plan", "--baseline", str(root / "baseline.json"), "--output", str(root / "cli-lambda1"),
            "--manifest", str(root / "manifest.jsonl"), "--units", str(root / "units"),
            "--teacher", str(root / "teacher"), "--calibration", str(calibration),
            "--method", "annealed_distillation", "--kl-lambda", "1.0"])
    cli_plan = json.loads(capsys.readouterr().out)
    assert cli_plan["transfer"]["coefficient_policy"]["initial_lambda"] == 1.


@pytest.mark.parametrize("native_data", [2], indirect=True)
@pytest.mark.parametrize("method, flag, expected", [
    ("mask_distillation", "mask_lambda", 10.), ("annealed_distillation", "kl_lambda", 1.)])
def test_explicit_condition_native_state_and_checkpoint(calibrated, native_data, capsys, monkeypatch, method, flag, expected):
    from fairseq import tasks, utils
    from fairseq_cli import train
    from alignment_transfer_s2ut.alignment.losses import layer_statistics, worst_layer
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks

    root, result = calibrated
    source = root / "analysis/alignment-calibration.json"
    checksum = file_hash(source)
    shared = dict(method=method, manifest=root / "manifest.jsonl", units=root / "units",
                  teacher=root / "teacher", calibration=source)
    original = b.make_plan(root / "baseline.json", root / "old-mask", **shared)
    write_json(root / "old-mask/native-run.json", original)
    changed = b.make_plan(root / "baseline.json", root / "old-mask", **shared, **{flag: expected})
    assert changed["contract_hash"] != original["contract_hash"]
    with pytest.raises(ValueError, match="output/contract exists"):
        b.train_native(changed, root / "old-mask", resume=True)
    run = root / "fixed"
    plan = b.make_plan(root / "baseline.json", run, **shared, **{flag: expected})
    assert plan["contract_hash"] != original["contract_hash"]
    assert plan["scientific_config"] == native_data.record["scientific_config"]
    write_json(run / "native-run.json", plan)
    cfg = b.native_config(plan["config"])
    utils.set_torch_seed(cfg.common.seed)
    task = tasks.setup_task(cfg.task)
    model, criterion = task.build_model(cfg.model), task.build_criterion(cfg.criterion)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in native_data.initial.items())
    for name, auxiliary in task.multitask_tasks.items():
        criterion.set_multitask_loss_weight(name, auxiliary.args.get_loss_weight(0))
    task.load_dataset("train")
    ds = task.dataset("train")
    sample = ds.collater([ds[0]])
    main, size, _, raw = criterion.components(model, sample)
    kind = "mask" if method == "mask_distillation" else "kl"
    stats = layer_statistics(sample["alignment_prior"], raw, kind, .8)
    base = worst_layer(stats["losses"], result["tau_" + kind], stats["counts"])[0]
    actual, _, _ = criterion(model, sample)
    torch.testing.assert_close(actual, main + size * expected * base)
    with trainer_hooks(plan["contract_hash"]):
        trainer = train.Trainer(cfg, task, model, criterion)
        trainer.train_step([sample])
        checkpoint = run / "checkpoints/checkpoint_last.pt"
        trainer.consolidate_optimizer()
        trainer.save_checkpoint(str(checkpoint), {})
        saved = b.load_native(checkpoint)
        policy = saved["extra_state"]["alignment_coefficient_policy"]
        assert policy["initial_lambda"] == expected and policy["lambda_source"] == "explicit_fixed"
        assert policy["calibration_lambda"] == result["lambda_" + kind]
        assert read_jsonl(run / "alignment-train.jsonl")[-1]["alignment/lambda"] == expected
    restored = root / "copied-pilot"
    new = b.make_plan(root / "baseline.json", restored, **shared, **{flag: expected})
    write_json(restored / "native-run.json", new)
    (restored / "checkpoints").mkdir()
    saved["extra_state"]["alignment_contract_hash"] = original["contract_hash"]
    torch.save(saved, restored / "checkpoints/checkpoint_last.pt")
    with pytest.raises(ValueError, match="different training contract"):
        b.train_native(new, restored, resume=True)
    if method == "mask_distillation":
        wrong = b.make_plan(root / "baseline.json", run, **shared, mask_lambda=10., kl_lambda=2.)
        assert wrong == plan
    else:
        wrong = b.make_plan(root / "baseline.json", run, **shared, kl_lambda=1., mask_lambda=20.)
        assert wrong == plan
    capsys.readouterr()
    option = "--" + flag.replace("_", "-")
    b.main(["plan", "--baseline", str(root / "baseline.json"), "--output", str(root / "cli-explicit"),
            "--manifest", str(root / "manifest.jsonl"), "--units", str(root / "units"),
            "--teacher", str(root / "teacher"), "--calibration", str(source), "--method", method,
            option, str(expected)])
    assert json.loads(capsys.readouterr().out)["transfer"]["coefficient_policy"]["initial_lambda"] == expected
    from alignment_transfer_s2ut import cli
    parsed = cli.parser_for("train_s2ut").parse_args([
        "--baseline-record", str(root / "baseline.json"), "--output", str(root / "generic-cli"),
        "--manifest", str(root / "manifest.jsonl"), "--units", str(root / "units"),
        "--teacher", str(root / "teacher"), "--calibration", str(source), "--method", method,
        "--kl-lambda", "1.", "--mask-lambda", "10."])
    monkeypatch.setattr(b, "train_native", lambda plan, *args: plan)
    assert cli.run("train_s2ut", parsed)["transfer"]["coefficient_policy"]["initial_lambda"] == expected
    assert file_hash(source) == checksum


@pytest.mark.gpu
@pytest.mark.parametrize("native_data", [(2, "cuda", 2, None)], indirect=True)
def test_fp16_precision_receipt_and_legacy_migration(calibrated):
    root, result = calibrated
    plan = b.make_plan(root / "baseline.json", root / "calibration", manifest=root / "manifest.jsonl",
                       units=root / "units", teacher=root / "teacher", for_calibration=True)
    assert result["format"] == "alignment-calibration-v3"
    assert result["number_of_batches"] == 16
    report = read_json(root / "analysis/alignment-calibration-precision.json")
    scale = report["selected_loss_scale"]
    assert report["candidates"][str(scale)]["qualified"]
    assert result["gradient_precision"]["precision_artifact_hash"] == report["artifact_hash"]
    for batch in report["per_batch"]:
        assert batch["fp32"] == "measured"
        assert not any(metric["nonfinite"] for metric in batch["metrics"][str(scale)].values())
    old = root / "previous-calibration" / "alignment-calibration.json"
    previous = deepcopy(result)
    previous["format"] = "alignment-calibration-v2"
    previous.pop("gradient_precision")
    previous["artifact_hash"] = digest({key: value for key, value in previous.items() if key != "artifact_hash"})
    write_json(old, previous)
    previous_hash = file_hash(old)
    with pytest.raises(ValueError, match="precision-qualified"):
        b.make_plan(root / "baseline.json", root / "transfer", manifest=root / "manifest.jsonl",
                    units=root / "units", teacher=root / "teacher", calibration=old)
    with pytest.raises(ValueError, match="separate path"):
        c.calibrate(plan, old, root / "analysis/top-p.json", .8, 16, previous_calibration=old)
    repeated = c.calibrate(plan, root / "new/alignment-calibration.json", root / "analysis/top-p.json", .8, 16,
                           previous_calibration=old)
    assert repeated["previous_calibration"]["tau_matches"]
    assert file_hash(old) == previous_hash
    assert repeated["per_batch"] == result["per_batch"]


@pytest.mark.parametrize("native_data", [2], indirect=True)
def test_both_calibration_passes_use_native_encoder_padding(calibrated, monkeypatch):
    from fairseq import tasks
    from alignment_transfer_s2ut.s2ut import benchmark_extension as extension
    from alignment_transfer_s2ut.alignment.losses import encoder_padding_mask

    root, _ = calibrated
    plan = b.make_plan(root / "baseline.json", root / "run", manifest=root / "manifest.jsonl",
                       units=root / "units", teacher=root / "teacher", for_calibration=True)
    def padded_batches(trainer, count):
        trainer.task.load_dataset("train")
        dataset = trainer.task.dataset("train")
        for _ in range(count):
            yield [dataset.collater([dataset[0], dataset[1]])]

    monkeypatch.setattr(c, "logical_batches", padded_batches)
    analysis = root / "analysis/padded-top-p.json"
    c.diagnostics(plan, analysis, batches=2)
    seen = []
    original = c.layer_statistics

    def capture(teacher, student, kind, top_p, source_padding=None):
        assert source_padding is not None
        seen.append(source_padding.clone())
        return original(teacher, student, kind, top_p, source_padding)

    monkeypatch.setattr(c, "layer_statistics", capture)
    c.calibrate(plan, root / "analysis/padding.json", analysis, .8, batches=2)
    assert len(seen) == 8
    assert all(torch.equal(seen[0], mask) for mask in seen)

    run = root / "native"
    plan = b.make_plan(root / "baseline.json", run, manifest=root / "manifest.jsonl",
                       units=root / "units", teacher=root / "teacher", calibration=root / "analysis/alignment-calibration.json")
    write_json(run / "native-run.json", plan)
    cfg = b.native_config(plan["config"])
    task = tasks.setup_task(cfg.task)
    task.load_dataset("train")
    dataset = task.dataset("train")
    sample = dataset.collater([dataset[0], dataset[1]])
    model, criterion = task.build_model(cfg.model), task.build_criterion(cfg.criterion)
    for name, auxiliary in task.multitask_tasks.items():
        criterion.set_multitask_loss_weight(name, auxiliary.args.get_loss_weight(0))
    native_masks = []
    native_loss = extension.loss_and_logs

    def capture_native(teacher, student, policy, step, budget, padding):
        native_masks.append(padding)
        return native_loss(teacher, student, policy, step, budget, padding)

    monkeypatch.setattr(extension, "loss_and_logs", capture_native)
    criterion(model, sample)
    mask, = native_masks
    assert mask.any() and (~mask).any()
    torch.testing.assert_close(mask, seen[0])
    torch.testing.assert_close(mask, encoder_padding_mask(model, sample, sample["alignment_prior"]))
    prior = sample["alignment_prior"]
    assert not prior.masked_select(mask[:, None, None, :]).count_nonzero()
    target_lengths = sample["target"].ne(task.target_dictionary.pad()).sum(-1)
    for batch, size in enumerate(target_lengths.tolist()):
        assert not prior[batch, :, size - 1:].count_nonzero()


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
