"""All-layer research path: real pinned fairseq task, criterion and Trainer."""
from copy import deepcopy
from pathlib import Path
import shutil
from argparse import Namespace
import numpy as np
import pytest
import torch
from test_benchmark_native import native_data
from alignment_transfer_s2ut import benchmark as b
from alignment_transfer_s2ut.io import ArtifactStore,write_json,digest,write_jsonl
from alignment_transfer_s2ut.data.manifest import load_manifest
from alignment_transfer_s2ut.guidance.attention import layer_kl_sums,layer_row_counts


def layered_teacher(root):
    store=ArtifactStore(root/"teacher")
    for row in load_manifest(root/"manifest.jsonl",check_audio=False):
        meta,packet=store.get(row["pair_id"])
        t,s=packet["mt_attention"].shape
        matrices=np.zeros((2,t,s),np.float32)
        matrices[0,:max(1,t//2),0]=1
        matrices[1,:,s-1]=1
        duration=float(packet["source_duration"])
        packet["source_spans"]=np.array([[i*duration/s,(i+1)*duration/s] for i in range(s)])
        packet["target_spans"]=np.array([[i*duration/t,(i+1)*duration/t] for i in range(t)])
        packet.update(mt_attention_layers=matrices,mt_attention=matrices[-1])
        spec=dict(all_layers=True,decoder_layers=2,layer_indices=[0,1],heads=None)
        store.put(row["pair_id"],{**meta,**spec,"attention_spec_hash":digest(spec)},packet,overwrite=True)


@pytest.mark.parametrize("native_data",[2],indirect=True)
@pytest.mark.parametrize("method",["baseline","distillation","annealed_guidance","guidance"])
def test_native_all_layers_two_updates_and_teacher_free_inference(native_data,method):
    from fairseq import tasks,utils
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root=native_data.root; layered_teacher(root)
    plan=b.make_plan(root/"baseline.json",root/"run",method,"t2tt",root/"manifest.jsonl",root/"units",root/"teacher",anneal_steps=1,all_layers=method!="baseline")
    assert plan["transfer"]["all_layers"]==(method!="baseline")
    write_json(root/"run/native-run.json",plan)
    cfg=b.native_config(plan["config"]); utils.set_torch_seed(cfg.common.seed)
    task=tasks.setup_task(cfg.task); model=task.build_model(cfg.model); criterion=task.build_criterion(cfg.criterion)
    for name in task.multitask_tasks:
        criterion.set_multitask_loss_weight(name,task.multitask_tasks[name].args.get_loss_weight(0))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in native_data.initial.items())
    task.load_dataset("train"); dataset=task.dataset("train")
    samples=[dataset.collater([dataset[i]]) for i in range(2)]
    if method!="baseline":
        assert samples[0]["alignment_prior"].shape[1]==2
        assert not torch.equal(samples[0]["alignment_prior"][:,0],samples[0]["alignment_prior"][:,1])
        model.train(); model.set_num_updates(0)
        captured=[]
        model.register_forward_hook(lambda *a:captured.append((model.alignment_raw.detach(),model.alignment_guided.detach())))
        loss,size,logs=criterion(model,samples[0])
        assert logs["guidance_beta"]==(1. if method in {"guidance","annealed_guidance"} else 0.)
        if method in {"guidance","annealed_guidance"}:
            raw,guided=captured[-1]
            for l in range(2):
                support=samples[0]["alignment_prior"][:,l]>0
                assert guided[:,l][support].sum()>raw[:,l][support].sum()
        model.set_num_updates(1); _,_,after=criterion(model,samples[0])
        if method=="annealed_guidance": assert after["guidance_beta"]==0
        loss.backward()
        for block in model.decoder.layers:
            assert torch.isfinite(block.encoder_attn.q_proj.weight.grad).all()
            assert block.encoder_attn.q_proj.weight.grad.abs().sum()>0
        model.zero_grad()
    with trainer_hooks(plan["contract_hash"]):
        trainer=train.Trainer(cfg,task,model,criterion)
        for _ in range(2): trainer.train_step(samples)
        assert trainer.get_num_updates()==2
        if method!="baseline":
            rows=sum((layer_row_counts(s["alignment_prior"]) for s in samples))
            torch.testing.assert_close(criterion.alignment_scales,sum(s["ntokens"] for s in samples)/rows.clamp_min(1)/2)
        folder=root/"run/checkpoints"; folder.mkdir()
        for name in ("checkpoint_last.pt","checkpoint_best.pt"):
            trainer.consolidate_optimizer(); trainer.save_checkpoint(str(folder/name),{})
        assert model.alignment_raw is None if method!="baseline" else True
    b.register_transfer(root/"run",folder/"checkpoint_best.pt",root/"registered.json")
    if method!="baseline":
        from alignment_transfer_s2ut.smoke_pipeline import invoke
        output=root/"visualized"
        invoke("visualize_attention","--checkpoint",folder/"checkpoint_best.pt","--manifest",root/"manifest.jsonl",
            "--units",root/"units","--teacher",root/"teacher","--pair-id","2","--layer",0,"--output",output)
        meta,values=ArtifactStore(output).get("2")
        assert meta["all_layers"] and meta["layer"]==0
        np.testing.assert_array_equal(values["t2tt"],ArtifactStore(root/"teacher").get("2")[1]["mt_attention_layers"][0])
    shutil.rmtree(root/"teacher")
    row=load_manifest(root/"manifest.jsonl",split="dev",check_audio=False)[0]
    write_jsonl(root/"source.jsonl",[dict(pair_id=row["pair_id"],ja_audio=row["ja_audio"])])
    from alignment_transfer_s2ut.inference.native import infer_native
    values=infer_native(Namespace(record=root/"registered.json",input=root/"source.jsonl",output=root/"pred.jsonl",
        units_only=True,vocoder_config=None,device="cpu",beam_size=2,max_len_a=0.,max_len_b=8,length_penalty=1.,resume=False,overwrite=False))
    assert values[0]["inference_beta"]==0 and values[0]["units"]


@pytest.mark.parametrize("native_data",[2],indirect=True)
def test_native_scope_and_layer_count_contract_rejections(native_data):
    root=native_data.root
    args=(root/"baseline.json",root/"run","distillation","t2tt",root/"manifest.jsonl",root/"units",root/"teacher")
    with pytest.raises(ValueError,match="all-layer T2TT teacher"): b.make_plan(*args,all_layers=True)
    layered_teacher(root)
    plan=b.make_plan(*args,all_layers=True)
    single=b.make_plan(*args,all_layers=False)
    assert plan["contract_hash"]!=single["contract_hash"]
    write_json(root/"run/native-run.json",plan)
    with pytest.raises(ValueError,match="contract exists"): b.train_native(single,root/"run",resume=True)
    store=ArtifactStore(root/"teacher"); meta,packet=store.get("0")
    packet["mt_attention_layers"]=packet["mt_attention_layers"][:1]
    store.put("0",meta,packet,overwrite=True)
    with pytest.raises(ValueError,match="layer count"): b.make_plan(*args,all_layers=True)


@pytest.mark.parametrize("native_data",[2],indirect=True)
def test_native_criterion_uses_per_layer_global_rows(native_data):
    from fairseq import tasks,utils
    root=native_data.root; layered_teacher(root)
    plan=b.make_plan(root/"baseline.json",root/"run","distillation","t2tt",root/"manifest.jsonl",root/"units",root/"teacher",all_layers=True)
    write_json(root/"run/native-run.json",plan)
    cfg=b.native_config(plan["config"]); utils.set_torch_seed(cfg.common.seed)
    task=tasks.setup_task(cfg.task); model=task.build_model(cfg.model); criterion=task.build_criterion(cfg.criterion)
    for name in task.multitask_tasks:
        criterion.set_multitask_loss_weight(name,task.multitask_tasks[name].args.get_loss_weight(0))
    task.load_dataset("train"); ds=task.dataset("train")
    samples=[ds.collater([ds[i]]) for i in range(2)]
    rows=sum(layer_row_counts(s["alignment_prior"]) for s in samples)
    assert rows[0]!=rows[1]
    tokens=sum(s["ntokens"] for s in samples)
    criterion.alignment_scales=tokens/rows/2
    captured=[]
    model.register_forward_hook(lambda *a:captured.append(model.alignment_raw))
    extra=[]; expected=[]
    for sample in samples:
        task.transfer_policy["lambda_align"]=1.
        aligned,_,logs=criterion(model,sample)
        raw=captured[-1]
        assert raw.shape==sample["alignment_prior"].shape
        expected.append(layer_kl_sums(sample["alignment_prior"],raw)/rows)
        task.transfer_policy["lambda_align"]=0.
        main,_,_=criterion(model,sample)
        extra.append(aligned-main)
    torch.testing.assert_close(sum(extra)/tokens,sum(expected).mean(),atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize("native_data",[2],indirect=True)
@pytest.mark.parametrize("all_layers",[False,True])
def test_two_layer_beta_zero_logits_loss_update_match_stock(native_data,all_layers):
    from fairseq import tasks,utils
    from fairseq.trainer import Trainer as StockTrainer
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root=native_data.root; layered_teacher(root)
    plan=b.make_plan(root/"baseline.json",root/"zero","guidance","t2tt",root/"manifest.jsonl",root/"units",root/"teacher",beta=0.,all_layers=all_layers)
    write_json(root/"zero/native-run.json",plan)
    cfg=b.native_config(plan["config"]); task=tasks.setup_task(cfg.task)
    utils.set_torch_seed(cfg.common.seed); model=task.build_model(cfg.model)
    task.load_dataset("train"); ds=task.dataset("train"); samples=[ds.collater([ds[i]]) for i in range(2)]
    stock_model=native_data.task.build_model(native_data.cfg.model); stock_model.load_state_dict(native_data.initial)
    model.eval(); stock_model.eval()
    with torch.no_grad():
        expected=stock_model(**native_data.samples[0]["net_input"])[0]
        model.transfer_context=(samples[0]["alignment_prior"],0.,.05,all_layers)
        actual=model(**samples[0]["net_input"])[0]; model.transfer_context=None
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    stock=StockTrainer(native_data.cfg,native_data.task,stock_model,native_data.task.build_criterion(native_data.cfg.criterion))
    expected_log=stock.train_step(native_data.samples)
    with trainer_hooks(plan["contract_hash"]):
        trainer=train.Trainer(cfg,task,model,task.build_criterion(cfg.criterion))
        actual_log=trainer.train_step(samples)
    assert actual_log["loss"]==expected_log["loss"]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in stock_model.state_dict().items())
    for block in model.decoder.layers:
        assert block.encoder_attn.raw_attention is None and block.encoder_attn.guided_attention is None


@pytest.mark.parametrize("native_data",[2],indirect=True)
def test_mixed_head_specification_is_rejected_during_native_loading(native_data):
    from fairseq import tasks
    root=native_data.root; layered_teacher(root)
    plan=b.make_plan(root/"baseline.json",root/"run","distillation","t2tt",root/"manifest.jsonl",root/"units",root/"teacher",all_layers=True)
    write_json(root/"run/native-run.json",plan)
    store=ArtifactStore(root/"teacher"); meta,packet=store.get("1")
    meta["heads"]=[0]
    spec={k:meta[k] for k in ("all_layers","decoder_layers","layer_indices","heads")}
    meta["attention_spec_hash"]=digest(spec); store.put("1",meta,packet,overwrite=True)
    cfg=b.native_config(plan["config"]); task=tasks.setup_task(cfg.task); task.load_dataset("train")
    with pytest.raises(ValueError,match="mixed teacher"): task.dataset("train")[1]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason="all-layer native FP16 connection requires CUDA")
@pytest.mark.parametrize("native_data",[2],indirect=True)
def test_all_layer_fp16_trainer_and_fp32_prior(native_data):
    from fairseq import tasks
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root=native_data.root; layered_teacher(root)
    plan=b.make_plan(root/"baseline.json",root/"cuda","annealed_guidance","t2tt",root/"manifest.jsonl",root/"units",root/"teacher",anneal_steps=1,all_layers=True)
    write_json(root/"cuda/native-run.json",plan)
    cfg=b.native_config(plan["config"]); cfg.common.cpu=False; cfg.common.fp16=True
    task=tasks.setup_task(cfg.task); task.load_dataset("train"); ds=task.dataset("train")
    samples=[ds.collater([ds[i]]) for i in range(2)]
    with trainer_hooks(plan["contract_hash"]):
        trainer=train.Trainer(cfg,task,task.build_model(cfg.model),task.build_criterion(cfg.criterion))
        prepared,_=trainer._prepare_sample(samples[0])
        assert prepared["alignment_prior"].dtype==torch.float32 and prepared["alignment_prior"].size(1)==2
        assert prepared["net_input"]["src_tokens"].dtype==torch.float16
        trainer.train_step(samples); trainer.train_step(samples)
        assert trainer.get_num_updates()==2 and all(torch.isfinite(p).all() for p in trainer.model.parameters())
