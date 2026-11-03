from native_plan_fixture import fixture_plan
"""All-layer research path: real pinned fairseq task, criterion and Trainer."""
import numpy as np
import pytest
import torch
from test_benchmark_native import native_data
from alignment_transfer_s2ut import benchmark as b
from alignment_transfer_s2ut.io import ArtifactStore,write_json,digest
from alignment_transfer_s2ut.data.manifest import load_manifest


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
def test_native_scope_and_layer_count_contract_rejections(native_data):
    root=native_data.root
    args=(root/"baseline.json",root/"run","distillation","t2tt",root/"manifest.jsonl",root/"units",root/"teacher")
    with pytest.raises(ValueError,match="all-layer T2TT teacher"): fixture_plan(*args,all_layers=True)
    layered_teacher(root)
    plan=fixture_plan(*args,all_layers=True)
    single=fixture_plan(*args,all_layers=False)
    assert plan["contract_hash"]!=single["contract_hash"]
    write_json(root/"run/native-run.json",plan)
    with pytest.raises(ValueError,match="contract exists"): b.train_native(single,root/"run",resume=True)
    store=ArtifactStore(root/"teacher"); meta,packet=store.get("0")
    packet["mt_attention_layers"]=packet["mt_attention_layers"][:1]
    store.put("0",meta,packet,overwrite=True)
    with pytest.raises(ValueError,match="layer count"): fixture_plan(*args,all_layers=True)




@pytest.mark.parametrize("native_data",[2],indirect=True)
@pytest.mark.parametrize("all_layers",[False,True])
def test_two_layer_beta_zero_logits_loss_update_match_stock(native_data,all_layers):
    from fairseq import tasks,utils
    from fairseq.trainer import Trainer as StockTrainer
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root=native_data.root; layered_teacher(root)
    plan=fixture_plan(root/"baseline.json",root/"zero","distillation","t2tt",root/"manifest.jsonl",root/"units",root/"teacher",beta=0.,all_layers=all_layers)
    write_json(root/"zero/native-run.json",plan)
    cfg=b.native_config(plan["config"]); task=tasks.setup_task(cfg.task)
    utils.set_torch_seed(cfg.common.seed); model=task.build_model(cfg.model)
    task.load_dataset("train"); ds=task.dataset("train"); samples=[ds.collater([ds[i]]) for i in range(2)]
    stock_model=native_data.task.build_model(native_data.cfg.model); stock_model.load_state_dict(native_data.initial)
    model.eval(); stock_model.eval()
    with torch.no_grad():
        expected=stock_model(**native_data.samples[0]["net_input"])[0]
        model.transfer_context=(tuple(samples[0]["alignment_prior"].shape),all_layers)
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
    plan=fixture_plan(root/"baseline.json",root/"run","distillation","t2tt",root/"manifest.jsonl",root/"units",root/"teacher",all_layers=True)
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
@pytest.mark.parametrize("method",["distillation","annealed_distillation","mask_distillation"])
def test_all_layer_fp16_trainer_and_fp32_prior(native_data,method):
    from fairseq import tasks
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root=native_data.root; layered_teacher(root)
    plan=fixture_plan(root/"baseline.json",root/"cuda",method,"t2tt",root/"manifest.jsonl",root/"units",root/"teacher",anneal_steps=1,all_layers=True)
    write_json(root/"cuda/native-run.json",plan)
    cfg=b.native_config(plan["config"]); cfg.common.cpu=False; cfg.common.fp16=True
    task=tasks.setup_task(cfg.task); task.load_dataset("train"); ds=task.dataset("train")
    samples=[ds.collater([ds[i]]) for i in range(2)]
    torch.cuda.reset_peak_memory_stats()
    with trainer_hooks(plan["contract_hash"]):
        trainer=train.Trainer(cfg,task,task.build_model(cfg.model),task.build_criterion(cfg.criterion))
        prepared,_=trainer._prepare_sample(samples[0])
        assert prepared["alignment_prior"].dtype==torch.float32 and prepared["alignment_prior"].size(1)==2
        assert prepared["net_input"]["src_tokens"].dtype==torch.float16
        trainer.train_step(samples)
        checkpoint=root/"cuda/checkpoints/checkpoint_last.pt";checkpoint.parent.mkdir()
        trainer.consolidate_optimizer();trainer.save_checkpoint(str(checkpoint),{})
        restarted=train.Trainer(cfg,task,task.build_model(cfg.model),task.build_criterion(cfg.criterion))
        restarted.load_checkpoint(str(checkpoint))
        assert restarted.get_num_updates()==1
        betas=[]
        restarted.model.register_forward_pre_hook(lambda model,args:betas.append(model.transfer_context[1]))
        restarted.train_step(samples)
        assert restarted.get_num_updates()==2 and all(torch.isfinite(p).all() for p in restarted.model.parameters())
        assert betas and all(scope is True for scope in betas)
        torch.cuda.synchronize()
        import json
        print(json.dumps(dict(kind="tiny_all_layer_gpu_preflight",method=method,decoder_layers=2,
            resumed_from_updates=1,completed_updates=2,gpu=torch.cuda.get_device_name(),
            all_layer_capture_after_resume=betas[-1],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())))


@pytest.mark.parametrize("native_data",[2],indirect=True)
@pytest.mark.parametrize("all_layers",[False,True])
def test_native_rejects_selected_head_teacher(native_data,all_layers):
    from alignment_transfer_s2ut.alignment.layers import attention_spec
    root=native_data.root
    if all_layers:layered_teacher(root)
    store=ArtifactStore(root/"teacher");meta,packet=store.get("0")
    meta["heads"]=[0];meta["attention_spec_hash"]=digest(attention_spec(meta))
    store.put("0",meta,packet,overwrite=True)
    with pytest.raises(ValueError,match="all-head-mean teacher"):
        fixture_plan(root/"baseline.json",root/"run","distillation","t2tt",root/"manifest.jsonl",root/"units",root/"teacher",all_layers=all_layers)


@pytest.mark.parametrize("native_data",[2],indirect=True)
def test_native_rejects_single_teacher_from_another_layer(native_data):
    from alignment_transfer_s2ut.alignment.layers import attention_spec
    root=native_data.root;store=ArtifactStore(root/"teacher")
    args=(root/"baseline.json",root/"run","distillation","t2tt",root/"manifest.jsonl",root/"units",root/"teacher")
    meta,packet=store.get("0")
    # Missing legacy metadata still means the final layer; its positive alias is equivalent.
    fixture_plan(*args)
    meta["layer"]=1;meta["attention_spec_hash"]=digest(attention_spec(meta))
    store.put("0",meta,packet,overwrite=True)
    fixture_plan(*args)
    meta["layer"]=0;meta["attention_spec_hash"]=digest(attention_spec(meta))
    store.put("0",meta,packet,overwrite=True)
    with pytest.raises(ValueError,match="layer mismatch"):fixture_plan(*args)

__all__ = ['native_data']  # pytest fixture exports
