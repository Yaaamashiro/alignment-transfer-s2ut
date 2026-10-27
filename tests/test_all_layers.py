"""Layer correspondence, single-pass capture and native beta-zero parity."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from alignment_transfer_s2ut.config import BackboneConfig
from alignment_transfer_s2ut.s2ut.model import build_model
from alignment_transfer_s2ut.training import UNIT_VOCABULARY
from alignment_transfer_s2ut.guidance.attention import alignment_kl
from test_attention_extraction import attention_data, save_checkpoint, arguments, legacy_packets


def tiny():
    return BackboneConfig(encoder_layers=1, decoder_layers=2, encoder_embed_dim=16,
        decoder_embed_dim=16, encoder_ffn_embed_dim=32, decoder_ffn_embed_dim=32,
        encoder_attention_heads=2, decoder_attention_heads=2, dropout=0,
        attention_dropout=0, activation_dropout=0, conv_channels=32, input_feat_per_channel=8)


@pytest.mark.parametrize("backend", ["smoke", "fairseq"])
def test_all_layer_capture_guidance_gradients_and_beta_zero(backend):
    if backend == "fairseq":
        pytest.importorskip("fairseq")
    torch.set_num_threads(1); torch.manual_seed(11)
    model = build_model(tiny(), "speech", [], UNIT_VOCABULARY, backend).eval()
    source, lengths, previous = torch.randn(2,40,8), torch.tensor([40,32]), torch.tensor([[2,4,5],[2,6,7]])
    encoded = model.encode(source,lengths)
    size = int(model.encoder_lengths(encoded).max())
    prior = torch.zeros(2,2,3,size); prior[:,0,:,0] = 1; prior[:,1,:,1] = 1
    plain = model.decode(previous,encoded)[0]
    calls=[]
    handles=[]
    if backend == "fairseq":
        handles.append(model.network.decoder.register_forward_hook(lambda *args: calls.append(1)))
    actual, raw, guided = model.decode(previous,encoded,prior,beta=0,all_layers=True)
    torch.testing.assert_close(actual,plain,atol=0,rtol=0)
    assert raw.shape == guided.shape == prior.shape
    if backend == "fairseq":
        assert calls == [1]
    assert not torch.equal(raw[:,0],raw[:,1])
    _, raw, guided = model.decode(previous,encoded,prior,beta=1,all_layers=True)
    assert guided[:,0,:,0].mean() > raw[:,0,:,0].mean()
    assert guided[:,1,:,1].mean() > raw[:,1,:,1].mean()
    loss=sum(alignment_kl(prior[:,l],raw[:,l]) for l in range(2))/2
    loss.backward()
    blocks=model.network.decoder.layers if backend=="fairseq" else model.decoder
    for block in blocks:
        attn=block.encoder_attn if backend=="fairseq" else block.cross
        for projection in ((attn.q_proj,attn.k_proj) if backend=="fairseq" else (attn.q,attn.k)):
            assert torch.isfinite(projection.weight.grad).all() and projection.weight.grad.abs().sum()>0
        if backend=="fairseq":
            assert attn.transfer_prior is None and not attn.transfer_capture
            assert attn.raw_attention is None and attn.guided_attention is None
    for h in handles: h.remove()


def test_native_capture_cleanup_after_decoder_exception(monkeypatch):
    pytest.importorskip("fairseq")
    model=build_model(tiny(),"speech",[],UNIT_VOCABULARY).eval()
    encoded=model.encode(torch.randn(1,40,8),torch.tensor([40]))
    def fail(*args,**kw): raise RuntimeError("decoder failure")
    monkeypatch.setattr(model.network.decoder,"forward",fail)
    with pytest.raises(RuntimeError,match="decoder failure"):
        model.decode(torch.tensor([[2,4]]),encoded,all_layers=True)
    for block in model.network.decoder.layers:
        assert block.encoder_attn.transfer_prior is None and not block.encoder_attn.transfer_capture
        assert block.encoder_attn.raw_attention is None and block.encoder_attn.guided_attention is None


@pytest.mark.parametrize("backend",["smoke","fairseq"])
def test_extraction_all_layers_once_masks_and_resume(tmp_path,attention_data,backend,monkeypatch):
    from alignment_transfer_s2ut import training as t
    from alignment_transfer_s2ut.io import ArtifactStore
    if backend=="fairseq": pytest.importorskip("fairseq")
    rows,manifest=attention_data
    checkpoint,saved=save_checkpoint(tmp_path,rows,backend,tiny())
    # Text frontend does not use the acoustic feature width.
    model=t.checkpoint_model(saved).eval(); calls=[]
    model.register_forward_hook(lambda *args:calls.append(1))
    monkeypatch.setattr(t,"checkpoint_model",lambda *args:model)
    args=arguments(manifest,checkpoint,tmp_path/"all","--all-layers")
    result=t.extract_attention(args)
    assert calls==[1] and result["forwards"]==1
    meta,packet=ArtifactStore(args.output).get("0")
    assert meta["all_layers"] and meta["layer_indices"]==[0,1]
    assert packet["attention_layers"].shape==(2,2,2)
    assert packet["attention_layers"].dtype==np.float32
    assert not np.array_equal(packet["attention_layers"][0],packet["attention_layers"][1])
    np.testing.assert_array_equal(packet["attention"],packet["attention_layers"][-1])
    assert not ArtifactStore(args.output).get("5")[1]["attention_layers"].any()
    args.resume=True
    assert t.extract_attention(args)["forwards"]==0 and calls==[1]
    legacy=arguments(manifest,checkpoint,tmp_path/"single")
    t.extract_attention(legacy)
    with pytest.raises(ValueError,match="provenance|scope"):
        t.extract_attention(arguments(manifest,checkpoint,legacy.output,"--all-layers","--resume"))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason="all-layer extraction parity requires CUDA")
def test_all_layer_extraction_on_cuda(tmp_path,attention_data):
    from alignment_transfer_s2ut import training as t
    from alignment_transfer_s2ut.io import ArtifactStore
    pytest.importorskip("fairseq")
    rows,manifest=attention_data; checkpoint,saved=save_checkpoint(tmp_path,rows,"fairseq",tiny())
    expected=[legacy_packets(rows,saved,checkpoint,heads=None,device="cuda",layer=l) for l in range(2)]
    output=tmp_path/"all-cuda"
    assert t.extract_attention(arguments(manifest,checkpoint,output,"--all-layers","--device","cuda"))["forwards"]==1
    for row in rows:
        packet=ArtifactStore(output).get(row["pair_id"])[1]
        for l in range(2):
            np.testing.assert_allclose(packet["attention_layers"][l],expected[l][row["pair_id"]][1],atol=2e-6,rtol=2e-5)


def test_projection_layers_equal_independent_composition():
    from alignment_transfer_s2ut.alignment.projection import project_packet_layers,project_packet
    packet=dict(mt_attention_layers=np.array([[[1.,0],[0,1]],[[0.,1],[1,0]]]),
        source_spans=np.array([[0.,.5],[.5,1.]]),target_spans=np.array([[0.,.4],[.4,1.]]),
        unit_spans=np.array([[0.,.2],[.2,.7],[.7,1.]]),source_duration=np.array(1.))
    projected=project_packet_layers(packet,4)
    assert projected.shape==(2,3,4)
    for l in range(2):
        np.testing.assert_array_equal(projected[l],project_packet(dict(packet,mt_attention=packet["mt_attention_layers"][l]),4))


def test_attention_budget_includes_decoder_layer_factor(tmp_path,attention_data,monkeypatch):
    from alignment_transfer_s2ut import training as t
    rows,manifest=attention_data
    checkpoint,saved=save_checkpoint(tmp_path,rows,"smoke",tiny())
    model=t.checkpoint_model(saved).eval(); cells=[]
    def inspect(module,args): cells.append(args[0].size(0)*2*args[0].size(1)*args[2].size(1))
    model.register_forward_pre_hook(inspect)
    monkeypatch.setattr(t,"checkpoint_model",lambda *args:model)
    result=t.extract_attention(arguments(manifest,checkpoint,tmp_path/"limited","--all-layers","--max-attention-cells",80))
    assert result["forwards"]>1 and max(cells)<=80
    with pytest.raises(ValueError,match="budget"):
        t.extract_attention(arguments(manifest,checkpoint,tmp_path/"small","--all-layers","--max-attention-cells",8))


def test_cli_scope_conflict_is_only_explicit():
    from alignment_transfer_s2ut.cli import parser_for
    for stage in ("train_t2tt","train_s2ut","extract_t2tt_attention","visualize_attention"):
        required=["--manifest","m","--output","o"]
        if stage in {"train_s2ut","visualize_attention"}: required += ["--units","u"]
        if stage in {"extract_t2tt_attention","visualize_attention"}: required += ["--checkpoint","c"]
        if stage=="visualize_attention": required += ["--pair-id","p"]
        assert parser_for(stage).parse_args([*required,"--all-layers"]).all_layers
        with pytest.raises(ValueError,match="mutually exclusive"):
            parser_for(stage).parse_args([*required,"--all-layers","--layer","-1"])


def test_layer_means_preserve_accumulated_gradients_with_different_row_counts():
    from alignment_transfer_s2ut.guidance.attention import layer_kl_sums,layer_row_counts
    torch.manual_seed(3)
    teacher=torch.rand(3,2,4,5); teacher/=teacher.sum(-1,keepdim=True)
    teacher[0,0,1:]=0; teacher[1,1,:2]=0; teacher[2,0]=0
    counts=layer_row_counts(teacher)
    assert counts[0]!=counts[1]
    full=torch.randn_like(teacher,requires_grad=True); accumulated=full.detach().clone().requires_grad_()
    expected=(layer_kl_sums(teacher,full.softmax(-1))/counts).mean(); expected.backward()
    for part in (slice(0,1),slice(1,3)):
        loss=(layer_kl_sums(teacher[part],accumulated[part].softmax(-1))/counts).mean()
        loss.backward()
    torch.testing.assert_close(full.grad,accumulated.grad,atol=1e-8,rtol=1e-6)


@pytest.mark.parametrize("backend",["smoke","fairseq"])
def test_all_layer_teacher_and_legacy_training_pipeline(tmp_path,backend):
    import soundfile as sf
    import yaml
    from alignment_transfer_s2ut.smoke_pipeline import invoke,synthetic_frontends
    from alignment_transfer_s2ut.io import write_jsonl,read_json,ArtifactStore
    if backend=="fairseq": pytest.importorskip("fairseq")
    rows=[]
    for i,split in enumerate(("train","train","dev")):
        row=dict(pair_id=str(i),split=split,ja_text="あい"+"あ"*i,en_text="ab"+"b"*i)
        for lang in ("ja","en"):
            path=tmp_path/f"{i}-{lang}.wav"
            sf.write(path,.1*np.sin(2*np.pi*(200+i*20)*np.arange(8000)/16000),16000)
            row[lang+"_audio"]=str(path)
        rows.append(row)
    write_jsonl(tmp_path/"input.jsonl",rows)
    manifest=tmp_path/"manifest.jsonl"
    invoke("prepare_manifest","--input",tmp_path/"input.jsonl","--output",manifest)
    config=tmp_path/"tiny.yaml"; config.write_text(yaml.safe_dump(dict(backbone=tiny().asdict())),encoding="utf-8")
    common=["--manifest",manifest,"--config",config,"--backend",backend,"--batch-size",1,"--update-frequency",2,"--max-updates",2,"--warmup-updates",0]
    invoke("train_t2tt",*common,"--output",tmp_path/"t2tt")
    invoke("extract_t2tt_attention","--manifest",manifest,"--checkpoint",tmp_path/"t2tt/checkpoint_best.pt","--all-layers","--output",tmp_path/"mt")
    synthetic_frontends(manifest,tmp_path/"ctc",tmp_path/"units")
    invoke("build_teacher_alignment","--manifest",manifest,"--config",config,"--mt-attention",tmp_path/"mt","--alignment",tmp_path/"ctc","--units",tmp_path/"units","--output",tmp_path/"teacher")
    meta,packet=ArtifactStore(tmp_path/"teacher").get("0")
    assert packet["mt_attention_layers"].shape==(2,2,2) and meta["attention_spec_hash"]
    from argparse import Namespace
    from alignment_transfer_s2ut.diagnose_alignment import diagnose
    audit=diagnose(Namespace(manifest=manifest,config=config,teacher=tmp_path/"teacher",units=tmp_path/"units",
        output=tmp_path/"audit",split="dev",seed=1,max_samples=1,heatmaps=0,source_mapping="frontend",
        prior_smoothing=.05,overwrite=False,all_layers=True))
    assert len(audit["layers"])==2 and audit["compatibility_heatmap_layer"]==1
    # An old single-layer packet can be reused without adding metadata/writes.
    invoke("extract_t2tt_attention","--manifest",manifest,"--checkpoint",tmp_path/"t2tt/checkpoint_best.pt","--output",tmp_path/"old-mt")
    teacher_args=["--manifest",manifest,"--config",config,"--mt-attention",tmp_path/"old-mt","--alignment",tmp_path/"ctc","--units",tmp_path/"units","--output",tmp_path/"old-teacher"]
    invoke("build_teacher_alignment",*teacher_args)
    old=ArtifactStore(tmp_path/"old-teacher")
    for row in rows:
        metadata,values=old.get(row["pair_id"])
        metadata={k:v for k,v in metadata.items() if k not in {"all_layers","layer","heads","attention_spec_hash"}}
        old.put(row["pair_id"],metadata,values,overwrite=True)
    hashes=[old.hash(row["pair_id"]) for row in rows]
    invoke("build_teacher_alignment",*teacher_args,"--resume")
    assert hashes==[old.hash(row["pair_id"]) for row in rows]
    from alignment_transfer_s2ut.training import prior_builder
    from alignment_transfer_s2ut.data.manifest import load_manifest,manifest_hash
    canonical=load_manifest(manifest)
    params=([canonical[0]],old,ArtifactStore(tmp_path/"units"),tiny(),manifest_hash(canonical),backend,"cpu")
    assert prior_builder(*params)(torch.tensor([4]),6).shape==(1,6,4)
    with pytest.raises(ValueError,match="all-layer T2TT teacher"):
        prior_builder(*params,all_layers=True)
    for method in ("distillation","annealed_guidance","guidance"):
        invoke("train_s2ut",*common,"--trainer","legacy","--all-layers","--method",method,"--anneal-steps",1,"--teacher",tmp_path/"teacher","--units",tmp_path/"units","--output",tmp_path/method)
        logs=[__import__('json').loads(x) for x in (tmp_path/method/"metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all("alignment/kl_layer_0" in log and "alignment/kl_layer_1" in log for log in logs)
        assert read_json(tmp_path/method/"experiment.json")["transfer"]["all_layers"]
        invoke("visualize_attention","--manifest",manifest,"--checkpoint",tmp_path/method/"checkpoint_best.pt","--pair-id","2","--units",tmp_path/"units","--teacher",tmp_path/"teacher","--layer",0,"--output",tmp_path/(method+"-plot"))
