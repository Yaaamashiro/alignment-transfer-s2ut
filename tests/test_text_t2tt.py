from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from alignment_transfer_s2ut.smoke_pipeline import invoke,synthetic_frontends
from alignment_transfer_s2ut.data.manifest import TEXT_IDENTITY,text_manifest_hash,load_manifest,manifest_hash
from alignment_transfer_s2ut.io import write_jsonl,read_json,file_hash,ArtifactStore
from alignment_transfer_s2ut.training import load_checkpoint
from test_all_layers import tiny


def dataset(tmp_path):
    rows=[dict(pair_id=str(i),split="train" if i<2 else "dev",ja_text="私は猫 playing "+str(i),en_text="playing play ing "+str(i)) for i in range(3)]
    path=tmp_path/"manifest.jsonl";write_jsonl(path,rows)
    config=tmp_path/"tiny.yaml";config.write_text(yaml.safe_dump(dict(backbone=tiny().asdict())),encoding="utf-8")
    return rows,path,config


@pytest.mark.parametrize("backend",["smoke","fairseq"])
@pytest.mark.parametrize("tokenizer",["character","sentencepiece"])
def test_text_only_train_eval_extract_and_audio_attachment_resume(tmp_path,monkeypatch,backend,tokenizer):
    if backend=="fairseq":pytest.importorskip("fairseq")
    rows,path,config=dataset(tmp_path)
    args=["--manifest",path,"--config",config,"--backend",backend,"--batch-size",1,"--update-frequency",1,"--max-updates",2,
        "--tokenizer",tokenizer,"--source-vocab-size",96,"--target-vocab-size",96,"--warmup-updates",0]
    def forbidden(*a,**kw):raise AssertionError("T2TT touched audio")
    with monkeypatch.context() as patch:
        patch.setattr("alignment_transfer_s2ut.training.features",forbidden)
        patch.setattr("alignment_transfer_s2ut.training._audio_hash",forbidden)
        patch.setattr("alignment_transfer_s2ut.data.manifest.local_path",forbidden)
        resolve=Path.resolve
        def safe_resolve(p,*a,**kw):
            assert p.suffix!=".wav","audio path was resolved"
            return resolve(p,*a,**kw)
        patch.setattr(Path,"resolve",safe_resolve)
        invoke("train_t2tt",*args,"--output",tmp_path/"t2tt")
        checkpoint=tmp_path/"t2tt/checkpoint_best.pt";saved=load_checkpoint(checkpoint)
        assert saved["manifest_identity"]==TEXT_IDENTITY
        assert saved["train_hash"]==text_manifest_hash(rows[:2]) and saved["dev_hash"]==text_manifest_hash(rows[2:])
        evaluation=["--manifest",path,"--checkpoint",checkpoint,"--beam-size",1,"--max-chars",4,"--output",tmp_path/"dev.json"]
        result=invoke("evaluate_t2tt",*evaluation)
        extraction=["--manifest",path,"--checkpoint",checkpoint,"--all-layers","--output",tmp_path/"attention"]
        invoke("extract_t2tt_attention",*extraction)
        before={pair:ArtifactStore(tmp_path/"attention").hash(pair) for pair in ("0","1","2")}
        original=file_hash(checkpoint)
        write_jsonl(path,[dict(row,ja_audio=f"missing-ja{i}.wav",en_audio=f"missing-en{i}.wav",ja_sha256="future",en_sha256="future") for i,row in enumerate(rows)],overwrite=True)
        invoke("train_t2tt",*args,"--output",tmp_path/"t2tt","--resume")
        assert file_hash(checkpoint)==original
        assert invoke("evaluate_t2tt",*evaluation,"--resume")==result
        invoke("extract_t2tt_attention",*extraction,"--resume")
        assert {pair:ArtifactStore(tmp_path/"attention").hash(pair) for pair in before}==before
    changed=deepcopy(rows);changed[0]["en_text"]="changed text";write_jsonl(path,changed,overwrite=True)
    with pytest.raises(ValueError,match="training pairs"):
        invoke("evaluate_t2tt",*evaluation,"--output",tmp_path/"changed.json")


def test_text_attention_projects_after_audio_is_added(tmp_path):
    import soundfile as sf
    rows,path,config=dataset(tmp_path)
    invoke("train_t2tt","--manifest",path,"--config",config,"--backend","smoke","--max-updates",2,"--output",tmp_path/"t2tt")
    invoke("extract_t2tt_attention","--manifest",path,"--checkpoint",tmp_path/"t2tt/checkpoint_best.pt","--all-layers","--output",tmp_path/"attention")
    identity=text_manifest_hash(rows)
    for i,row in enumerate(rows):
        for lang in ("ja","en"):
            audio=tmp_path/f"{lang}{i}.wav";sf.write(audio,.1*np.sin(np.arange(8000)*.1),16000)
            row[lang+"_audio"]=str(audio)
    write_jsonl(path,rows,overwrite=True)
    synthetic_frontends(path,tmp_path/"ctc",tmp_path/"units")
    invoke("build_teacher_alignment","--manifest",path,"--config",config,"--mt-attention",tmp_path/"attention",
        "--alignment",tmp_path/"ctc","--units",tmp_path/"units","--output",tmp_path/"teacher")
    meta,packet=ArtifactStore(tmp_path/"teacher").get("0")
    assert meta["manifest_hash"]==manifest_hash(load_manifest(path))
    assert meta["text_manifest_hash"]==identity
    assert packet["mt_attention_layers"].shape[0]==2


@pytest.mark.parametrize("stage",["train_s2ut","align_speech_text","extract_units","build_teacher_alignment","evaluate","visualize_attention"])
def test_audio_stages_reject_text_only_before_loading_assets(tmp_path,stage):
    rows,path,config=dataset(tmp_path)
    arguments={
        "train_s2ut":["--backend","smoke","--trainer","legacy","--units","missing","--config",config],
        "align_speech_text":["--config","missing"],"extract_units":["--config","missing"],
        "build_teacher_alignment":["--mt-attention","missing","--alignment","missing","--units","missing"],
        "evaluate":["--predictions","missing"],
        "visualize_attention":["--checkpoint","missing","--units","missing","--pair-id","0"]}
    with pytest.raises(ValueError,match="audio-required"):
        invoke(stage,"--manifest",path,"--output",tmp_path/"output",*arguments[stage])
