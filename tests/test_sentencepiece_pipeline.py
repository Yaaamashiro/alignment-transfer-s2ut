from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from alignment_transfer_s2ut.smoke_pipeline import invoke,synthetic_frontends
from alignment_transfer_s2ut.io import ArtifactStore,write_jsonl,read_json,read_jsonl,digest,file_hash
from alignment_transfer_s2ut.training import load_checkpoint,prior_builder
from alignment_transfer_s2ut.data.manifest import load_manifest,manifest_hash
from alignment_transfer_s2ut.tokenization.factory import tokenizers_from_checkpoint
from alignment_transfer_s2ut.tokenization.alignment import aggregate_ctc_spans
from test_all_layers import tiny


@pytest.mark.parametrize("backend",["smoke","fairseq"])
def test_spm_train_resume_evaluate_attention_teacher_and_student(tmp_path,backend):
    import soundfile as sf
    if backend=="fairseq":pytest.importorskip("fairseq")
    rows=[]
    for i,split in enumerate(("train","train","dev")):
        row=dict(pair_id=str(i),split=split,ja_text="私は猫が好き playing play "+str(i),en_text="playing play ing ab "+str(i))
        for lang in ("ja","en"):
            path=tmp_path/f"{i}-{lang}.wav";sf.write(path,.1*np.sin(np.arange(8000)*.1),16000)
            row[lang+"_audio"]=str(path)
        rows.append(row)
    manifest=tmp_path/"manifest.jsonl";write_jsonl(manifest,rows)
    config=tmp_path/"tiny.yaml";config.write_text(yaml.safe_dump(dict(backbone=tiny().asdict())),encoding="utf-8")
    common=["--manifest",manifest,"--config",config,"--backend",backend,"--batch-size",1,"--update-frequency",2,
        "--max-updates",2,"--warmup-updates",0]
    run=tmp_path/"spm";sp=["--tokenizer","sentencepiece","--source-vocab-size",96,"--target-vocab-size",96]
    invoke("train_t2tt",*common,*sp,"--output",run)
    saved=load_checkpoint(run/"checkpoint_best.pt");source,target=tokenizers_from_checkpoint(saved)
    assert saved["backbone"]==tiny().asdict() and saved["step"]>=1
    report=read_json(run/"tokenization.json")
    assert report["source"]["actual_vocab_size"]==len(source.vocabulary)
    assert report["statistics"]["dev_ja"]["unknown_codepoints"]==1  # The held-out digit 2.
    before=file_hash(run/"checkpoint_last.pt")
    with pytest.raises(ValueError,match="vocabulary request changed"):
        invoke("train_t2tt",*common,*sp,"--source-vocab-size",128,"--output",run,"--resume")
    asset=run/"tokenizers/ja.model";original=asset.read_bytes();asset.write_bytes(original+b"bad")
    with pytest.raises(ValueError,match="asset differs"):
        invoke("train_t2tt",*common,*sp,"--output",run,"--resume")
    assert file_hash(run/"checkpoint_last.pt")==before
    asset.write_bytes(original)
    invoke("train_t2tt",*common,*sp,"--max-updates",3,"--output",run,"--resume")
    assert load_checkpoint(run/"checkpoint_last.pt")["step"]==3
    checkpoint=run/"checkpoint_best.pt"
    invoke("evaluate_t2tt","--manifest",manifest,"--checkpoint",checkpoint,"--output",tmp_path/"dev.json","--beam-size",1,"--max-chars",4)
    # Embedded models suffice after the optional exported tokenizer files disappear.
    for asset in (run/"tokenizers").iterdir():asset.unlink()
    assert tokenizers_from_checkpoint(load_checkpoint(checkpoint))[0].fingerprint==source.fingerprint
    for all_layers in (False,True):
        output=tmp_path/("mt-all" if all_layers else "mt-single")
        invoke("extract_t2tt_attention","--manifest",manifest,"--checkpoint",checkpoint,"--output",output,*(["--all-layers"] if all_layers else []))
        meta,values=ArtifactStore(output).get("0")
        expected=(len(target.encode(rows[0]["en_text"])),len(source.encode(rows[0]["ja_text"])))
        assert values["attention"].shape==expected
        if all_layers:assert values["attention_layers"].shape==(2,*expected)
        assert values["source_token_spans"].shape==(expected[1],2)
        assert meta["tokenizer_type"]=="sentencepiece"
        invoke("extract_t2tt_attention","--manifest",manifest,"--checkpoint",checkpoint,"--output",output,"--resume",*(["--all-layers"] if all_layers else []))
    synthetic_frontends(manifest,tmp_path/"ctc",tmp_path/"units")
    teacher=tmp_path/"teacher"
    args=["--manifest",manifest,"--config",config,"--mt-attention",tmp_path/"mt-all","--alignment",tmp_path/"ctc","--units",tmp_path/"units","--output",teacher]
    invoke("build_teacher_alignment",*args)
    meta,packet=ArtifactStore(teacher).get("0")
    _,ctc=ArtifactStore(tmp_path/"ctc").get("0:ja")
    np.testing.assert_array_equal(packet["source_spans"],aggregate_ctc_spans(ctc["spans"],packet["source_token_spans"]))
    canonical=load_manifest(manifest)
    build=prior_builder([canonical[0]],ArtifactStore(teacher),ArtifactStore(tmp_path/"units"),tiny(),manifest_hash(canonical),backend,"cpu",all_layers=True)
    prior=build(torch.tensor([8]),len(packet["reduced_units"])+1)
    assert prior.shape[1]==2 and prior.shape[-1]==8
    sums=prior.sum(-1);assert torch.all((sums==0)|torch.isclose(sums,torch.ones_like(sums)))
    invoke("train_s2ut",*common,"--max-updates",1,"--trainer","legacy","--all-layers","--method","distillation",
        "--teacher",teacher,"--units",tmp_path/"units","--output",tmp_path/"student")
    # Merely labelling subword attention as character cannot bypass span checks.
    store=ArtifactStore(tmp_path/"mt-all");bad,values=store.get("0");bad["tokenizer_type"]="character"
    store.put("0",bad,values,overwrite=True)
    with pytest.raises(ValueError,match="mixed"):
        invoke("build_teacher_alignment",*args,"--resume")


@pytest.mark.parametrize("name",["checkpoint_last.pt","checkpoint_best.pt"])
def test_resume_from_character_into_spm_is_rejected(tmp_path,name):
    from alignment_transfer_s2ut.tokenization.factory import fit_tokenizers
    torch.save(dict(source_vocabulary=[],target_vocabulary=[]),tmp_path/name)
    with pytest.raises(ValueError,match="start at step 0"):
        fit_tokenizers([],Namespace(tokenizer="sentencepiece",resume=True,output=tmp_path),"text")
