from pathlib import Path
import sys
import numpy as np
import pytest
import yaml
from alignment_transfer_s2ut import spm_runtime as s,colab_runtime as c
from alignment_transfer_s2ut.io import write_jsonl,read_json,file_hash
from alignment_transfer_s2ut.training import load_checkpoint
from alignment_transfer_s2ut.smoke_pipeline import invoke,TINY_CONFIG
from test_benchmark_colab import benchmark_runtime
from test_t2tt_extension import equal_state


def test_fresh_spm_verified_backup_restore_and_identical_training(tmp_path,benchmark_runtime,monkeypatch):
    # Keep reference/worker reductions identical, as in the extension tests.
    import torch
    old=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setenv("OMP_NUM_THREADS","1");monkeypatch.setenv("MKL_NUM_THREADS","1")
    try:
        manifest=tmp_path/"manifest.jsonl"
        write_jsonl(manifest,[dict(pair_id=str(i),split="train" if i<2 else "dev",ja_text="私は猫 playing "+str(i),
            en_text="playing play ing "+str(i),ja_audio=f"ja{i}.wav",en_audio=f"en{i}.wav") for i in range(3)])
        config=tmp_path/"config.yaml";config.write_text(yaml.safe_dump(TINY_CONFIG),encoding="utf-8")
        work,backup=tmp_path/"t2tt-spm-200k",tmp_path/"drive/spm"
        arguments=["--manifest",str(manifest),"--config",str(config),"--backend","smoke","--warmup-updates","0",
            "--max-updates","4","--batch-size","1","--update-frequency","1","--tokenizer","sentencepiece",
            "--source-vocab-size","96","--target-vocab-size","96","--validation-interval","1","--save-interval-updates","1"]
        command=[sys.executable,"-m","alignment_transfer_s2ut.train_t2tt",*arguments,"--output",str(work),"--resume-if-exists"]
        original=benchmark_runtime.publish_packed
        def interrupted(files,destination,identity,updates):
            value=original(files,destination,identity,updates)
            if updates==2:raise OSError("simulated published interruption")
            return value
        monkeypatch.setattr(benchmark_runtime,"publish_packed",interrupted)
        with pytest.raises(OSError,match="simulated"):
            s.start(command,work,backup,snapshot_updates=2)
        monkeypatch.setattr(benchmark_runtime,"publish_packed",original)
        assert c.runtime().latest(backup,c.identity_for(command,work))[1]["updates"]==2
        work.rename(tmp_path/"lost-vm")
        assert s.start(command,work,backup,snapshot_updates=2)["durable_updates"]==4
        assert s.status(work,4)["completed_updates"]==4
        actual=load_checkpoint(work/"checkpoint_last.pt")
        reference=tmp_path/"reference"
        invoke("train_t2tt",*arguments,"--output",reference)
        expected=load_checkpoint(reference/"checkpoint_last.pt")
        for key in ("model","optimizer","step","epoch","cursor","torch_rng","contract_hash"):
            equal_state(actual[key],expected[key])
        receipt=c.runtime().latest(backup,c.identity_for(command,work))
        for asset in ("tokenization.json","tokenizers/ja.model","tokenizers/en.model"):
            assert asset in receipt[1]["files"] and (work/asset).is_file()
        summary=read_json(receipt[0]/"metrics-summary.json")
        assert summary["source_updates"]==0 and summary["current_update"]==4
        before=file_hash(work/"checkpoint_last.pt")
        assert s.start(command,work,backup,snapshot_updates=2)["durable_updates"]==4
        assert file_hash(work/"checkpoint_last.pt")==before
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("name",["checkpoint_last.pt","checkpoint_best.pt"])
def test_character_stage_is_not_renamed_or_overwritten(tmp_path,benchmark_runtime,name):
    import torch
    work=tmp_path/"t2tt-200k";work.mkdir()
    path=work/name;torch.save(dict(modality="text"),path)
    before=file_hash(path)
    command=[sys.executable,"-m","alignment_transfer_s2ut.train_t2tt","--manifest",str(tmp_path/"manifest.jsonl"),
        "--output",str(work),"--tokenizer","sentencepiece","--resume-if-exists"]
    with pytest.raises(ValueError,match="character checkpoint is read-only"):
        s.start(command,work,tmp_path/"drive")
    assert file_hash(path)==before and not list(tmp_path.glob("*.interrupted-*"))
