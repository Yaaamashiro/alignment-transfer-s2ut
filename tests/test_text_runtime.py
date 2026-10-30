from pathlib import Path
import sys
import pytest
import torch
import yaml
from alignment_transfer_s2ut import text_t2tt_runtime as runtime,colab_runtime as c
from alignment_transfer_s2ut.io import write_jsonl,read_json,file_hash
from alignment_transfer_s2ut.text_manifest import prepare
from alignment_transfer_s2ut.training import load_checkpoint
from alignment_transfer_s2ut.smoke_pipeline import invoke,TINY_CONFIG
from test_benchmark_colab import benchmark_runtime
from test_t2tt_extension import equal_state


@pytest.mark.parametrize("tokenizer",["character","sentencepiece"])
def test_text_runtime_restore_keeps_canonical_identity_after_audio_metadata_added(tmp_path,monkeypatch,benchmark_runtime,tokenizer):
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setenv("OMP_NUM_THREADS","1");monkeypatch.setenv("MKL_NUM_THREADS","1")
    try:
        values=[dict(pair_id=str(i),split="train" if i<2 else "dev",ja_text="日本語 text "+str(i),en_text="text play "+str(i)) for i in range(3)]
        source,manifest=tmp_path/"external.jsonl",tmp_path/"input/manifest.jsonl";write_jsonl(source,values);prepare(source,manifest)
        initial=file_hash(manifest)
        config=tmp_path/"config.yaml";config.write_text(yaml.safe_dump(TINY_CONFIG),encoding="utf-8")
        arguments=["--manifest",str(manifest),"--config",str(config),"--backend","smoke","--tokenizer",tokenizer,
            "--source-vocab-size","96","--target-vocab-size","96","--batch-size","1","--update-frequency","1", "--max-updates","4"]
        work,backup=tmp_path/"text-run",tmp_path/"drive/run"
        command=[sys.executable,"-m","alignment_transfer_s2ut.train_t2tt",*arguments,"--output",str(work),"--resume-if-exists"]
        publish=benchmark_runtime.publish_packed
        def interrupted(files,destination,identity,updates):
            result=publish(files,destination,identity,updates)
            if updates==2:raise OSError("interrupted after update 2")
            return result
        monkeypatch.setattr(benchmark_runtime,"publish_packed",interrupted)
        with pytest.raises(OSError,match="interrupted"):
            runtime.start(command,work,backup,snapshot_updates=2)
        monkeypatch.setattr(benchmark_runtime,"publish_packed",publish)
        work.rename(tmp_path/"lost-vm")
        write_jsonl(source,[dict(row,ja_audio="future-ja.wav",en_audio="future-en.wav") for row in values],overwrite=True)
        prepare(source,manifest,resume=True);assert file_hash(manifest)==initial
        assert runtime.start(command,work,backup,snapshot_updates=2)["durable_updates"]==4
        assert runtime.status(work,4)["manifest_identity"]=="text-v1"
        reference=tmp_path/"reference";invoke("train_t2tt",*arguments,"--output",reference)
        for key in ("model","optimizer","torch_rng","epoch","cursor","contract_hash"):
            equal_state(load_checkpoint(work/"checkpoint_last.pt")[key],load_checkpoint(reference/"checkpoint_last.pt")[key])
        current=c.runtime().latest(backup,c.identity_for(command,work))
        assert read_json(current[0]/"metrics-summary.json")["durable_update"]==4
    finally:
        torch.set_num_threads(old_threads)
