import numpy as np
import pytest
import torch
from alignment_transfer_s2ut import benchmark as b
from alignment_transfer_s2ut.data.manifest import load_manifest
from alignment_transfer_s2ut.io import ArtifactStore,write_json,digest
from alignment_transfer_s2ut.tokenization.sentencepiece import SentencePieceTokenizer
from alignment_transfer_s2ut.tokenization.alignment import aggregate_ctc_spans
from test_benchmark_native import native_data
from test_all_layers_native import layered_teacher


@pytest.mark.parametrize("native_data",[2],indirect=True)
def test_native_criterion_accepts_subword_teacher_without_changing_unit_labels(native_data):
    from fairseq import tasks
    from alignment_transfer_s2ut.s2ut import benchmark_extension  # Register the native task/criterion in standalone runs.
    root=native_data.root;layered_teacher(root)
    rows=load_manifest(root/"manifest.jsonl",check_audio=False)
    source,target=[SentencePieceTokenizer.fit(rows,field,128) for field in ("ja_text","en_text")]
    store=ArtifactStore(root/"teacher")
    for row in rows:
        meta,packet=store.get(row["pair_id"])
        src=np.array([p[1:] for p in source.encode_with_spans(row["ja_text"])])
        tgt=np.array([p[1:] for p in target.encode_with_spans(row["en_text"])])
        layers=np.zeros((2,len(tgt),len(src)),np.float32);layers[0,:,0]=1.;layers[1,:,-1]=1.
        packet.update(mt_attention=layers[-1],mt_attention_layers=layers,source_token_spans=src,target_token_spans=tgt,
            source_spans=aggregate_ctc_spans(packet["source_spans"],src),target_spans=aggregate_ctc_spans(packet["target_spans"],tgt))
        meta.update(tokenizer_type="sentencepiece",source_tokenizer_hash=source.fingerprint,target_tokenizer_hash=target.fingerprint,
            source_tokenizer_spec=source.spec,target_tokenizer_spec=target.spec,token_span_unit="unicode-codepoint",
            tokenizer_hash=digest([source.fingerprint,target.fingerprint]))
        store.put(row["pair_id"],meta,packet,overwrite=True)
    plan=b.make_plan(root/"baseline.json",root/"run","distillation","t2tt",root/"manifest.jsonl",root/"units",root/"teacher",all_layers=True)
    write_json(root/"run/native-run.json",plan)
    cfg=b.native_config(plan["config"]);task=tasks.setup_task(cfg.task);task.load_dataset("train")
    ds=task.dataset("train");sample=ds.collater([ds[0]])
    assert sample["alignment_prior"].shape[1]==2
    model=task.build_model(cfg.model);criterion=task.build_criterion(cfg.criterion)
    for name in task.multitask_tasks:
        criterion.set_multitask_loss_weight(name,task.multitask_tasks[name].args.get_loss_weight(0))
    loss,_,_=criterion(model,sample);loss.backward()
    assert torch.isfinite(loss) and model.decoder.layers[0].encoder_attn.q_proj.weight.grad is not None
