from pathlib import Path
import json
import pytest
from alignment_transfer_s2ut.io import write_json
from alignment_transfer_s2ut.cli import parser_for
from test_notebook_commands import comparison_paths,validate_command
from test_t2tt_extension_notebook import cell


@pytest.mark.parametrize("extended",[False,True])
def test_spm_runs_from_zero_in_independent_stage_and_preserves_controls(tmp_path,extended):
    ns=comparison_paths(tmp_path,extended,all_layers=True,tokenizer="sentencepiece")
    calls=[]
    ns.update(RUN_T2TT_EXTENSION=True,RUN_T2TT_SENTENCEPIECE=True,RUN_TRANSFER=False,RUN_MONOTONIC=False,
        PYTHON="python",PERSISTENT=tmp_path/"drive",MANIFEST=tmp_path/"manifest.jsonl",BACKBONE_CONFIG=tmp_path/"backbone.json",
        T2TT_SNAPSHOT_UPDATES=10000,T2TT_KEEP_LAST=2,LOGS=tmp_path/"logs",log_event=lambda *a:None,
        execute=lambda *a,**kw:calls.append(list(a)),run=lambda *a,**kw:pytest.fail("character stage was started"))
    exec(cell("benchmark_comparison-22"),ns)
    exec(cell("t2tt-extension-run"),ns)
    exec(cell("t2tt-spm-run"),ns)
    assert len(calls)==1
    command=calls[0];validate_command(command[2],command[3:])
    pivot=command.index("alignment_transfer_s2ut.spm_runtime")
    validate_command(command[pivot],command[pivot+1:])
    pivot=command.index("alignment_transfer_s2ut.train_t2tt")
    args=parser_for("train_t2tt").parse_args(command[pivot+1:])
    assert args.tokenizer=="sentencepiece" and args.source_vocab_size==args.target_vocab_size==8000
    assert args.max_updates==200000 and args.batch_size==8 and args.update_frequency==4 and args.max_tokens==20000
    assert args.learning_rate==.0005 and args.warmup_updates==10000 and args.label_smoothing==.2
    assert args.validation_interval==1000 and args.save_interval_updates==100 and args.resume_if_exists
    assert ns["T2TT_ACTIVE_STAGE"]=="t2tt-spm-200k" and ns["T2TT_REQUIRES_COMPLETE"]
    assert ns["T2TT_CHECKPOINT"]==tmp_path/"t2tt-spm-200k/checkpoint_best.pt"
    assert ns["CTC_STAGE"]=="speech-text-alignment-200k"
    assert ns["TEACHER_ROOT"]==tmp_path/"teacher-alignment-spm-200k-all-layers"
    assert ns["DISTILLATION"].name=="distillation-t2tt-spm-200k-all-layers"


def test_spm_evaluation_and_attention_require_same_completed_checkpoint(tmp_path):
    ns=comparison_paths(tmp_path,False,all_layers=True,tokenizer="sentencepiece")
    calls=[];checks=[]
    def run(name,module,*args,**kw):validate_command(module,args);calls.append((name,list(args)))
    def execute(*args,**kw):
        validate_command(args[2],args[3:]);checks.append(args)
        return dict(checkpoint_hash="a"*64)
    ns.update(RUN_TRANSFER=True,RUN_MONOTONIC=False,RUN_TEACHER_ARTIFACTS=True,PYTHON="python",json=json,
        MANIFEST=tmp_path/"manifest.jsonl",execute_json=execute,run=run)
    for path in (ns["T2TT_DEV_OUTPUT"],ns["T2TT_EXTERNAL_OUTPUT"]):write_json(path,dict(checkpoint_hash="a"*64))
    exec(cell("benchmark_comparison-24"),ns)
    assert checks[0][2]=="alignment_transfer_s2ut.spm_runtime"
    assert calls[0][1][calls[0][1].index("--checkpoint")+1]==ns["T2TT_CHECKPOINT"]
    assert calls[0][0]=="t2tt-attention-spm-200k-all-layers"
