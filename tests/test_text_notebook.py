import ast
import json
from pathlib import Path
import pytest
from alignment_transfer_s2ut.cli import parser_for
from test_notebook_commands import validate_command


def cells():
    path=Path(__file__).resolve().parents[1]/"notebooks/colab/jparacrawl_t2tt.ipynb"
    return {c["id"]:''.join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]}


@pytest.mark.parametrize("tokenizer",["character","sentencepiece"])
def test_text_notebook_all_commands_use_audio_free_clis_and_independent_outputs(tmp_path,tokenizer):
    source=cells();settings=source["text-settings"].replace("/content/drive/MyDrive/SET_TEXT_MANIFEST.jsonl",(tmp_path/"external.jsonl").as_posix())
    ns={};exec(settings,ns)
    # Re-run with the real form variable changed, retaining all path expressions.
    settings=settings.replace("TOKENIZER = 'sentencepiece'",f"TOKENIZER = '{tokenizer}'")
    exec(settings,ns)
    for key in ("ROOT","PERSISTENT","WORK","BACKUP","EVAL_ROOT","EVAL_BACKUP","ATTENTION","ATTENTION_BACKUP","LOGS"):
        ns[key]=tmp_path/key.lower()
    ns.update(SOURCE_MANIFEST=str(tmp_path/"external.jsonl"),MANIFEST=tmp_path/"text-manifest.jsonl",CONFIG=tmp_path/"config.yaml",
        CHECKPOINT=tmp_path/"work/checkpoint_best.pt",DEV_METRICS=tmp_path/"eval/dev-text.json",RUN_ATTENTION=True)
    calls=[]
    def execute(*args,**kw):
        calls.append(list(map(str,args)))
        if len(args)>2 and args[1]=="-m":validate_command(args[2],args[3:])
    def execute_json(*args,**kw):
        execute(*args,**kw)
        if args[2]=="alignment_transfer_s2ut.text_manifest":return dict(rows=3)
        if args[2]=="alignment_transfer_s2ut.text_t2tt_runtime":return dict(completed_updates=200000)
        return dict(checkpoint_restored=True)
    ns.update(execute=execute,execute_json=execute_json,log_event=lambda *a:None)
    exec(source["text-stage-input"],ns);exec(source["text-train"],ns)
    training=calls[-1];pivot=training.index("alignment_transfer_s2ut.text_t2tt_runtime")
    validate_command(training[pivot],training[pivot+1:])
    pivot=training.index("alignment_transfer_s2ut.train_t2tt")
    args=parser_for("train_t2tt").parse_args(training[pivot+1:])
    assert args.tokenizer==tokenizer and args.max_updates==200000
    assert args.batch_size==8 and args.update_frequency==4 and args.max_tokens==20000
    ns["DEV_METRICS"].parent.mkdir(parents=True);ns["DEV_METRICS"].write_text(json.dumps(dict(text_bleu=0.)),encoding="utf-8")
    exec(source["text-evaluate"],ns);exec(source["text-attention"],ns)
    assert not any("align_speech_text" in str(c) or "extract_units" in str(c) or "train_s2ut" in str(c) for c in calls)
    assert "--all-layers" in calls[-1]


def test_text_notebook_logging_matches_canonical_code():
    source=cells();root=Path(__file__).resolve().parents[1]
    canonical=ast.parse((root/"src/alignment_transfer_s2ut/process_logging.py").read_text(encoding="utf-8"))
    expected={n.name:ast.dump(n) for n in canonical.body if isinstance(n,ast.FunctionDef)}
    actual={n.name:ast.dump(n) for n in ast.parse(source["text-process-logging"]).body if isinstance(n,ast.FunctionDef)}
    assert all(actual[key]==value for key,value in expected.items())
    for identity,code in source.items():
        if identity=="text-process-logging" or identity.endswith(("guide","intro","resume")):continue
        tree=ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert not (isinstance(node.func.value,ast.Name) and node.func.value.id=="subprocess")
