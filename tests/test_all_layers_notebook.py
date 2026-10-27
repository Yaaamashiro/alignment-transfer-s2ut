"""Execute generated Colab commands; no real GPU/Drive work in notebook tests."""
import json
from pathlib import Path
from types import ModuleType
import sys
import pytest
from test_notebook_commands import comparison_paths,validate_command
from test_t2tt_extension_notebook import cell
from alignment_transfer_s2ut.io import write_json


@pytest.mark.parametrize("extended",[False,True])
def test_four_primary_conditions_share_inputs_and_evaluation(tmp_path,monkeypatch,extended):
    ns=comparison_paths(tmp_path,extended,all_layers=True)
    calls=[]; plans=[]; tables=[]
    def run(name,module,*args,**kwargs):
        validate_command(module,args); calls.append((name,module,list(map(str,args))))
        if module=="alignment_transfer_s2ut.evaluate":
            write_json(Path(args[args.index('--output')+1]),dict(unit_error_rate=.2,asr_bleu=None))
    def plan(*args,**kw):
        validate_command(args[2],args[3:]); plans.append(args)
        return dict(training_budget=400000,initialization="fresh",selection_policy={})
    ns.update(RUN_TRANSFER=True,RUN_DISTILLATION=True,RUN_GUIDANCE=True,RUN_MONOTONIC=False,
        RUN_TEACHER_ARTIFACTS=True,BASELINE=tmp_path/'baseline.json',MANIFEST=tmp_path/'manifest.jsonl',
        BACKBONE_CONFIG=tmp_path/'backbone.json',PYTHON='python',ANNEAL_STEPS=200000,run=run,execute_json=plan,
        json=json,LOGS=tmp_path/'execution',log_event=lambda *a,**kw:None)
    if extended:
        for p in (ns['T2TT_DEV_OUTPUT'],ns['T2TT_EXTERNAL_OUTPUT']): write_json(p,dict(checkpoint_hash='best'))
        def execute(*args,**kw): return dict(checkpoint_hash='best')
        ns['execute_json']=execute
    exec(cell('benchmark_comparison-24'),ns)
    assert '--all-layers' in calls[0][2]
    ns['execute_json']=plan; calls.clear()
    exec(cell('benchmark_comparison-28'),ns)
    assert len(calls)==3 and len(plans)==3
    assert {a[a.index('--method')+1] for _,_,a in calls}=={'distillation','annealed_guidance','guidance'}
    for _,_,a in calls:
        assert '--all-layers' in a and '--max-updates' not in a
        assert a[a.index('--teacher')+1]==str(ns['TEACHER_ROOT'])
    for folder in (ns['TRANSFER'],ns['DISTILLATION'],ns['GUIDANCE']):
        (folder/'checkpoints').mkdir(parents=True); (folder/'checkpoints/checkpoint_best.pt').touch()
    display=ModuleType('IPython.display'); display.Markdown=str; display.display=tables.append
    monkeypatch.setitem(sys.modules,'IPython.display',display)
    ns.update(GENERATE_WAVEFORMS=False,ASR_BLEU=False)
    calls.clear(); exec(cell('benchmark_comparison-30'),ns)
    assert len([c for c in calls if c[1:] and c[2][0]=='infer'])==4
    for label in ('S2UT Baseline','All-layer Distillation','All-layer Annealed Guidance','All-layer Guidance'):
        assert label in tables[0]


def test_all_layer_paths_preserve_single_layer_and_t2tt_evaluations(tmp_path):
    single=comparison_paths(tmp_path,True,False); layered=comparison_paths(tmp_path,True,True)
    for k in ('T2TT_ATTENTION_ROOT','TEACHER_ROOT','TEACHER_AUDIT_ROOT','TRANSFER_NAME'):
        assert single[k]!=layered[k] and '-all-layers' in str(layered[k])
    for k in ('T2TT_CHECKPOINT','T2TT_DEV_OUTPUT','T2TT_EXTERNAL_OUTPUT','CTC_ROOT'):
        assert single[k]==layered[k]
