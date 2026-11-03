"""Execute the current four-condition Colab commands through real CLI parsers."""
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
    ns=comparison_paths(tmp_path,extended,all_layers=True,RUN_DISTILLATION=True,RUN_ANNEALED_DISTILLATION=True,RUN_MASK_DISTILLATION=True)
    calls=[];plans=[];tables=[]
    def run(name,module,*args,**kwargs):
        validate_command(module,args);calls.append((name,module,list(map(str,args))))
        if module=="alignment_transfer_s2ut.evaluate":
            write_json(Path(args[args.index('--output')+1]),dict(unit_error_rate=.2,asr_bleu=None))
    def plan(*args,**kw):
        validate_command(args[2],args[3:]);plans.append(args)
        return dict(training_budget=400000,initialization="fresh",selection_policy={},transfer=dict(calibration_hash="fixed"))
    ns.update(BASELINE=tmp_path/'baseline.json',MANIFEST=tmp_path/'manifest.jsonl',PYTHON='python',
        run=run,execute_json=plan,json=json,LOGS=tmp_path/'execution',log_event=lambda *a,**kw:None,MASK_TOP_P=.8)
    write_json(ns['CALIBRATION'],dict(fixture=True))
    exec(cell('benchmark_comparison-28'),ns)
    assert len(calls)==len(plans)==3
    methods=[a[a.index('--method')+1] for _,_,a in calls]
    assert methods==['distillation','annealed_distillation','mask_distillation']
    for _,_,a in calls:
        assert '--all-layers' in a and '--max-updates' not in a and '--beta' not in a
        assert a[a.index('--teacher')+1]==str(ns['TEACHER_ROOT'])
        assert a[a.index('--calibration')+1]==str(ns['CALIBRATION'])
    for method in methods:
        folder=ns['RESEARCH_ROOT']/method/'checkpoints';folder.mkdir(parents=True);(folder/'checkpoint_best.pt').touch()
    display=ModuleType('IPython.display');display.Markdown=str;display.display=tables.append
    monkeypatch.setitem(sys.modules,'IPython.display',display)
    ns.update(GENERATE_WAVEFORMS=False,ASR_BLEU=False)
    calls.clear();exec(cell('benchmark_comparison-30'),ns)
    assert len([c for c in calls if c[2][0]=='infer'])==4
    labels=('Baseline','KL-based Attention Distillation','Annealed KL-based Attention Distillation','Attention-level Knowledge Distillation')
    assert [tables[0].index('| '+label+' |') for label in labels]==sorted(tables[0].index('| '+label+' |') for label in labels)


def test_research_switches_default_off_and_explicit_selection_required(tmp_path):
    settings=cell('benchmark_comparison-06');ns={};exec(settings,ns)
    assert all(ns[k] is False for k in ('RUN_DISTILLATION','RUN_ANNEALED_DISTILLATION','RUN_MASK_DISTILLATION','RUN_ALIGNMENT_CALIBRATION','RUN_MASK_TOPP_DIAGNOSTICS','RUN_T2TT_EXTENSION','RUN_T2TT_SENTENCEPIECE'))
    assert ns['MASK_TOP_P'] is None and not ns['NEED_T2TT']
    ns=comparison_paths(tmp_path,all_layers=True);skips=[]
    ns.update(log_event=lambda *a:skips.append(a),LOGS=tmp_path/'logs')
    exec(cell('benchmark_comparison-28'),ns)
    assert len(skips)==3
    ns['RESEARCH_CONDITIONS']=[('mask_distillation','Attention-level Knowledge Distillation',True)]
    ns['MASK_TOP_P']=None
    with pytest.raises(ValueError,match='calibration'):exec(cell('benchmark_comparison-28'),ns)


def test_all_layer_paths_and_new_stages_are_separate_from_old_runs(tmp_path):
    ns=comparison_paths(tmp_path,True,True)
    assert ns['RESEARCH_ROOT'].name.startswith('research-v3')
    assert '-all-layers' in str(ns['TEACHER_ROOT'])
    assert ns['CALIBRATION'].parent!=ns['TOPP_REPORT'].parent
    source='\n'.join(cell(name) for name in ('benchmark_comparison-06','t2tt-active-paths','benchmark_comparison-28','benchmark_comparison-30'))
    for removed in ('RUN_TRANSFER','RUN_GUIDANCE','RUN_MONOTONIC','--beta','--lambda-align','annealed_guidance'):
        assert removed not in source


def test_top_p_review_calibration_and_saved_stage_restoration(tmp_path,monkeypatch):
    ns=comparison_paths(tmp_path,all_layers=True)
    calls=[]
    import types
    pandas=types.ModuleType('pandas')
    class Frame:
        def __init__(self,*args):pass
        @property
        def T(self):return self
    pandas.DataFrame=Frame
    monkeypatch.setitem(sys.modules,'pandas',pandas)
    display=types.ModuleType('IPython.display');display.display=lambda *a:None;display.Image=lambda **kw:kw
    monkeypatch.setitem(sys.modules,'IPython.display',display)
    plt=types.ModuleType('matplotlib.pyplot')
    for name in ('plot','xlabel','ylabel','legend','savefig','close'):
        setattr(plt,name,lambda *a,**kw:None)
    monkeypatch.setitem(sys.modules,'matplotlib.pyplot',plt)
    matplotlib=types.ModuleType('matplotlib');matplotlib.pyplot=plt;monkeypatch.setitem(sys.modules,'matplotlib',matplotlib)
    def run(name,module,*args,**kw):
        validate_command(module,args);calls.append(list(map(str,args)))
        if args[0]=='top-p':
            write_json(ns['TOPP_REPORT'],dict(candidates={str(p):dict(mean=.4,median=.4,p25=.2,p75=.5,p90=.6,
                valid_rows=10,selected_frames=20,histogram=[1]*10,by_layer={'0':dict(mean=.4)},
                by_source_length={'<=100':dict(mean=.4)}) for p in (.8,.9,.95)}),overwrite=True)
        else:
            write_json(ns['CALIBRATION'],dict(lambda_kl=1.,lambda_mask=2.,tau_kl=.3,tau_mask=.2,selected_top_p=.9,optimizer_updates=0),overwrite=True)
    ns.update(BASELINE=tmp_path/'baseline.json',MANIFEST=tmp_path/'manifest.jsonl',CALIBRATION_BATCHES=16,
        RUN_MASK_TOPP_DIAGNOSTICS=True,RUN_ALIGNMENT_CALIBRATION=True,MASK_TOP_P=None,run=run,json=json,
        LOGS=tmp_path/'logs',log_event=lambda *a:None)
    with pytest.raises(ValueError,match='MASK_TOP_P'):exec(cell('alignment-calibration'),ns)
    assert [a[0] for a in calls]==['top-p']
    ns.update(RUN_MASK_TOPP_DIAGNOSTICS=False,MASK_TOP_P=.9)
    exec(cell('alignment-calibration'),ns)
    assert calls[-1][0]=='calibrate' and calls[-1][calls[-1].index('--mask-top-p')+1]=='0.9'
    # A fresh VM can restore both verified stage commands with run switches off.
    for name in ('top-p','calibration'):
        marker=ns['PERSISTENT']/'stages'/('research-v3-'+name+ns['SCOPE_SUFFIX'])/'pack/snapshot.json'
        marker.parent.mkdir(parents=True);marker.touch()
    calls.clear();ns['RUN_ALIGNMENT_CALIBRATION']=False
    exec(cell('alignment-calibration'),ns)
    assert [a[0] for a in calls]==['top-p','calibrate']
