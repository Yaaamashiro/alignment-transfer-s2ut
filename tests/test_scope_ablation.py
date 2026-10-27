"""Review regressions: layer selection and explicitly controlled scope ablations."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from alignment_transfer_s2ut.io import ArtifactStore,digest,write_json,read_json
from alignment_transfer_s2ut.training import prior_builder
from alignment_transfer_s2ut.compare_experiments import CONTROLLED,compare_experiments
from test_all_layers import tiny


@pytest.fixture
def teacher_inputs(tmp_path):
    units,teacher=ArtifactStore(tmp_path/"units"),ArtifactStore(tmp_path/"teacher")
    row=dict(pair_id="p",split="train",ja_text="あい",en_text="ab",ja_sha256="ja",en_sha256="en")
    units.put("p",{},dict(reduced_units=np.array([0,1]),unit_spans=np.array([[0.,.5],[.5,1.]])))
    spec=dict(all_layers=True,decoder_layers=2,layer_indices=[0,1],heads=None)
    meta=dict(spec,split="train",manifest_hash="manifest",ja_audio_hash="ja",en_audio_hash="en",
        s2ut_architecture_hash=digest(tiny().asdict()),backend="smoke",units_hash=units.hash("p"),
        text_hash=digest([row["ja_text"],row["en_text"]]),mt_checkpoint_hash="mt",tokenizer_hash="v",
        unit_config_hash="u",ctc_config_hash="c",attention_spec_hash=digest(spec))
    layers=np.array([np.eye(2),np.eye(2)[::-1]],dtype=np.float32)
    teacher.put("p",meta,dict(mt_attention=layers[-1],mt_attention_layers=layers,
        source_spans=np.array([[0.,.5],[.5,1.]]),target_spans=np.array([[0.,.5],[.5,1.]]),
        unit_spans=np.array([[0.,.5],[.5,1.]]),source_duration=np.array(1.)))
    return ([row],teacher,units,tiny(),"manifest","smoke","cpu")


def test_single_prior_selects_matching_layer_from_full_teacher_without_writes(teacher_inputs):
    args=teacher_inputs
    teacher=args[1]
    before=teacher.hash("p")
    full=prior_builder(*args,manifest_hashes=True,all_layers=True)(torch.tensor([2]),3)
    for layer in (0,1,-1,-2):
        single=prior_builder(*args,manifest_hashes=True,layer=layer)(torch.tensor([2]),3)
        assert torch.equal(single,full[:,layer])
    assert not torch.equal(full[:,0],full[:,1])
    with pytest.raises(ValueError,match="out of range"):
        prior_builder(*args,manifest_hashes=True,layer=2)
    assert teacher.hash("p")==before


@pytest.mark.parametrize("teacher_layer,student_layer",[(1,-1),(-1,1),(0,-2),(-2,0)])
def test_single_teacher_layer_aliases_are_accepted(teacher_inputs,teacher_layer,student_layer):
    from alignment_transfer_s2ut.alignment.layers import attention_spec
    teacher=teacher_inputs[1];meta,packet=teacher.get("p")
    for key in ("all_layers","layer_indices","decoder_layers"):meta.pop(key,None)
    packet.pop("mt_attention_layers")
    meta["layer"]=teacher_layer;meta["attention_spec_hash"]=digest(attention_spec(meta))
    teacher.put("p",meta,packet,overwrite=True)
    before=teacher.hash("p")
    assert prior_builder(*teacher_inputs,manifest_hashes=True,layer=student_layer)(torch.tensor([2]),3).shape==(1,3,2)
    with pytest.raises(ValueError,match="layer mismatch"):
        prior_builder(*teacher_inputs,manifest_hashes=True,layer=(student_layer+1)%2)
    assert teacher.hash("p")==before


@pytest.mark.parametrize("teacher_layer",[2,-3,True,"1"])
def test_invalid_recorded_teacher_layer_is_not_wrapped(teacher_inputs,teacher_layer):
    from alignment_transfer_s2ut.alignment.layers import attention_spec
    teacher=teacher_inputs[1];meta,packet=teacher.get("p")
    meta.update(all_layers=False,layer=teacher_layer)
    meta["attention_spec_hash"]=digest(attention_spec(meta))
    packet.pop("mt_attention_layers")
    teacher.put("p",meta,packet,overwrite=True)
    with pytest.raises(ValueError,match="teacher attention layer out of range"):
        prior_builder(*teacher_inputs,manifest_hashes=True)


@pytest.mark.parametrize("all_layers",[False,True])
@pytest.mark.parametrize("teacher_heads,student_heads,accepted",[
    (None,None,True),([0],[0],True),([1,0],[0,1],True),
    ([0],[1],False),(None,[0],False),([0],None,False),([],None,False)])
def test_teacher_student_head_correspondence(teacher_inputs,all_layers,teacher_heads,student_heads,accepted):
    from alignment_transfer_s2ut.alignment.layers import attention_spec
    teacher=teacher_inputs[1];meta,packet=teacher.get("p")
    meta.update(heads=teacher_heads,all_layers=all_layers)
    meta["attention_spec_hash"]=digest(attention_spec(meta))
    if not all_layers:packet.pop("mt_attention_layers")
    teacher.put("p",meta,packet,overwrite=True)
    if accepted:
        prior=prior_builder(*teacher_inputs,manifest_hashes=True,all_layers=all_layers,heads=student_heads)(torch.tensor([2]),3)
        assert prior.ndim==(4 if all_layers else 3)
    else:
        with pytest.raises(ValueError,match="head"):
            prior_builder(*teacher_inputs,manifest_hashes=True,all_layers=all_layers,heads=student_heads)


def test_legacy_unspecified_selection_keeps_only_documented_defaults(teacher_inputs):
    teacher=teacher_inputs[1];meta,packet=teacher.get("p")
    for key in ("all_layers","layer_indices","decoder_layers","heads","attention_spec_hash"):meta.pop(key,None)
    packet.pop("mt_attention_layers")
    teacher.put("p",meta,packet,overwrite=True)
    before=teacher.hash("p")
    prior_builder(*teacher_inputs,manifest_hashes=True)
    with pytest.raises(ValueError,match="layer mismatch"):
        prior_builder(*teacher_inputs,manifest_hashes=True,layer=0)
    with pytest.raises(ValueError,match="heads mismatch"):
        prior_builder(*teacher_inputs,manifest_hashes=True,heads=[0])
    assert teacher.hash("p")==before


def test_audit_layer_cli_is_explicit_and_exclusive():
    from test_notebook_commands import validate_command
    command=[]  # Required paths are parsed without reading them.
    for key in ("manifest","teacher","units","config","output"):command += ["--"+key,"unused"]
    validate_command("alignment_transfer_s2ut.diagnose_alignment",[*command,"--layer","-2"])
    with pytest.raises(SystemExit):
        validate_command("alignment_transfer_s2ut.diagnose_alignment",[*command,"--all-layers","--layer","-1"])


@pytest.fixture
def experiments(tmp_path):
    common={key:key for key in CONTROLLED}
    common.update(backbone=dict(decoder_layers=2),teacher_signature=dict(checkpoint="same full teacher"))
    transfer=dict(all_layers=False,layer=-1,heads=None,source_mapping="frontend",prior_smoothing=.05,
        prior_type="t2tt",lambda_align=1.,beta=1.,beta_start=1.,beta_end=0.,anneal_steps=1,monotonic_sigma=.1)
    roots=[tmp_path/name for name in ("baseline","single","all")]
    for i,root in enumerate(roots):
        write_json(root/"experiment.json",dict(deepcopy(common),method="baseline" if i==0 else "distillation",
            transfer={} if i==0 else dict(transfer,all_layers=i==2)))
    return roots


def test_scope_ablation_is_explicit_and_records_the_variable(experiments):
    strict=compare_experiments(experiments)
    assert not strict["fair_comparison"] and "transfer/all_layers" in strict["differences"]
    result=compare_experiments(experiments,allow_scope_ablation=True)
    assert result["fair_comparison"]
    assert result["independent_variables"]["transfer_scope"]==[dict(all_layers=False,layer=1),dict(all_layers=True,layer=None)]


@pytest.mark.parametrize("field",["recipe","initialization_hash","planned_max_updates","teacher_signature","heads","prior_smoothing","lambda_align","method"])
def test_ablation_does_not_relax_other_controls(experiments,field):
    path=experiments[-1]/"experiment.json"; doc=read_json(path)
    if field in {"heads","prior_smoothing","lambda_align"}:
        doc["transfer"][field]=[0] if field=="heads" else .3
    else:
        doc[field]="guidance" if field=="method" else "changed"
    write_json(path,doc,overwrite=True)
    assert not compare_experiments(experiments,allow_scope_ablation=True)["fair_comparison"]


def test_distinct_single_layers_are_explicit_scope_conditions(experiments):
    for root,layer in zip(experiments[1:],(0,1)):
        path=root/"experiment.json";doc=read_json(path);doc["transfer"].update(all_layers=False,layer=layer)
        write_json(path,doc,overwrite=True)
    assert not compare_experiments(experiments)["fair_comparison"]
    result=compare_experiments(experiments,allow_scope_ablation=True)
    assert result["fair_comparison"] and [r["layer"] for r in result["conditions"][1:]]==[0,1]


def test_scope_ablation_cli_records_explicit_option(experiments,tmp_path,monkeypatch):
    import sys
    from alignment_transfer_s2ut.compare_experiments import main
    output=tmp_path/"comparison.json"
    monkeypatch.setattr(sys,"argv",["compare_experiments",*map(str,experiments),"--allow-scope-ablation","--output",str(output)])
    main()
    assert read_json(output)["fair_comparison"] and read_json(output)["independent_variables"]
