"""Review regressions: layer selection and explicitly controlled scope ablations."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from alignment_transfer_s2ut.io import ArtifactStore,digest,write_json,read_json
from alignment_transfer_s2ut.training import prior_builder
from alignment_transfer_s2ut.compare_experiments import CONTROLLED,compare_experiments
from test_all_layers import tiny


def test_single_prior_selects_matching_layer_from_full_teacher_without_writes(tmp_path):
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
    before=teacher.hash("p")
    args=([row],teacher,units,tiny(),"manifest","smoke","cpu")
    full=prior_builder(*args,manifest_hashes=True,all_layers=True)(torch.tensor([2]),3)
    for layer in (0,1,-1,-2):
        single=prior_builder(*args,manifest_hashes=True,layer=layer)(torch.tensor([2]),3)
        assert torch.equal(single,full[:,layer])
    assert not torch.equal(full[:,0],full[:,1])
    with pytest.raises(ValueError,match="out of range"):
        prior_builder(*args,manifest_hashes=True,layer=2)
    assert teacher.hash("p")==before


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
