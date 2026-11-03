import yaml
import pytest
from alignment_transfer_s2ut import experiment_conditions as c
from alignment_transfer_s2ut.alignment.losses import METHODS, METHOD_LABELS


def test_canonical_conditions_and_public_labels():
    conditions=c.load_conditions()
    assert tuple(conditions)==METHODS
    assert conditions["annealed_distillation"]["anneal_fraction"]==.8
    assert list(METHOD_LABELS.values())==["Baseline","KL-based Attention Distillation",
        "Annealed KL-based Attention Distillation","Attention-level Knowledge Distillation"]


def test_packaged_config_fallback_does_not_use_a_python_default(tmp_path,monkeypatch):
    conditions=c.load_conditions();conditions["annealed_distillation"]["anneal_fraction"]=.7
    package=tmp_path/"installed/site-packages/alignment_transfer_s2ut"
    package.mkdir(parents=True)
    (package/"conditions.yaml").write_text(yaml.safe_dump(conditions),encoding="utf-8")
    monkeypatch.setattr(c,"__file__",str(package/"experiment_conditions.py"))
    assert c.load_conditions()["annealed_distillation"]["anneal_fraction"]==.7


def test_invalid_yaml_schedule_fails_before_training(tmp_path):
    conditions=c.load_conditions();conditions["annealed_distillation"]["anneal_schedule"]="linear"
    path=tmp_path/"conditions.yaml";path.write_text(yaml.safe_dump(conditions),encoding="utf-8")
    with pytest.raises(ValueError,match="anneal_schedule"):
        c.load_conditions(path)
