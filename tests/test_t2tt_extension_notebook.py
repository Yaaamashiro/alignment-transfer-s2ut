"""Execute actual notebook cells with parsed commands and both teacher choices."""
import json
from pathlib import Path

import pytest

from alignment_transfer_s2ut.io import write_json, file_hash
from test_notebook_commands import comparison_paths, validate_command


def cell(identity):
    path = Path(__file__).resolve().parents[1] / "notebooks/colab/benchmark_comparison.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "".join(next(c for c in notebook["cells"] if c.get("id") == identity)["source"])


@pytest.mark.parametrize("extended", [False, True])
def test_active_best_is_used_by_dev_without_overwriting_20k(tmp_path, extended):
    calls = []
    values = comparison_paths(tmp_path, extended)
    old = tmp_path / "text-evaluation/dev-text.json"
    write_json(old, dict(retain="20k result"))
    before = file_hash(old)
    def run(name, module, *args, **kwargs):
        validate_command(module, args)
        calls.append((name, list(args)))
        write_json(Path(args[args.index("--output") + 1]), dict(text_bleu=1., checkpoint_step=123), overwrite=True)
    def execute(*args, **kwargs):
        validate_command(args[2], args[3:])
        return dict(completed_updates=200000)
    values.update(RUN_TRANSFER=True, RUN_MONOTONIC=False, PYTHON="python", PERSISTENT=tmp_path / "drive",
        MANIFEST=tmp_path / "manifest.jsonl", json=json, run=run, execute_json=execute)
    exec(cell("t2tt-dev-evaluate"), values)
    name, args = calls[0]
    assert args[args.index("--checkpoint") + 1] == values["T2TT_CHECKPOINT"]
    assert name == ("t2tt-dev-200k" if extended else "t2tt-dev")
    assert (file_hash(old) == before) == extended


def test_extension_cell_never_invokes_normal_20k_training(tmp_path):
    calls = []
    values = comparison_paths(tmp_path, True)
    values.update(RUN_T2TT_EXTENSION=True, RUN_TRANSFER=False, RUN_MONOTONIC=False, PYTHON="python",
        PERSISTENT=tmp_path / "drive", MANIFEST=tmp_path / "manifest.jsonl", BACKBONE_CONFIG=tmp_path / "backbone.json",
        T2TT_SNAPSHOT_UPDATES=10000, T2TT_KEEP_LAST=2, LOGS=tmp_path / "logs",
        run=lambda *a, **kw: pytest.fail("20k stage was started"),
        log_event=lambda *a, **kw: None, execute=lambda *a, **kw: calls.append(a))
    exec(cell("benchmark_comparison-22"), values)
    exec(cell("t2tt-extension-run"), values)
    assert len(calls) == 1
    command = list(calls[0])
    pivot = command.index("--")
    validate_command(command[2], command[3:])
    assert command[pivot + 3] == "alignment_transfer_s2ut.t2tt_extension"
    validate_command(command[pivot + 3], command[pivot + 4:])
    assert command[command.index("--source-work") + 1] == str(tmp_path / "t2tt")
    assert command[command.index("--work") + 1] == str(tmp_path / "t2tt-200k")
    assert command[command.index("--source-backup") + 1] != command[command.index("--backup") + 1]


def test_attention_requires_review_and_completed_matching_evaluations(tmp_path):
    values = comparison_paths(tmp_path, True)
    calls = []
    def run(name, module, *args, **kwargs):
        validate_command(module, args)
        calls.append((name, list(args)))
    def execute(*args, **kw):
        validate_command(args[2], args[3:])
        return dict(checkpoint_hash="a" * 64)
    values.update(RUN_TEACHER_ARTIFACTS=False, RUN_TRANSFER=True, RUN_MONOTONIC=False,
        MANIFEST=tmp_path / "manifest.jsonl", PYTHON="python", json=json, execute_json=execute, run=run)
    with pytest.raises(RuntimeError, match="RUN_TEACHER_ARTIFACTS"):
        exec(cell("benchmark_comparison-24"), values)
    assert not calls
    values["RUN_TEACHER_ARTIFACTS"] = True
    with pytest.raises(RuntimeError, match="評価を完了"):
        exec(cell("benchmark_comparison-24"), values)
    for output in (values["T2TT_DEV_OUTPUT"], values["T2TT_EXTERNAL_OUTPUT"]):
        write_json(output, dict(checkpoint_hash="a" * 64))
    exec(cell("benchmark_comparison-24"), values)
    assert [name for name, _ in calls] == ["t2tt-attention-200k", "speech-text-alignment-200k"]
    attention = calls[0][1]
    assert attention[attention.index("--checkpoint") + 1] == tmp_path / "t2tt-200k/checkpoint_best.pt"
    assert attention[attention.index("--output") + 1] == tmp_path / "t2tt-attention-200k"


@pytest.mark.parametrize("extended", [False, True])
def test_projected_teacher_and_student_never_mix_teacher_generations(tmp_path, extended):
    values = comparison_paths(tmp_path, extended)
    calls = []
    def run(name, module, *args, **kw):
        validate_command(module, args)
        calls.append((name, list(args)))
    def plan(*args, **kw):
        validate_command(args[2], args[3:])
        return dict(training_budget=400000, initialization="fresh", selection_policy={})
    values.update(RUN_TRANSFER=True, RUN_MONOTONIC=False, MANIFEST=tmp_path / "manifest.jsonl",
        BACKBONE_CONFIG=tmp_path / "backbone.json", BASELINE=tmp_path / "baseline.json",
        ANNEAL_STEPS=200000, PYTHON="python", run=run, execute_json=plan, LOGS=tmp_path / "logs", log_event=lambda *a: None)
    exec(cell("benchmark_comparison-26"), values)
    exec(cell("benchmark_comparison-28"), values)
    teacher_args = calls[0][1]
    assert teacher_args[teacher_args.index("--mt-attention") + 1] == values["T2TT_ATTENTION_ROOT"]
    assert teacher_args[teacher_args.index("--alignment") + 1] == values["CTC_ROOT"]
    transfer = values["TRANSFER_ARGS"]
    assert transfer[transfer.index("--teacher") + 1] == str(values["TEACHER_ROOT"])
    assert values["TRANSFER"].name == ("annealed-t2tt-200k" if extended else "annealed-t2tt")


def test_checkpoint_reference_is_centralized_and_old_stage_defaults_remain_explicit():
    path = Path(__file__).resolve().parents[1] / "notebooks/colab/benchmark_comparison.ipynb"
    source = "\n".join("".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code")
    assert "t2tt/checkpoint_best.pt" not in source
    assert "RUN_T2TT_EXTENSION = True" in source and "USE_EXTENDED_T2TT = True" in source
    for identity in ("t2tt-dev-evaluate", "external-text-evaluate", "benchmark_comparison-24"):
        assert "T2TT_CHECKPOINT" in cell(identity)


def test_monotonic_only_skips_extended_teacher_gate_and_review(tmp_path):
    values = comparison_paths(tmp_path, True)
    notices = []
    values.update(RUN_MONOTONIC=True, RUN_TRANSFER=False, RUN_TEACHER_ARTIFACTS=False,
        run=lambda *a, **kw: pytest.fail("Monotonic-only run touched teacher artifacts"),
        execute_json=lambda *a, **kw: pytest.fail("Monotonic-only run checked extended teacher"),
        LOGS=tmp_path / "logs", log_event=lambda *a: notices.append(a))
    exec(cell("benchmark_comparison-24"), values)
    exec(cell("extension-review"), values)
    assert any(a[:2] == ("t2tt-learning-review", "skipped") for a in notices)


def test_evaluation_only_of_extended_teacher_does_not_run_base_training(tmp_path):
    values = comparison_paths(tmp_path, True)
    values.update(RUN_T2TT_EXTENSION=False, RUN_TRANSFER=False, RUN_MONOTONIC=False,
        LOGS=tmp_path / "logs", log_event=lambda *a: None,
        run=lambda *a, **kw: pytest.fail("evaluation-only mode started source training"))
    exec(cell("benchmark_comparison-22"), values)
