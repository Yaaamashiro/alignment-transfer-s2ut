"""Pilot preparation never needs the real corpus, CUDA or model downloads."""
import ast
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import soundfile as sf
from alignment_transfer_s2ut.pilot import prepare, commands, run_logged, report, CONDITIONS
from alignment_transfer_s2ut.io import read_json, read_jsonl, write_json, write_jsonl, file_hash
from alignment_transfer_s2ut.data.manifest import load_manifest
from alignment_transfer_s2ut.cli import parser_for


def _source(root, corpus=False):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, split in enumerate(["train"] * 5 + ["dev"] * 3 + ["test"]):
        row = dict(pair_id=str(index), split=split, ja_text=f"日本語{index}", en_text=f"English {index}")
        for language in ("ja", "en"):
            filename = f"{index}-{language}.wav"
            path = root / ("production/audio_16k" if corpus else "audio") / filename
            if split != "test":
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, .1 * np.sin(2 * np.pi * (200 + index * 15) * np.arange(800) / 16000), 16000)
            # Excluded test audio intentionally does not exist.
            row[language + ("_wav_16k" if corpus else "_audio")] = (
                "/old-colab/audio_16k/" + filename if corpus else "audio/" + filename)
        rows.append(row)
    source = root / "accepted.jsonl"
    write_jsonl(source, rows)
    return source


def test_pilot_selection_preserves_splits_resume_and_original_paths(tmp_path):
    source = _source(tmp_path / "source")
    before = file_hash(source)
    result = prepare(source, tmp_path / "pilot", train_limit=2, dev_limit=2, seed=13)
    selected = load_manifest(tmp_path / "pilot/manifest.jsonl")
    assert result["selected_counts"] == {"train": 2, "dev": 2}
    assert result["population_counts"]["test"] == 1 and not result["test_used"]
    assert {row["split"] for row in selected} == {"train", "dev"}
    assert all(Path(row["ja_audio"]).is_relative_to(tmp_path / "source") for row in selected)
    assert all("ja_sha256" in row and "en_sha256" in row for row in selected)
    assert prepare(source, tmp_path / "pilot", train_limit=2, dev_limit=2, seed=13, resume=True) == result
    again = prepare(source, tmp_path / "other", train_limit=2, dev_limit=2, seed=13)
    assert again["selection_contract"] == result["selection_contract"]
    assert file_hash(source) == before
    with pytest.raises(ValueError, match="selection changed"):
        prepare(source, tmp_path / "pilot", train_limit=3, dev_limit=2, seed=13, resume=True)


def test_default_dev_limit_matches_function_cli_and_notebook(tmp_path, monkeypatch):
    from alignment_transfer_s2ut import pilot
    source = _source(tmp_path / "source")
    result = prepare(source, tmp_path / "pilot")
    assert result["selection_contract"]["dev_limit"] == 200
    monkeypatch.setattr(sys, "argv", ["pilot", "prepare", "--input", str(source),
                                    "--output", str(tmp_path / "cli-pilot")])
    pilot.main()
    assert read_json(tmp_path / "cli-pilot/selection.json")["selection_contract"]["dev_limit"] == 200


def test_pilot_corpus_aliases_relocation_and_excluded_leakage(tmp_path):
    source = _source(tmp_path / "corpus", corpus=True)
    prepare(source, tmp_path / "pilot", corpus_root=source.parent, train_limit=2, dev_limit=1)
    assert len(load_manifest(tmp_path / "pilot/manifest.jsonl")) == 3
    with pytest.raises(ValueError, match="separate"):
        prepare(source, source.parent / "derived", corpus_root=source.parent)
    rows = read_jsonl(source)
    rows[-1]["ja_text"], rows[-1]["en_text"] = rows[0]["ja_text"], rows[0]["en_text"]
    write_jsonl(source, rows, overwrite=True)
    with pytest.raises(ValueError, match="leakage"):
        prepare(source, tmp_path / "bad", corpus_root=source.parent, train_limit=1, dev_limit=1)


def test_pilot_commands_have_three_strong_conditions_and_teacher_free_tail():
    args = Namespace(manifest="manifest.jsonl", config="config.yaml", units="units", teacher="teacher", output="runs",
                     max_updates=10000, anneal_steps=5000, seed=1, batch_size=8, update_frequency=4,
                     max_tokens=20000, warmup_updates=1000, interval=500)
    plan = commands(args)
    assert list(plan) == [name for name, _, _ in CONDITIONS]
    for name, command in plan.items():
        parsed = parser_for("train_s2ut").parse_args(command[3:])
        assert parsed.recipe == "strong" and parsed.max_updates == 10000 and parsed.anneal_steps == 5000
        assert parsed.resume_if_exists and parsed.device == "cuda"
        assert parsed.loader_workers == 2 and parsed.prefetch_batches == 2 and parsed.pin_memory
        assert parsed.teacher == ("teacher" if name == "annealed_t2tt" else None)
    args.anneal_steps = args.max_updates
    with pytest.raises(ValueError, match="teacher-free tail"):
        commands(args)


def test_pilot_runner_records_failures_and_preserves_each_attempt(tmp_path):
    result = run_logged("condition", [sys.executable, "-c", "print('complete')"], tmp_path)
    assert result["status"] == "succeeded" and result["elapsed_seconds"] > 0
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_logged("condition", [sys.executable, "-c", "import sys; print('failed stage'); sys.exit(7)"], tmp_path)
    assert error.value.returncode == 7
    records = [read_json(path) for path in tmp_path.glob("*.json")]
    assert len(records) == 2
    failed = next(row for row in records if row["returncode"] == 7)
    assert "failed stage" in Path(failed["log"]).read_text()
    assert failed["gpu_csv"] is None


def test_fast_failure_displays_preflight_details_despite_progress_throttling(tmp_path, monkeypatch, capsys):
    from alignment_transfer_s2ut import pilot
    monkeypatch.setattr(pilot.time, "monotonic", lambda: 1.)
    script = "import json,sys; print(json.dumps({'checks': {'dependencies': {'ok': False, 'detail': 'version mismatch'}}}, indent=2)); sys.exit(1)"
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_logged("environment", [sys.executable, "-c", script], tmp_path)
    assert error.value.returncode == 1
    visible = capsys.readouterr().out
    assert '"dependencies"' in visible and '"ok": false' in visible
    assert "version mismatch" in visible and "full log:" in visible
    record = next(read_json(path) for path in tmp_path.glob("*.json"))
    assert "version mismatch" in Path(record["log"]).read_text(encoding="utf-8")


def test_pilot_report_handles_resumed_steps_and_rejects_mixed_evaluation(tmp_path):
    logs = tmp_path / "execution"
    logs.mkdir()
    for index, (name, method, prior) in enumerate(CONDITIONS):
        folder = tmp_path / "runs" / name
        folder.mkdir(parents=True)
        write_json(folder / "experiment.json", dict(method=method, backend="fairseq", dev_hash="common", transfer=dict(prior_type=prior),
            teacher_signature={"checkpoint": "teacher"} if name == "annealed_t2tt" else None,
            planned_max_updates=2, recipe={"speech_recipe": {"name": "strong"}}))
        write_jsonl(folder / "metrics.jsonl", [
            {"step": 1, "train/main_loss": 8., "dev/main_loss": 9., "guidance/beta": 1.},
            {"step": 2, "train/main_loss": 3., "dev/main_loss": 3. + index, "guidance/beta": 0.},
            {"step": 1, "train/main_loss": 4., "dev/main_loss": 4. + index, "guidance/beta": 1.}])
        write_json(logs / (name + ".json"), dict(name=name, elapsed_seconds=10, status="succeeded"))
        write_json(folder / "dev-metrics.json", dict(split="dev", samples=2, backend="fairseq", vocoder_hash=None,
            unit_error_rate=.5, evaluation_contract={"manifest_hash": "common", "predictions_hash": name}))
    result = report(tmp_path / "runs", logs, tmp_path / "report.json")
    assert result["complete"] and result["fair_comparison"]["fair_comparison"]
    assert result["conditions"][0]["best_dev_step"] == 2
    assert result["conditions"][0]["recorded_seconds"] == 10
    teacher_run = tmp_path / "runs/annealed_t2tt/experiment.json"
    experiment = read_json(teacher_run)
    experiment.update(backend="fairseq", dev_hash="common", teacher_signature={"mt_checkpoint_hash": "teacher"})
    write_json(teacher_run, experiment, overwrite=True)
    teacher_metrics = dict(split="dev", backend="fairseq", checkpoint_hash="teacher", text_bleu=12.,
                           evaluation_contract={"manifest_hash": "common"})
    write_json(tmp_path / "t2tt-dev.json", teacher_metrics)
    attached = report(tmp_path / "runs", logs, tmp_path / "report.json", tmp_path / "t2tt-dev.json")
    assert attached["t2tt_dev_evaluation"]["text_bleu"] == 12.
    for replacement in ({"checkpoint_hash": "different-teacher"}, {"split": "test"},
                        {"evaluation_contract": {"manifest_hash": "different-dev"}}):
        write_json(tmp_path / "t2tt-dev.json", {**teacher_metrics, **replacement}, overwrite=True)
        with pytest.raises(ValueError, match="teacher checkpoint and dev"):
            report(tmp_path / "runs", logs, tmp_path / "report.json", tmp_path / "t2tt-dev.json")
    path = tmp_path / "runs/annealed_t2tt/dev-metrics.json"
    wrong = read_json(path)
    wrong["evaluation_contract"]["manifest_hash"] = "different"
    write_json(path, wrong, overwrite=True)
    with pytest.raises(ValueError, match="same dev"):
        report(tmp_path / "runs", logs, tmp_path / "bad-report.json")


def test_pilot_notebook_syntax_and_separate_update_budgets(tmp_path):
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/colab/pilot.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(source)
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and (isinstance(node.value, ast.Constant) or
                      isinstance(node.value, ast.List) and all(isinstance(item, ast.Constant) for item in node.value.elts))}
    assert constants["T2TT_UPDATES"] == 20000 and constants["S2UT_UPDATES"] == 10000
    assert constants["ANNEAL_STEPS"] == 5000
    assert constants["DEV_LIMIT"] == 200
    assert constants["CONDITIONS"] == [name for name, _, _ in CONDITIONS]
    assert "check=True" in source and "alignment-audit-" in source
    assert "alignment_transfer_s2ut.evaluate_t2tt" in source and "--t2tt-evaluation" in source
    # Exercise the notebook's JSONL-writing expression, catching escaped-newline errors.
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
                node.func.value.id == "tiny_manifest" and node.func.attr == "write_text")
    target = tmp_path / "integration.jsonl"
    eval(compile(ast.Expression(call), "pilot-integration-cell", "eval"),
         {"json": json, "tiny_manifest": target, "tiny": [{"pair_id": "a"}, {"pair_id": "b"}]})
    assert read_jsonl(target) == [{"pair_id": "a"}, {"pair_id": "b"}]
