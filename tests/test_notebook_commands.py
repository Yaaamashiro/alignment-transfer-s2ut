"""Check notebook commands against the real CLI parsers without running training."""
import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def validate_command(module, arguments):
    target = importlib.import_module(module)
    parse = argparse.ArgumentParser.parse_args
    class Parsed(Exception):
        pass
    def stop_after_parse(parser, args=None, namespace=None):
        parse(parser, args, namespace)
        raise Parsed
    with patch.object(argparse.ArgumentParser, "parse_args", stop_after_parse), \
            patch.object(sys, "argv", [module, *map(str, arguments)]):
        with pytest.raises(Parsed):
            if target.main.__module__ == "alignment_transfer_s2ut.cli":
                target.main(module.rsplit(".", 1)[-1])
            else:
                target.main()


def pipeline_cells():
    path = Path(__file__).resolve().parents[1] / "notebooks/colab/pipeline.ipynb"
    return ["".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code"]


def comparison_paths(root, extended=False):
    path = Path(__file__).resolve().parents[1] / "notebooks/colab/benchmark_comparison.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(next(c for c in notebook["cells"] if c.get("id") == "t2tt-active-paths")["source"])
    namespace = dict(Path=Path, ROOT=str(root), USE_EXTENDED_T2TT=extended,
                     T2TT_BASE_UPDATES=20000, T2TT_TARGET_UPDATES=200000)
    exec(source, namespace)
    return {k: v for k, v in namespace.items() if k != "__builtins__"}


def test_standalone_commands_parse_and_preserve_all_five_conditions():
    calls = []
    namespace = dict(ROOT="/tmp/output with spaces", CORPUS="/tmp/corpus with spaces",
        RUNS="/tmp/output with spaces/strong", PYTHON="python", UPDATES=2,
        ANNEAL_STEPS=10000, RECIPE="strong", execute=lambda *args: calls.append(list(map(str, args))))
    for source in pipeline_cells():
        if "execute(" in source and "git" not in source and "def execute(" not in source:
            exec(source, namespace)
    for command in calls:
        assert command[1] == "-m"
        validate_command(command[2], command[3:])
    training = [command for command in calls if command[2].endswith("train_s2ut")]
    assert len(training) == 5
    assert [command[command.index("--method") + 1] for command in training] == [
        "baseline", "distillation", "guidance", "annealed_guidance", "annealed_guidance"]
    assert "--teacher" not in training[0] and "--teacher" not in training[-1]
    assert all("--teacher" in command for command in training[1:-1])
    assert all(command[command.index("--trainer") + 1] == "legacy" for command in training)
    assert calls[0][calls[0].index("--corpus-root") + 1] == namespace["CORPUS"]


def test_standalone_commands_stop_on_failure_and_preserve_path_arguments():
    setup = next(source for source in pipeline_cells() if "def execute(" in source)
    helper = next(node for node in ast.parse(setup).body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    calls = []
    def run(args, logs, **kwargs):
        calls.append(args)
        assert kwargs["capture_json"] is False
        raise subprocess.CalledProcessError(7, args)
    namespace = dict(run_process=run, LOGS="local-logs", Path=Path)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "notebook-execute", "exec"), namespace)
    with pytest.raises(subprocess.CalledProcessError):
        namespace["execute"]("python", "--output", "/tmp/path with spaces", 2)
    assert calls == [["python", "--output", "/tmp/path with spaces", "2"]]


def test_comparison_attention_and_ctc_keep_existing_resume_commands(tmp_path):
    path = Path(__file__).resolve().parents[1] / "notebooks/colab/benchmark_comparison.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "run(T2TT_ATTENTION_STAGE," in "".join(cell["source"]))
    calls = []
    def run(name, module, *arguments, **kwargs):
        values = list(map(str, arguments))
        validate_command(module, values)
        assert kwargs == {"gpu": True}
        calls.append((name, module, values))
    namespace = dict(RUN_TRANSFER=True, RUN_MONOTONIC=False, ROOT=str(tmp_path),
                     MANIFEST=tmp_path / "manifest.jsonl", Path=Path, run=run)
    namespace.update(comparison_paths(tmp_path))
    exec(source, namespace)
    assert [call[0] for call in calls] == ["attention", "ctc"]
    assert all("--resume" in call[2] for call in calls)
    assert "--batch-size" not in calls[0][2] and "--max-tokens" not in calls[0][2]
    from alignment_transfer_s2ut.cli import parser_for
    args = parser_for("extract_t2tt_attention").parse_args(calls[0][2])
    assert args.batch_size == 8 and args.layer == -1 and args.heads is None
    assert args.output == str(tmp_path / "mt")


def test_comparison_cuda_preflight_includes_batched_attention(tmp_path):
    path = Path(__file__).resolve().parents[1] / "notebooks/colab/benchmark_comparison.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code" and
                  "'tests/test_benchmark_native.py::test_native_fp16_trainer_and_fp32_prior'" in "".join(cell["source"]))
    calls = []
    exec(source, dict(PREPARE_AUDIO_PACKS=False, PYTHON="python", REPO=tmp_path,
                      execute=lambda *a, **kw: calls.append((a, kw))))
    assert len(calls) == 1
    assert calls[0][0][:3] == ("python", "-m", "pytest")
    assert "tests/test_attention_extraction.py::test_batched_native_attention_matches_single_pair_on_cuda" in calls[0][0]


@pytest.mark.parametrize("notebook", ["benchmark_comparison", "pilot", "pipeline"])
def test_notebook_overrides_colab_inline_backend_for_cli_subprocesses(tmp_path, monkeypatch, notebook):
    path = Path(__file__).resolve().parents[1] / f"notebooks/colab/{notebook}.ipynb"
    source = "\n".join("".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
                       if cell["cell_type"] == "code")
    setting = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign) and
                   isinstance(node.targets[0], ast.Subscript) and
                   ast.unparse(node.targets[0]) == "os.environ['MPLBACKEND']")
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    exec(compile(ast.Module(body=[setting], type_ignores=[]), notebook, "exec"), {"os": os})
    assert os.environ["MPLBACKEND"] == "Agg"
    output = tmp_path / "plot.png"
    command = [sys.executable, "-c", "import matplotlib.pyplot as p,sys; p.plot([0,1]); p.savefig(sys.argv[1])", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes().startswith(b"\x89PNG")


def staging_helper(tmp_path, print_function=print):
    path = Path(__file__).resolve().parents[1] / "notebooks/colab/benchmark_comparison.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    common = next("".join(cell["source"]) for cell in cells if "process-logging" in cell.get("metadata", {}).get("tags", []))
    source = next("".join(cell["source"]) for cell in cells
                  if cell["cell_type"] == "code" and "def stage_inputs_live(" in "".join(cell["source"]))
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    namespace = dict(print=print_function)
    exec(common, namespace)
    namespace["LOGS"] = tmp_path
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook-input-staging", "exec"), namespace)
    return namespace["stage_inputs_live"]


def test_input_staging_displays_progress_before_completion_and_preserves_json(tmp_path):
    acknowledged = tmp_path / "progress-displayed"
    finished = tmp_path / "finished"
    visible = []
    def display(*args, **kwargs):
        visible.append(" ".join(map(str, args)))
        if args == ("[progress] copying audio\n",):
            assert not finished.exists()
            acknowledged.touch()
    # The child cannot complete until its progress has reached notebook print.
    script = """import sys,time,json
from pathlib import Path
print('[progress] copying audio',file=sys.stderr,flush=True)
deadline=time.monotonic()+10
while not Path(sys.argv[1]).exists():
    if time.monotonic()>deadline: raise SystemExit(9)
    time.sleep(.01)
Path(sys.argv[2]).touch()
print(json.dumps({'input_map':'local-map.json','files':128}))
"""
    result = staging_helper(tmp_path, display)([sys.executable, "-c", script, str(acknowledged), str(finished)])
    assert result == dict(input_map="local-map.json", files=128)
    assert any("pid=" in line for line in visible)
    record = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    progress_log = Path(record["log"])
    assert "[progress] copying audio" in progress_log.read_text(encoding="utf-8")
    assert json.loads(next(tmp_path.glob("*.stdout.log")).read_text(encoding="utf-8")) == result


def test_input_staging_retains_error_log_and_fails_on_child_error(tmp_path, capsys):
    script = "import sys; print('copy checksum mismatch',file=sys.stderr,flush=True); sys.exit(7)"
    with pytest.raises(subprocess.CalledProcessError) as error:
        staging_helper(tmp_path)([sys.executable, "-c", script])
    assert error.value.returncode == 7
    assert "copy checksum mismatch" in capsys.readouterr().out
    assert any("copy checksum mismatch" in p.read_text(encoding="utf-8")
               for p in tmp_path.glob("*.log"))


def test_input_staging_interrupt_stops_only_its_child(tmp_path):
    children = []
    original = subprocess.Popen
    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    def display(*args, **kwargs):
        if args == ("ready\n",):
            raise KeyboardInterrupt
    script = "import sys,time; print('ready',file=sys.stderr,flush=True); time.sleep(60)"
    with patch.object(subprocess, "Popen", launch), pytest.raises(KeyboardInterrupt):
        staging_helper(tmp_path, display)([sys.executable, "-c", script])
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.parametrize("name", ["benchmark_comparison", "pilot", "pipeline"])
def test_every_notebook_command_uses_the_canonical_logger(name):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / f"notebooks/colab/{name}.ipynb").read_text(encoding="utf-8"))
    canonical = ast.parse((root / "src/alignment_transfer_s2ut/process_logging.py").read_text(encoding="utf-8"))
    expected = {n.name: ast.dump(n) for n in canonical.body if isinstance(n, ast.FunctionDef)}
    common = next(c for c in notebook["cells"] if "process-logging" in c.get("metadata", {}).get("tags", []))
    actual = {n.name: ast.dump(n) for n in ast.parse("".join(common["source"])).body if isinstance(n, ast.FunctionDef)}
    assert all(actual[key] == value for key, value in expected.items())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or cell is common:
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"), ast.unparse(node)
