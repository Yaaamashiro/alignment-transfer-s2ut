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


def test_standalone_commands_parse_and_preserve_all_five_conditions():
    calls = []
    namespace = dict(ROOT="/tmp/output with spaces", CORPUS="/tmp/corpus with spaces",
        RUNS="/tmp/output with spaces/strong", PYTHON="python", UPDATES=2,
        ANNEAL_STEPS=10000, RECIPE="strong", execute=lambda *args: calls.append(list(map(str, args))))
    for source in pipeline_cells():
        if "execute(" in source and "git" not in source:
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
    helper = next(node for node in ast.parse(setup).body if isinstance(node, ast.FunctionDef))
    calls = []
    def run(args, check):
        calls.append(args)
        assert check is True
        raise subprocess.CalledProcessError(7, args)
    namespace = dict(subprocess=SimpleNamespace(run=run))
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "notebook-execute", "exec"), namespace)
    with pytest.raises(subprocess.CalledProcessError):
        namespace["execute"]("python", "--output", "/tmp/path with spaces", 2)
    assert calls == [["python", "--output", "/tmp/path with spaces", "2"]]


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


def staging_helper(tmp_path, print_function=print, subprocess_module=subprocess):
    path = Path(__file__).resolve().parents[1] / "notebooks/colab/benchmark_comparison.ipynb"
    source = next("".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
                  if cell["cell_type"] == "code" and "def stage_inputs_live(" in "".join(cell["source"]))
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    namespace = dict(LOGS=tmp_path, time=time, subprocess=subprocess_module, json=json, print=print_function)
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
    assert any("PID:" in line for line in visible)
    progress_log = next(p for p in tmp_path.glob("stage-inputs-*.log") if not p.name.endswith(".stdout.log"))
    assert "[progress] copying audio" in progress_log.read_text(encoding="utf-8")
    assert json.loads(next(tmp_path.glob("*.stdout.log")).read_text(encoding="utf-8")) == result


def test_input_staging_retains_error_log_and_fails_on_child_error(tmp_path, capsys):
    script = "import sys; print('copy checksum mismatch',file=sys.stderr,flush=True); sys.exit(7)"
    with pytest.raises(subprocess.CalledProcessError) as error:
        staging_helper(tmp_path)([sys.executable, "-c", script])
    assert error.value.returncode == 7
    assert "copy checksum mismatch" in capsys.readouterr().out
    assert any("copy checksum mismatch" in p.read_text(encoding="utf-8")
               for p in tmp_path.glob("stage-inputs-*.log"))


def test_input_staging_interrupt_stops_only_its_child(tmp_path):
    children = []
    def launch(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        children.append(child)
        return child
    def display(*args, **kwargs):
        if args == ("ready\n",):
            raise KeyboardInterrupt
    proxy = SimpleNamespace(Popen=launch, PIPE=subprocess.PIPE,
        TimeoutExpired=subprocess.TimeoutExpired, CalledProcessError=subprocess.CalledProcessError)
    script = "import sys,time; print('ready',file=sys.stderr,flush=True); time.sleep(60)"
    with pytest.raises(KeyboardInterrupt):
        staging_helper(tmp_path, display, proxy)([sys.executable, "-c", script])
    assert len(children) == 1 and children[0].poll() is not None
