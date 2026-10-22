"""Check notebook commands against the real CLI parsers without running training."""
import argparse
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
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
