"""Observe progress during execution, not just a log dump after completion."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from alignment_transfer_s2ut import process_logging as logging


def record_at(root):
    return json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))


def test_silent_child_reports_liveness_and_live_metadata_before_completion(tmp_path, monkeypatch):
    acknowledge, finished = tmp_path / "seen", tmp_path / "finished"
    seen = []
    def display(*args, **kwargs):
        line = " ".join(map(str, args))
        seen.append(line)
        if "liveness only; progress not confirmed" in line and not acknowledge.exists():
            value = record_at(tmp_path)
            assert value["status"] == "running" and value["pid"] > 0
            assert not finished.exists()
            assert "liveness only" in Path(value["log"]).read_text(encoding="utf-8")
            acknowledge.touch()
    monkeypatch.setattr(logging, "print", display, raising=False)
    script = """import sys,time,json
from pathlib import Path
deadline=time.monotonic()+10
while not Path(sys.argv[1]).exists():
    if time.monotonic()>deadline: raise SystemExit(9)
    time.sleep(.01)
Path(sys.argv[2]).touch()
print(json.dumps({'complete':True}))
"""
    result = logging.run_process([sys.executable, "-c", script, str(acknowledge), str(finished)],
        tmp_path, name="silent-stage", capture_json=True, heartbeat_seconds=.1)
    assert json.loads(result.stdout) == {"complete": True}
    assert any("started" in line for line in seen) and any("completed" in line for line in seen)
    assert record_at(tmp_path)["status"] == "succeeded"


def test_fast_lines_and_errors_are_not_throttled_or_lost(tmp_path, capsys):
    script = "import sys; print('first'); print('important-middle'); print('last',file=sys.stderr); sys.exit(7)"
    with pytest.raises(subprocess.CalledProcessError) as error:
        logging.run_process([sys.executable, "-c", script], tmp_path, name="failure")
    assert error.value.returncode == 7
    visible = capsys.readouterr().out
    record = record_at(tmp_path)
    saved = Path(record["log"]).read_text(encoding="utf-8")
    assert all(line in visible and line in saved for line in ("first", "important-middle", "last"))
    assert "failed" in visible and "full log:" in visible
    assert record["status"] == "failed_or_interrupted" and record["returncode"] == 7


def test_json_stdout_is_separate_and_environment_overrides_are_preserved(tmp_path, capsys):
    environment = dict(os.environ, PYTHONUNBUFFERED="0", TEST_LOG_ENV="retained")
    original = environment.copy()
    script = "import sys,json,os; print('download step 1/2',file=sys.stderr); print(json.dumps({'unbuffered':os.environ['PYTHONUNBUFFERED'],'custom':os.environ['TEST_LOG_ENV']}))"
    result = logging.run_process([sys.executable, "-c", script], tmp_path,
        capture_json=True, env=environment, name="json-stage")
    assert environment == original
    assert json.loads(result.stdout) == {"unbuffered": "1", "custom": "retained"}
    record = record_at(tmp_path)
    assert json.loads(Path(record["stdout_log"]).read_text(encoding="utf-8")) == json.loads(result.stdout)
    assert "download step 1/2" in capsys.readouterr().out
    assert "download step 1/2" in Path(record["log"]).read_text(encoding="utf-8")


def test_launch_failure_is_recorded(tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        logging.run_process([str(tmp_path / "does-not-exist")], tmp_path, name="missing-command")
    record = record_at(tmp_path)
    assert record["status"] == "failed_or_interrupted" and record["pid"] is None
    assert "FileNotFoundError" in record["exception"]
    assert "failed" in capsys.readouterr().out


def test_invalid_json_is_a_failure_even_when_the_process_exits_zero(tmp_path):
    with pytest.raises(json.JSONDecodeError):
        logging.run_process([sys.executable, "-c", "print('not JSON')"], tmp_path, capture_json=True)
    record = record_at(tmp_path)
    assert record["status"] == "failed_or_interrupted" and record["returncode"] == 0
    assert "not JSON" in Path(record["stdout_log"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("nested", [False, True])
def test_nested_observers_do_not_escape_the_notebook_process_group(tmp_path, monkeypatch, nested):
    original = subprocess.Popen
    observed = []
    def launch(*args, **kwargs):
        observed.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", launch)
    environment = dict(os.environ)
    environment.pop("AT_LOG_MANAGED_GROUP", None)
    if nested:
        environment["AT_LOG_MANAGED_GROUP"] = "1"
    logging.run_process([sys.executable, "-c", "print('done')"], tmp_path, env=environment)
    assert observed[0]["start_new_session"] == (os.name == "posix" and not nested)
    if os.name == "posix":
        assert observed[0]["env"]["AT_LOG_MANAGED_GROUP"] == "1"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cancellation")
def test_interrupt_terminates_nested_loggers_and_their_worker(tmp_path, monkeypatch):
    worker_pid = tmp_path / "worker-pid"
    def display(*args, **kwargs):
        if args == ("worker-ready\n",):
            raise KeyboardInterrupt
    monkeypatch.setattr(logging, "print", display, raising=False)
    worker = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); print('worker-ready',file=sys.stderr,flush=True); time.sleep(60)"
    script = "from alignment_transfer_s2ut.process_logging import run_process; import sys; run_process([sys.executable,'-c',sys.argv[1],sys.argv[2]],sys.argv[3],name='nested')"
    environment = dict(os.environ)
    environment.pop("AT_LOG_MANAGED_GROUP", None)
    with pytest.raises(KeyboardInterrupt):
        logging.run_process([sys.executable, "-c", script, worker, str(worker_pid), str(tmp_path / "inner")],
                            tmp_path / "outer", env=environment, heartbeat_seconds=.1)
    pid = int(worker_pid.read_text())
    state = Path(f"/proc/{pid}/stat")
    def stopped():
        try:
            return state.read_text().split()[2] == "Z"
        except FileNotFoundError:
            return True
    deadline = time.monotonic() + 3
    while not stopped() and time.monotonic() < deadline:
        time.sleep(.01)
    assert stopped()
