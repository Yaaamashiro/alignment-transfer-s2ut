"""Bootstrap authentication and credential containment before package install."""
import ast
import base64
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType

import pytest

from alignment_transfer_s2ut.process_logging import run_process

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools/colab_github.py"
AUTH = runpy.run_path(str(HELPER))
REPOSITORY = "https://github.com/Yaaamashiro/alignment-transfer-s2ut.git"
BENCHMARK = "https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark.git"
TOKEN = "test_secret_for_mock_git_only"
BASIC = base64.b64encode(("x-access-token:" + TOKEN).encode()).decode()


def secrets(monkeypatch, token=None, error=None):
    colab, userdata = ModuleType("google.colab"), ModuleType("google.colab.userdata")
    def get(key):
        assert key == "GITHUB_TOKEN"
        if error:
            raise error("Secret retrieval failed")
        return token
    userdata.get = get
    colab.userdata = userdata
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)


def call(command, logs, **kwargs):
    return AUTH["run_github_git"](command, logs, runner=run_process, repositories=[REPOSITORY], **kwargs)


@pytest.fixture
def fake_transport(tmp_path):
    # The actual bootstrap worker and process logger run in separate processes.
    # Replace only their Git transport, so no real private repo or PAT is needed.
    shim = tmp_path / "transport"
    shim.mkdir()
    fake = shim / "fake_git.py"
    fake.write_text('''import base64, os, sys
if "remote" in sys.argv:
    print("https://github.com/Yaaamashiro/alignment-transfer-s2ut.git")
    raise SystemExit(0)
headers = [os.environ.get(f"GIT_CONFIG_VALUE_{i}", "")
           for i in range(int(os.environ.get("GIT_CONFIG_COUNT", "0")))]
authorization = next((h for h in headers if h.startswith("Authorization: Basic ")), "")
if authorization:
    encoded = authorization.split()[-1]
    token = base64.b64decode(encoded).decode().split(":", 1)[1]
    print("echoed " + authorization)
    print("echoed " + token, file=sys.stderr)
code = 0 if os.environ["MOCK_PRIVATE"] == "0" or authorization == os.environ["MOCK_VALID_HEADER"] else 128
print("mock Git completed" if code == 0 else "fatal: Authentication failed")
raise SystemExit(code)
''', encoding="utf-8")
    (shim / "sitecustomize.py").write_text('''import os, subprocess, sys
from pathlib import Path
original = subprocess.Popen
def transport(args, *a, **kw):
    if Path(args[0]).name.lower() in {"git", "git.exe"}:
        args = [sys.executable, os.environ["MOCK_GIT_SCRIPT"], *args[1:]]
    return original(args, *a, **kw)
subprocess.Popen = transport
''', encoding="utf-8")
    return dict(os.environ, PYTHONPATH=str(shim), MOCK_GIT_SCRIPT=str(fake),
                MOCK_VALID_HEADER="Authorization: Basic " + BASIC)


@pytest.mark.parametrize("private,token,success", [
    (False, None, True), (False, TOKEN, True), (True, TOKEN, True),
    (True, None, False), (True, "invalid_test_secret", False),
])
def test_public_private_auth_and_all_log_outputs(tmp_path, monkeypatch, capsys, fake_transport, private, token, success):
    secrets(monkeypatch, token)
    environment = dict(fake_transport, MOCK_PRIVATE="1" if private else "0")
    before = environment.copy()
    command = ["git", "clone", "--no-checkout", REPOSITORY, str(tmp_path / "checkout")]
    if success:
        assert call(command, tmp_path / "logs", env=environment).returncode == 0
        exception = ""
    else:
        with pytest.raises(RuntimeError, match="GITHUB_TOKEN") as error:
            call(command, tmp_path / "logs", env=environment)
        exception = str(error.value)
    output = capsys.readouterr().out + exception
    assert environment == before
    for path in (tmp_path / "logs").rglob("*"):
        if path.is_file():
            output += path.read_text(encoding="utf-8")
    for value in filter(None, (token, TOKEN, BASIC)):
        assert value not in output
    if token:
        assert "[REDACTED]" in output
    if not success:
        assert "Contents: Read-only" in output and "Secrets" in output


@pytest.mark.parametrize("error", [ImportError, PermissionError, TimeoutError])
def test_unavailable_colab_secrets_allow_public_reads(tmp_path, monkeypatch, error):
    secrets(monkeypatch, error=error)
    environments = []
    def runner(command, logs, **kw):
        environments.append(dict(kw["env"]))
        return subprocess.CompletedProcess(command, 0)
    AUTH["run_github_git"](["git", "ls-remote", REPOSITORY, "HEAD"], tmp_path,
                           runner=runner, repositories=[REPOSITORY])
    assert all(not value.startswith("Authorization:") for value in environments[0].values())


def test_scoped_headers_prompt_trace_controls_and_environment_cleanup(tmp_path, monkeypatch):
    secrets(monkeypatch, TOKEN)
    received, references = [], []
    def runner(command, logs, **kw):
        assert TOKEN not in repr(command) and BASIC not in repr(command)
        references.append(kw["env"])
        received.append(dict(kw["env"]))
        return subprocess.CompletedProcess(command, 0)
    env = dict(os.environ, GIT_TRACE="1", GIT_TRACE_CURL="1", GIT_TRACE_CURL_NO_DATA="0", GIT_CURL_VERBOSE="1")
    before = env.copy()
    AUTH["run_github_git"](["git", "ls-remote", REPOSITORY, "HEAD"], tmp_path,
                           runner=runner, repositories=[REPOSITORY], env=env)
    assert env == before and references == [{}]
    child = received[0]
    assert child["GIT_TERMINAL_PROMPT"] == "0" and child["GIT_ASKPASS"] == ""
    assert all(key not in child for key in ("GIT_TRACE", "GIT_TRACE_CURL", "GIT_CURL_VERBOSE"))
    for url, expected in ((REPOSITORY + "/info/refs", "Authorization: Basic " + BASIC),
                          (BENCHMARK, ""), ("https://example.com/repo.git", "")):
        result = subprocess.run(["git", "config", "--get-urlmatch", "http.extraHeader", url],
                                env=child, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == expected
    AUTH["run_github_git"](["git", "ls-remote", BENCHMARK, "HEAD"], tmp_path,
                           runner=runner, repositories=[REPOSITORY, BENCHMARK])
    settings = received[-1]
    matched = subprocess.run(["git", "config", "--get-urlmatch", "http.extraHeader", BENCHMARK],
                             env=settings, capture_output=True, text=True, check=True)
    assert matched.stdout.strip() == "Authorization: Basic " + BASIC


def test_exceptions_cannot_reveal_headers_and_clear_the_child_environment(tmp_path, monkeypatch):
    secrets(monkeypatch, TOKEN)
    environments = []
    def fail(command, logs, **kw):
        environments.append(kw["env"])
        raise OSError(TOKEN + BASIC)
    with pytest.raises(RuntimeError) as error:
        AUTH["run_github_git"](["git", "ls-remote", REPOSITORY], tmp_path,
                               runner=fail, repositories=[REPOSITORY])
    assert TOKEN not in str(error.value) and BASIC not in str(error.value)
    assert environments == [{}]


@pytest.mark.parametrize("url", ["https://secret@github.com/o/r.git", "https://github.com/o/r.git?secret=value",
                                 "https://github.com/o/r.git#secret"])
def test_credentials_in_urls_are_rejected_before_logging(tmp_path, url):
    def unexpected(*a, **kw):
        pytest.fail("unsafe command reached the logger")
    with pytest.raises(ValueError, match="without credentials"):
        AUTH["run_github_git"](["git", "clone", url], tmp_path, runner=unexpected, repositories=[REPOSITORY])


@pytest.mark.parametrize("token", [None, TOKEN])
def test_real_git_clone_fetch_sha_and_reexecution_keep_clean_remotes(tmp_path, monkeypatch, token):
    secrets(monkeypatch, token)
    source, checkout = tmp_path / "source", tmp_path / "checkout"
    subprocess.run(["git", "init", "--initial-branch=main", str(source)], check=True, capture_output=True)
    (source / "data.txt").write_text("first", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "data.txt"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-m", "fixture"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    call(["git", "clone", "--no-checkout", str(source), str(checkout)], tmp_path / "logs")
    for _ in range(2):
        call(["git", "-C", str(checkout), "fetch", "origin", sha], tmp_path / "logs")
        call(["git", "-C", str(checkout), "checkout", "--detach", sha], tmp_path / "logs")
        assert subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() == sha
    config = (checkout / ".git/config").read_text(encoding="utf-8")
    assert str(source).replace("\\", "/") in config.replace("\\\\", "/")
    assert TOKEN not in config and BASIC not in config and "extraheader" not in config.lower()


def test_stored_remote_credentials_are_rejected_without_disclosure(tmp_path, monkeypatch, capsys):
    secrets(monkeypatch, TOKEN)
    subprocess.run(["git", "init", str(tmp_path / "checkout")], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path / "checkout"), "remote", "add", "origin",
                    "https://old_secret@github.com/o/r.git"], check=True)
    with pytest.raises(RuntimeError):
        call(["git", "-C", str(tmp_path / "checkout"), "fetch", "origin", "HEAD"], tmp_path / "logs")
    output = capsys.readouterr().out + "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "logs").iterdir())
    assert "old_secret" not in output and TOKEN not in output


def test_every_notebook_embeds_the_bootstrap_and_routes_git_through_it(tmp_path, monkeypatch):
    secrets(monkeypatch)
    expected = {node.name: ast.dump(node) for node in ast.parse(HELPER.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef)}
    notebooks = list((ROOT / "notebooks").rglob("*.ipynb"))
    assert len(notebooks) == 7
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["cells"][1]["id"] == "github-access-guide"
        code, = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
                  and "def run_process(" in "".join(c["source"])]
        ns = {}
        exec(code, ns)
        assert ns["GITHUB_GIT_WORKER"] == AUTH["GITHUB_GIT_WORKER"]
        actual = {node.name: ast.dump(node) for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
        assert all(actual[name] == value for name, value in expected.items())
        received = []
        ns.update(LOGS=tmp_path, REPO_URL=REPOSITORY,
                  run_process=lambda *a, **kw: received.append((a, kw)))
        ns["execute"]("git", "fetch", "origin", "fixed-sha")
        wrapped = received.pop()[0][0]
        assert wrapped[:3] == [sys.executable, "-c", AUTH["GITHUB_GIT_WORKER"]]
        assert wrapped[3:] == ["git", "fetch", "origin", "fixed-sha"]
        ns["execute"]("git", "checkout", "--detach", "fixed-sha")
        assert received.pop()[0][0] == ["git", "checkout", "--detach", "fixed-sha"]
