"""Standard-library GitHub authentication, embedded before Colab installation."""


GITHUB_GIT_WORKER = r'''
import base64
import os
import subprocess
import sys
from urllib.parse import quote

redactions = []
for i in range(int(os.environ.get("GIT_CONFIG_COUNT", "0"))):
    value = os.environ.get(f"GIT_CONFIG_VALUE_{i}", "")
    if value.startswith("Authorization: Basic "):
        encoded = value.split()[-1]
        token = base64.b64decode(encoded).decode("utf-8").split(":", 1)[1]
        redactions.extend((value, encoded, token, quote(token, safe="")))
redactions = sorted(set(redactions), key=len, reverse=True)
process = None
try:
    command = sys.argv[1:]
    position = 1
    while position < len(command) and command[position] == "-C":
        position += 2
    if command[position] in {"fetch", "pull"}:
        remote = next((arg for arg in command[position + 1:] if not arg.startswith("-")), "origin")
        if "://" not in remote:
            checked = subprocess.run([*command[:position], "remote", "get-url", "--all", remote],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace")
            from urllib.parse import urlsplit
            for address in checked.stdout.splitlines():
                url = urlsplit(address)
                if url.username or url.password or url.query or url.fragment:
                    raise ValueError("unsafe stored remote")
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1)
    for line in process.stdout:
        for secret in redactions:
            line = line.replace(secret, "[REDACTED]")
        print(line, end="", flush=True)
    code = process.wait()
except Exception:
    print("Git could not complete the requested operation.", file=sys.stderr, flush=True)
    code = 1
finally:
    if process is not None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
if code:
    print("Check the repository URL and revision. For a private repository, add "
          "GITHUB_TOKEN to Colab Secrets, enable notebook access, and grant the "
          "fine-grained token Contents: Read-only for the selected repository. "
          "Check token expiry and any organization approval.", file=sys.stderr, flush=True)
sys.exit(code)
'''


def run_github_git(command, logs, *, runner, repositories, name="git", env=None, cwd=None,
                   capture_json=False):
    """Authenticate network Git calls with URL-scoped, process-only headers."""
    import base64
    import os
    from pathlib import Path
    import re
    import sys
    from urllib.parse import urlsplit

    command = list(map(str, command))
    position = 1
    while position < len(command) and command[position] == "-C":
        position += 2
    operation = command[position] if position < len(command) else ""
    if operation not in {"clone", "fetch", "pull", "ls-remote"}:
        return runner(command, logs, name=name, env=env, cwd=cwd, capture_json=capture_json)
    if Path(command[0]).name.lower() not in {"git", "git.exe"}:
        raise ValueError("GitHub authentication requires a git command")
    for argument in command:
        if "://" in argument:
            url = urlsplit(argument)
            if url.username or url.password or url.query or url.fragment:
                raise ValueError("Use repository URLs without credentials, queries, or fragments")
    scopes = []
    for repository in repositories:
        url = urlsplit(str(repository))
        if (url.scheme != "https" or url.netloc != "github.com" or url.query or url.fragment
                or not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", url.path)):
            raise ValueError("Authentication scopes must be plain HTTPS GitHub repository URLs")
        scopes.append(url.geturl())

    token = None
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        pass  # Public repositories remain accessible when Secrets are unavailable.
    if token is not None and (not isinstance(token, str) or any(ord(c) < 33 or ord(c) > 126 for c in token)):
        token = None
        raise ValueError("GITHUB_TOKEN must contain a valid GitHub token without whitespace")
    git_env = dict(os.environ if env is None else env)
    for key in list(git_env):
        if (key.startswith(("GIT_TRACE", "GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
                or key in {"GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_CURL_VERBOSE", "GITHUB_TOKEN", "GH_TOKEN"}):
            git_env.pop(key)
    git_env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", GIT_ASKPASS="", GIT_TRACE_REDACT="1")
    settings = [("credential.helper", ""), ("http.extraHeader", ""), ("http.followRedirects", "false")]
    header = None
    try:
        if token:
            header = "Authorization: Basic " + base64.b64encode(("x-access-token:" + token).encode()).decode()
            settings.extend(("http." + scope + ".extraHeader", header) for scope in dict.fromkeys(scopes))
        git_env["GIT_CONFIG_COUNT"] = str(len(settings))
        for i, (key, value) in enumerate(settings):
            git_env[f"GIT_CONFIG_KEY_{i}"] = key
            git_env[f"GIT_CONFIG_VALUE_{i}"] = value
        return runner([sys.executable, "-c", GITHUB_GIT_WORKER, *command], logs,
                      name=name, env=git_env, cwd=cwd, capture_json=capture_json)
    except Exception:
        raise RuntimeError("GitHub access failed. Check the logged URL/revision error and Colab Secrets "
                           "GITHUB_TOKEN access, selected repositories, Contents: Read-only, and expiry.") from None
    finally:
        token = header = None
        settings.clear()
        git_env.clear()
