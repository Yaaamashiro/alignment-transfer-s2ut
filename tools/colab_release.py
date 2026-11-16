"""Validate the two Colab checkouts and synchronize the embedded bootstrap."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import uuid


GIT_CHECKOUT_STATE = r"""
import json
import subprocess
import sys

def read_git(path, *arguments):
    result = subprocess.run(
        ["git", "-C", path, *arguments], stdout=subprocess.PIPE, check=False,
        encoding="utf-8", errors="replace", text=True)
    return result.stdout if result.returncode == 0 else None

states = []
for path in sys.argv[1:]:
    head = read_git(path, "rev-parse", "--verify", "HEAD")
    status = read_git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none")
    states.append({
        "actual_head": head.strip() if head is not None else None,
        "worktree": ("unknown" if status is None else "dirty" if status else "clean"),
        "status_entries": status.rstrip("\0").split("\0") if status else [],
    })
print(json.dumps(states, ensure_ascii=True))
"""


def prepare_colab_release(execute, log_event, *, notebook, notebook_updated_at, experiment_id, transfer_id,
                          alignment, benchmark, logs, persistent):
    """Checkout only the pinned SHAs, record both observations, and fail closed."""
    checkouts = dict(alignment=alignment, benchmark_runtime=benchmark)
    errors = []
    for key, (path, url, revision) in checkouts.items():
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            errors.append(f"{key}: expected a full lowercase commit SHA")
    if not errors:
        for key, (path, url, revision) in checkouts.items():
            label = key.replace("_", "-")
            try:
                if not (path / ".git").exists():
                    execute("git", "clone", "--no-checkout", url, path, name="clone-" + label)
                execute("git", "-C", path, "fetch", "origin", revision, name="fetch-" + label)
                execute("git", "-C", path, "checkout", "--detach", revision, name="checkout-" + label)
            except Exception:
                errors.append(f"{key}: Git checkout failed; see the command log")
    try:
        result = execute(sys.executable, "-c", GIT_CHECKOUT_STATE,
                         *(value[0] for value in checkouts.values()),
                         name="inspect-colab-revisions", capture_json=True)
        states = json.loads(result.stdout)
    except Exception:
        errors.append("could not inspect the repositories; see the command log")
        states = [{"actual_head": None, "worktree": "unknown", "status_entries": []} for _ in checkouts]
    repositories = {}
    for (key, (_, _, pin)), state in zip(checkouts.items(), states, strict=True):
        repositories[key] = {"pinned_revision": pin, **state}
        if state["actual_head"] != pin:
            errors.append(f"{key}: HEAD differs from the pinned revision")
        if state["worktree"] != "clean":
            errors.append(f"{key}: worktree is {state['worktree']}")
    report = dict(schema="colab-release-v2", run_id=uuid.uuid4().hex,
                  started_utc=datetime.now(timezone.utc).isoformat(), notebook=notebook,
                  notebook_updated_at=notebook_updated_at, experiment_id=experiment_id,
                  transfer_experiment_id=transfer_id, status="FAIL" if errors else "PASS",
                  repositories=repositories, errors=errors)
    payload = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    filename = "colab-release-" + report["run_id"] + ".json"
    for root in (Path(logs), Path(persistent) / "execution-provenance"):
        root.mkdir(parents=True, exist_ok=True)
        with (root / filename).open("x", encoding="utf-8") as target:
            target.write(payload)
    log_event("colab-revision-check", report["status"], logs, json.dumps(report, ensure_ascii=True))
    print(payload, end="", flush=True)
    if errors:
        raise RuntimeError("Colab revision check failed. Resolve the reported Git state before running the experiment.")
    return report


# END NOTEBOOK BOOTSTRAP


def notebook_update_label(value):
    updated = datetime.fromisoformat(value)
    if updated.isoformat(timespec="minutes") != value or not value.endswith("+09:00"):
        raise ValueError("expected NOTEBOOK_UPDATED_AT in YYYY-MM-DDTHH:MM+09:00 format")
    return updated.strftime("%Y-%m-%d %H:%M JST")


def sync_colab_release(root, *, check=False):
    """Apply a reviewed release manifest to both standalone notebooks."""
    notebooks = root / "notebooks" / "colab"
    release = json.loads((notebooks / "release.json").read_text(encoding="utf-8"))
    names = ("NOTEBOOK_UPDATED_AT", "REPO_REVISION", "BENCHMARK_RUNTIME_REVISION")
    if set(release) != set(names) or any(
            not re.fullmatch(r"[0-9a-f]{40}", release[name]) for name in names[1:]):
        raise ValueError("expected a notebook update time and two fixed Git SHAs")
    updated_label = notebook_update_label(release["NOTEBOOK_UPDATED_AT"])
    bootstrap = (root / "tools" / "colab_release.py").read_text(encoding="utf-8").split(
        "# END NOTEBOOK BOOTSTRAP", 1)[0].rstrip()
    for name in ("training", "evaluate"):
        path = notebooks / (name + ".ipynb")
        original = path.read_text(encoding="utf-8")
        notebook = json.loads(original)
        cells = {cell.get("id"): cell for cell in notebook["cells"]}
        for key, cell_id in zip(names, ("settings", "paths", "paths"), strict=True):
            cell = cells[cell_id]
            contents, count = re.subn(rf"(?m)^{key} = .*", f"{key} = {release[key]!r}", "".join(cell["source"]))
            if count != 1:
                raise ValueError(f"{name}: expected one assignment to {key}")
            cell["source"] = contents.splitlines(keepends=True)
        for cell, label in ((notebook["cells"][0], "**Last updated** "),
                            (cells["settings"], "#@markdown **Last updated** ")):
            title, _, rest = "".join(cell["source"]).partition("\n")
            rest = re.sub(rf"(?m)^{re.escape(label)}[^\n]*\n?", "", rest, count=1).lstrip("\n")
            cell["source"] = (f"{title}\n\n{label}{updated_label}\n\n" + rest).splitlines(keepends=True)
        revision = cells["release-check"]
        marker = "# Start the revision check\n"
        _, found, suffix = "".join(revision["source"]).partition(marker)
        if not found:
            raise ValueError(f"{name}: missing bootstrap entry point")
        revision["source"] = ("#@title Verify Git revisions and record the notebook update time\n" +
                              bootstrap + "\n\n" + found + suffix).splitlines(keepends=True)
        updated = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        if updated != original:
            if check:
                raise ValueError(f"{name}: run python tools/colab_release.py to sync the release")
            path.write_text(updated, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    sync_colab_release(Path(__file__).resolve().parents[1], check=parser.parse_args().check)
