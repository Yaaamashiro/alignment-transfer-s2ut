import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from test_training_notebook import namespace


def publish_receipt(work, backup, step, *, complete=True):
    target = backup / f"{step:012d}-fixture"
    target.mkdir(parents=True)
    (target / ("snapshot.zip" if complete else "snapshot.zip.tmp")).touch()
    if complete:
        plan_sha = hashlib.sha256((work / "native-run.json").read_bytes()).hexdigest()
        (target / "snapshot.json").write_text(json.dumps({
            "updates": step, "storage": "zip-v1", "identity": {"work": str(work)},
            "files": {"native-run.json": plan_sha},
        }), encoding="utf-8")


def test_tracking_resumes_and_uploads_only_published_numeric_rows(tmp_path, monkeypatch):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup, logs = [tmp_path / name for name in ("work", "backup", "logs")]
    work.mkdir()
    (work / "native-run.json").write_text('{"contract_hash":"matching-plan"}', encoding="utf-8")
    (work / "alignment-train.jsonl").write_text(
        "".join(json.dumps({"step": step, "train/main_loss": 1.5, "alignment/lambda": 10.,
                            "experiment_run": "/private/path", "train/seconds": float("nan")}) + "\n"
                for step in range(1, 102)), encoding="utf-8")
    (work / "native-dev.jsonl").write_text('{"step":100,"dev/main_loss":2.5}\n', encoding="utf-8")
    publish_receipt(work, backup, 100)
    publish_receipt(work, backup, 101, complete=False)
    points, inits, summary = [], [], {}

    def log(values):
        points.append(values)
        summary.update(values)

    def init(**kwargs):
        inits.append(kwargs)
        return SimpleNamespace(summary=summary, log=log, define_metric=lambda *a, **kw: None,
                               url="https://wandb.ai/local/test", finish=lambda: None)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=dict))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project", entity="",
                              group="exp/trial", method="mask_distillation", contract="matching-plan")
    done = Event()
    done.set()
    tool["report"](args, done)
    assert [row["update"] for row in points] == [1, 100, 100]
    assert all("experiment_run" not in row and "train/seconds" not in row for row in points)
    publish_receipt(work, backup, 102)
    with (work / "alignment-train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"step":102,"train/main_loss":1.2}')
    tool["report"](args, done)
    assert len(points) == 3
    with (work / "alignment-train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    tool["report"](args, done)
    assert [row["update"] for row in points] == [1, 100, 100, 102]
    assert len({session["id"] for session in inits}) == 1
    assert all(session["resume"] == "allow" for session in inits)


def test_notebook_wraps_benchmark_without_altering_recovery_command(tmp_path):
    source, ns = namespace(tmp_path, "training")
    calls = []
    ns.update(execute=lambda *args, **kwargs: calls.append((args, kwargs)))
    exec(source["helpers"], ns)
    module = "alignment_transfer_s2ut.benchmark"
    arguments = ["train", "--output", tmp_path / "mask", "--method", "mask_distillation"]
    ns["run"]("mask", module, *arguments, gpu=True)
    ns["run"]("mask", module, *arguments, gpu=True,
              wandb_tracking=(Path("/tracker/python"), "only-in-env", "contract", "mask_distillation"))
    original, tracking = [args[args.index("--") + 1:] for args, _ in calls]
    assert tracking[tracking.index("--") + 1:] == original
    assert "only-in-env" not in tracking and calls[-1][1]["env"]["WANDB_API_KEY"] == "only-in-env"



def test_backfill_layer_metrics_from_existing_matching_train_logs(tmp_path, monkeypatch):
    """Recover older fairseq values without touching the native scientific code."""
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, other, backup, logs = [tmp_path / name for name in ("work", "other", "backup", "logs")]
    for folder in (work, other, backup, logs):
        folder.mkdir()
    (work / "native-run.json").write_text('{"contract_hash":"matching-plan"}', encoding="utf-8")
    (work / "alignment-train.jsonl").write_text(
        '{"step":10,"train/main_loss":2.0,"alignment/lambda":1.0}\n',
        encoding="utf-8",
    )
    publish_receipt(work, backup, 10)

    def make_log(name, workpath, method, lines):
        path = logs / (name + ".log")
        path.write_text("".join(lines), encoding="utf-8")
        (logs / (name + ".json")).write_text(json.dumps({
            "name": "wandb-training-child",
            "command": ["python", "-m", "alignment_transfer_s2ut.benchmark",
                        "train", "--method", method, "--output", str(workpath)],
            "log": str(path),
        }), encoding="utf-8")
        return path

    progress = lambda step, value: (
        'INFO train_inner | ' + json.dumps({
            "update": step, "alignment/w_4": value,
            "alignment/D_4": value * 5, "alignment/kl_mean": 1.4,
            "alignment/worst_layer_fraction_4": 1.0,
            "secret": "do not upload",
        }) + "\n"
    )
    old_log = make_log("old-run", work, "distillation", [progress(1, .31), progress(11, .99)])
    make_log("other-method", other, "mask_distillation", [progress(2, .75)])
    make_log("wrong-method", work, "mask_distillation", [progress(3, .88)])

    points, summary = [], {}
    def log(values):
        points.append(values)
        summary.update(values)
    def init(**kwargs):
        return SimpleNamespace(summary=summary, log=log, define_metric=lambda *a, **kw: None,
                               url="https://wandb.ai/local/test", finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=dict))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project",
                              entity="", group="exp/trial", method="distillation",
                              contract="matching-plan")
    done = Event()
    done.set()
    tool["report"](args, done)
    layer_points = [p for p in points if "alignment/w_4" in p]
    assert len(layer_points) == 1
    assert layer_points[0]["update"] == 1
    assert layer_points[0]["alignment/w_4"] == .31
    assert "secret" not in layer_points[0]
    assert "alignment/worst_layer_fraction_4" not in layer_points[0]

    # Published updates only; earlier progress is not re-sent on a resumed W&B run.
    publish_receipt(work, backup, 11)
    tool["report"](args, done)
    layer_points = [p for p in points if "alignment/w_4" in p]
    assert [p["update"] for p in layer_points] == [1, 11]
    tool["report"](args, done)
    assert [p["update"] for p in points if "alignment/w_4" in p] == [1, 11]

    # Truncated progress and non-finite metrics cannot be published.
    with old_log.open("a", encoding="utf-8") as stream:
        stream.write('INFO train_inner | {"update": 12, "alignment/w_4":')
    publish_receipt(work, backup, 12)
    tool["report"](args, done)
    assert [p["update"] for p in points if "alignment/w_4" in p] == [1, 11]
