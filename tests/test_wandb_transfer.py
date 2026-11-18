import argparse
import hashlib
import json
import logging
import runpy
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from test_training_notebook import namespace


def native_command(work, method="mask_distillation"):
    return ["python", "-m", "alignment_transfer_s2ut.benchmark", "train",
            "--method", method, "--output", str(work), "--calibration", "verified.json",
            "--mask-top-p", "0.8", "--all-layers", "--resume"]


def publish_receipt(work, backup, step, *, complete=True, method="mask_distillation"):
    target = backup / f"{step:012d}-fixture"
    target.mkdir(parents=True)
    (target / ("snapshot.zip" if complete else "snapshot.zip.tmp")).touch()
    if complete:
        plan_sha = hashlib.sha256((work / "native-run.json").read_bytes()).hexdigest()
        (target / "snapshot.json").write_text(json.dumps({
            "updates": step, "storage": "zip-v1",
            "identity": {"work": str(work), "command": native_command(work, method)},
            "files": {"native-run.json": plan_sha, "checkpoints/checkpoint_last.pt": "last"},
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


@pytest.mark.parametrize("prefix", ["INFO:{tag}:", "2026-11-18 00:00:00 | INFO | {tag} | "])
@pytest.mark.parametrize("method", ["distillation", "mask_distillation", "annealed_distillation"])
def test_backfill_historical_fairseq_training_and_dev(tmp_path, monkeypatch, prefix, method):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, other, backup, logs = [tmp_path / name for name in ("work", "other", "backup", "logs")]
    for folder in (work, other, backup, logs):
        folder.mkdir()
    (work / "native-run.json").write_text('{"contract_hash":"matching-plan"}', encoding="utf-8")
    (work / "alignment-train.jsonl").write_text(
        '{"step":100,"train/main_loss":2.0,"alignment/lambda":10.0,"alignment/w_0":0.4}\n',
        encoding="utf-8",
    )
    (work / "native-dev.jsonl").write_text('{"step":100,"dev/main_loss":2.5}\n', encoding="utf-8")
    publish_receipt(work, backup, 100, method=method)
    expected = native_command(work, method)

    def make_log(name, command, lines, started="2026-11-18T00:00:02+00:00"):
        path = logs / (name + ".log")
        path.write_text("".join(lines), encoding="utf-8")
        (logs / (name + ".json")).write_text(json.dumps({
            "name": name, "started_utc": started, "command": command, "log": str(path),
        }), encoding="utf-8")
        return path

    def make_plan(name, second, contract):
        stdout = logs / (name + ".stdout.log")
        stdout.write_text(json.dumps({"contract_hash": contract}), encoding="utf-8")
        (logs / (name + ".json")).write_text(json.dumps({
            "status": "succeeded", "started_utc": f"2026-11-18T00:00:0{second}+00:00",
            "command": [*expected[:3], "plan", *expected[4:]], "stdout_log": str(stdout),
        }), encoding="utf-8")

    def checkpoint(step):
        return f"[recovery] learned_at_least={step} durable_updates={step}\n"

    def train(step, **fields):
        return prefix.format(tag="train_inner") + json.dumps({
            "epoch": 2, "update": 1.042, "num_updates": str(step), **fields,
        }) + "\n"

    layered = ("D", "w", "kl_layer", "valid_rows", "worst_layer_fraction")
    scalars = ("kl_mean", "kl_worst", "loss", "teacher_entropy", "student_entropy", "task_loss", "worst_layer")
    if method == "mask_distillation":
        layered += ("region_mass", "coverage", "selected_frames")
        scalars += ("top_p",)
    if method == "annealed_distillation":
        scalars += ("annealing_progress", "task_only_start", "task_only_phase")
    values = {f"alignment/{metric}_{layer}": "0.35" for layer in range(6) for metric in layered}
    values.update({f"alignment/{metric}": "0.12" for metric in scalars})
    values.update({"train/main_loss": "2.7", "alignment/w_0": "0.99", "alignment/lambda": "999",
                   "alignment/tau": "nan", "alignment/invalid": "inf", "loss_scale": "128", "secret": "private"})
    dev = prefix.format(tag="dev") + '{"dev_num_updates":"100","dev_loss":"3.3","dev_nll_loss":"2.9"}\n'
    make_plan("old-contract", 0, "different-plan")
    make_log("old-contract-train", expected, [train(100, **{"alignment/foreign": "7"}), checkpoint(100)],
             started="2026-11-18T00:00:00.500000+00:00")
    make_plan("matching-contract", 1, "matching-plan")
    # This is the outer notebook receipt from a run with W&B disabled.
    past = ["python", "-m", "alignment_transfer_s2ut.pilot", "--", "python", "-m",
            "alignment_transfer_s2ut.colab_runtime", "--training", "--", *expected, "--stop-after-update", "200"]
    historical = make_log("notebook:train", past,
                          [train(100, **values), dev, checkpoint(100), train(200, **{"alignment/w_4": ".99"})])
    make_log("wandb-training-child", expected, [train(100, **values), dev, checkpoint(100)])
    make_log("other-output", native_command(other, method), [train(100, **{"alignment/w_4": ".77"}), checkpoint(100)])
    wrong_method = "distillation" if method == "mask_distillation" else "mask_distillation"
    make_log("other-method", native_command(work, wrong_method), [train(100, **{"alignment/w_4": ".8"}), checkpoint(100)])
    make_log("other-policy", [*expected, "--mask-lambda=99"], [train(100, **{"alignment/w_4": ".8"}), checkpoint(100)])
    make_log("spoofed-progress", expected, [prefix.format(tag="other") + '{"num_updates":"100","alignment/w_4":"7"}\n',
                                           checkpoint(100)])
    points, summary = [], {}
    def log(values):
        points.append(values)
        summary.update(values)
    def init(**kwargs):
        return SimpleNamespace(summary=summary, log=log, define_metric=lambda *a, **kw: None,
                               url="https://wandb.ai/local/test", finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=dict))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project",
                              entity="", group="exp/trial", method=method,
                              contract="matching-plan")
    done = Event()
    done.set()
    tool["report"](args, done)
    recovered, = (p for p in points if "wandb_layer_update" in p)
    assert recovered["update"] == 100
    for key in values.keys() & recovered.keys() - {"secret", "train/main_loss", "alignment/w_0", "alignment/lambda"}:
        assert recovered[key] == float(values[key])
    expected_metrics = set(values) - {"secret", "train/main_loss", "alignment/w_0", "alignment/lambda",
                                      "alignment/tau", "alignment/invalid", "loss_scale"}
    assert expected_metrics <= recovered.keys()
    assert recovered["train/main_loss_fairseq"] == 2.7
    assert recovered["train/loss_scale"] == 128
    assert "alignment/foreign" not in recovered
    assert recovered["dev/loss"] == 3.3 and recovered["dev/nll_loss"] == 2.9
    assert {k for k in recovered if k in {"secret", "train/main_loss", "alignment/w_0", "alignment/lambda",
                                           "alignment/tau", "alignment/invalid", "dev/main_loss"}} == set()

    # A later snapshot cannot authorize the tail of a different attempt.
    publish_receipt(work, backup, 200, method=method)
    tool["report"](args, done)
    assert [p["update"] for p in points if "wandb_layer_update" in p] == [100]
    make_log("resumed", expected, [train(200, **{"alignment/w_4": ".74"}), checkpoint(200)])
    tool["report"](args, done)
    assert [(p["update"], p["alignment/w_4"]) for p in points if "wandb_layer_update" in p] == [(100, .35), (200, .74)]
    with historical.open("a", encoding="utf-8") as stream:
        stream.write(checkpoint(200))
        stream.write(prefix.format(tag="train_inner") + '{"num_updates": "300", "alignment/w_4":')
    tool["report"](args, done)
    assert [p["update"] for p in points if "wandb_layer_update" in p] == [100, 200]


def test_reads_observed_colab_log_format():
    # Match the Colab prefix from the incident report with representative metric values.
    fixture = Path(__file__).with_name("fixtures") / "fairseq_colab_progress.txt"
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    progress, validation = [tool["fairseq_row"](line) for line in fixture.read_bytes().splitlines()]
    assert progress == (7800, {"alignment/D_0": .125, "alignment/lambda": 10.0})
    assert validation == (7000, {"dev/loss": 3.5, "dev/main_loss_fairseq": 2.75})


@pytest.mark.parametrize("format_string", [logging.BASIC_FORMAT, "%(levelname)s | %(name)s | %(message)s"])
def test_reads_pinned_fairseq_json_progress_output(caplog, format_string):
    bar = pytest.importorskip("fairseq.logging.progress_bar").JsonProgressBar([None, None], epoch=2, log_interval=1)
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    with caplog.at_level("INFO"):
        for _ in bar:
            bar.log({"num_updates": 100, "alignment/D_0": .125, "alignment/lambda": 10.0}, tag="train_inner", step=100)
            break
        bar.print({"num_updates": 100, "loss": 3.5, "train/main_loss": 2.75}, tag="dev", step=100)
    outputs = [record for record in caplog.records if record.name in {"train_inner", "dev"}]
    formatter = logging.Formatter(format_string)
    progress, validation = [tool["fairseq_row"](formatter.format(record).encode()) for record in outputs]
    assert progress == (100, {"alignment/D_0": .125, "alignment/lambda": 10.0})
    assert validation == (100, {"dev/loss": 3.5, "dev/main_loss_fairseq": 2.75})
