import argparse
import hashlib
import json
import logging
import runpy
import subprocess
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


def publish_receipt(work, backup, step, *, complete=True, method="mask_distillation", checkpoint="last"):
    target = backup / f"{step:012d}-fixture"
    target.mkdir(parents=True)
    (target / ("snapshot.zip" if complete else "snapshot.zip.tmp")).touch()
    if complete:
        plan_sha = hashlib.sha256((work / "native-run.json").read_bytes()).hexdigest()
        (target / "snapshot.json").write_text(json.dumps({
            "updates": step, "storage": "zip-v1",
            "identity": {"work": str(work), "command": native_command(work, method)},
            "files": {"native-run.json": plan_sha, "checkpoints/checkpoint_last.pt": checkpoint},
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
              wandb_tracking=(Path("/tracker/python"), "only-in-env", "contract", "mask_distillation"),
              after_training=[ns["PYTHON"], ns["REPO"] / "tools/transfer_bleu.py", "evaluate"])
    original, tracking = [args[args.index("--") + 1:] for args, _ in calls]
    assert tracking[tracking.index("--") + 1:] == original
    assert "only-in-env" not in tracking and calls[-1][1]["env"]["WANDB_API_KEY"] == "only-in-env"
    assert json.loads(tracking[tracking.index("--after-training") + 1])[-1] == "evaluate"


def test_reuses_training_run_for_saved_bleu_and_preserves_history(tmp_path, monkeypatch):
    from test_transfer_bleu import bleu

    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup, logs, output = [tmp_path / name for name in ("work", "backup", "logs", "bleu")]
    work.mkdir()
    contract = "a" * 64
    plan = work / "native-run.json"
    plan.write_text(json.dumps({"contract_hash": contract}), encoding="utf-8")
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    publish_receipt(work, backup, 25000, checkpoint="b" * 64)
    publish_receipt(work, backup, 38000, checkpoint="c" * 64)
    original = {"update": 38000, "train/main_loss": 2.2, "wandb_train_update": 38000}
    history = [original.copy()]
    summary = original.copy()
    inits, metrics = [], {}
    identifier = hashlib.sha256(contract.encode()).hexdigest()[:24]
    remote = SimpleNamespace(state="running", config={"condition": "mask_distillation"})

    def add_score(step, checkpoint, *, stale=False, wrong_plan=False):
        values = dict(method="mask_distillation", contract_hash=contract, update=step, checkpoint_sha256=checkpoint,
                      aux_bleu=23., asr_bleu=15., aux_failures=0, asr_failures=0, status="completed",
                      dev_sha256="d" * 64, policy={"seed": 1})
        if stale:
            values.update(status="completed_with_failures", asr_failures=1)
        values["result_hash"] = bleu.result_hash(values)
        scope = bleu.evaluation_scope("f" * 64 if wrong_plan else plan_sha, values["policy"], values["dev_sha256"])
        target = output / "mask_distillation" / scope / f"{step:09d}" / "result.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(values), encoding="utf-8")
        return target

    saved = add_score(25000, "b" * 64)
    before = saved.read_bytes()
    add_score(1000, "e" * 64, stale=True)
    add_score(5000, "e" * 64, wrong_plan=True)
    add_score(50000, "e" * 64)

    def log(values):
        history.append(values)
        summary.update(values)

    def init(**kwargs):
        inits.append(kwargs)
        assert kwargs["id"] == identifier
        return SimpleNamespace(summary=summary, log=log, define_metric=lambda name, **kw: metrics.update({name: kw}),
                               url="https://wandb.ai/team/project", finish=lambda: None)

    def lookup(path):
        assert path == f"team/project/{identifier}"
        return remote

    api = SimpleNamespace(default_entity="team", run=lookup)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=dict, Api=lambda **kw: api))
    monkeypatch.setitem(tool["writer_lock"].__globals__, "gettempdir", lambda: str(tmp_path))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project", entity="",
                              group="exp/trial", method="mask_distillation", bleu_results=output, contract=None)
    with pytest.raises(RuntimeError, match="active"):
        tool["sync_saved_bleu"](args)
    assert len(inits) == 0 and history == [original]

    # The only active training reporter consumes the Drive queue.
    args.contract = contract
    done = Event()
    done.set()
    tool["report"](args, done)
    add_score(10000, "e" * 64)  # This scoring receipt survived the training archive.
    remote.state = "finished"
    tool["sync_saved_bleu"](args)
    tool["sync_saved_bleu"](args)
    assert history[0] == original and [row["update"] for row in history] == [38000, 25000, 10000]
    assert history[1]["eval/asr_bleu"] == 15. and history[1]["eval/aux_failures"] == 0
    assert metrics["train/*"] == metrics["eval/*"] == {"step_metric": "update"}
    assert [init["resume"] for init in inits] == ["allow", "must", "must"]
    assert all("config" not in init and "name" not in init for init in inits[1:])
    assert saved.read_bytes() == before

    from filelock import Timeout
    with tool["writer_lock"](contract), pytest.raises(Timeout):
        tool["sync_saved_bleu"](args)
    assert len(history) == 3
    # A completed score remains usable after archive retention prunes its source snapshot.
    (backup / "000000025000-fixture" / "snapshot.zip").unlink()
    points = tool["saved_bleu_points"](args, plan_sha, 38000, contract=contract)
    assert [step for step, _ in points] == [10000, 25000]


def test_wandb_worker_stays_active_until_checkpoint_evaluation_finishes(monkeypatch, tmp_path):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    monkeypatch.setitem(tool["writer_lock"].__globals__, "gettempdir", lambda: str(tmp_path))
    active, calls = Event(), []

    def report(args, stop):
        active.set()
        assert stop.wait(5)
        calls.append("flushed")

    def process(command, logs, name, env):
        assert active.wait(5)
        assert "WANDB_API_KEY" not in env
        calls.append(command[0])

    monkeypatch.setenv("WANDB_API_KEY", "fixture-secret")
    monkeypatch.setitem(tool["supervise"].__globals__, "report", report)
    monkeypatch.setitem(tool["supervise"].__globals__, "run_process", process)
    tool["supervise"](SimpleNamespace(contract="test", logs=tmp_path, command=["train"], after_training=["evaluate"]))
    assert calls == ["train", "evaluate", "flushed"]


def test_wandb_sdk_persists_out_of_order_update_coordinates(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    from wandb.proto import wandb_internal_pb2
    import struct

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    tools = Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"
    script = """
import runpy, sys
from pathlib import Path
from types import SimpleNamespace
import wandb
tool = runpy.run_path(sys.argv[1])
args = SimpleNamespace(entity='', project='bleu-axis-offline-test', logs=Path(sys.argv[2]),
                       group='fixture', method='distillation')
run = tool['open_run'](wandb, args, 'fixture')
run.log({'update': 38000, 'train/main_loss': 2.2})
sent = {}
for step in (25000, 1000):
    tool['sync_bleu'](run, [(step, dict(checkpoint_sha256=str(step), aux_bleu=21.0,
                                     asr_bleu=15.0, aux_failures=0, asr_failures=0))], sent)
run.finish()
"""
    subprocess.run([sys.executable, "-c", script, str(tools), str(tmp_path)], check=True)
    # Small offline .wandb files contain protobuf records with a 7-byte header
    # followed by complete frames. Read the actual records emitted by the SDK.
    data = next(tmp_path.rglob("*.wandb")).read_bytes()
    assert data.startswith(b":W&B")
    offset = 7
    history, metrics = [], {}
    while offset < len(data):
        _, size, kind = struct.unpack_from("<IHB", data, offset)
        assert kind == 1
        offset += 7
        record = wandb_internal_pb2.Record.FromString(data[offset:offset + size])
        offset += size
        if record.HasField("metric"):
            metrics[record.metric.name or record.metric.glob_name] = record.metric.step_metric
        if record.HasField("history"):
            history.append({item.key or "/".join(item.nested_key): json.loads(item.value_json)
                            for item in record.history.item})
    assert metrics["eval/*"] == metrics["train/*"] == "update"
    assert [(row["_step"], row["update"]) for row in history] == [(0, 38000), (1, 25000), (2, 1000)]
    assert history[0]["train/main_loss"] == 2.2
    assert [row["eval/asr_bleu"] for row in history[1:]] == [15., 15.]


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
