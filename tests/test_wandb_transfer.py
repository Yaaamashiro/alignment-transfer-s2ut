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
from zipfile import ZipFile

import pytest

from test_training_notebook import namespace


class RunNotFoundError(ValueError):
    pass


class CommError(Exception):
    def __init__(self, exc):
        super().__init__(str(exc))
        self.exc = exc


def sdk(init, lookup=None):
    def missing(path):
        raise CommError(RunNotFoundError(path))
    return SimpleNamespace(init=init, Settings=dict, apis=SimpleNamespace(public=SimpleNamespace(RunNotFoundError=RunNotFoundError)),
                           errors=SimpleNamespace(CommError=CommError),
                           Api=lambda **kw: SimpleNamespace(default_entity="team", run=lookup or missing))


def native_command(work, method="mask_distillation"):
    return ["python", "-m", "alignment_transfer_s2ut.benchmark", "train",
            "--method", method, "--output", str(work), "--calibration", "verified.json",
            "--mask-top-p", "0.8", "--all-layers", "--resume"]


def publish_receipt(work, backup, step, *, complete=True, method="mask_distillation", checkpoint="last", archive=False):
    target = backup / f"{step:012d}-fixture"
    target.mkdir(parents=True)
    (target / ("snapshot.zip" if complete else "snapshot.zip.tmp")).touch()
    if complete:
        files = {"native-run.json": hashlib.sha256((work / "native-run.json").read_bytes()).hexdigest(),
                 "checkpoints/checkpoint_last.pt": checkpoint}
        if archive:
            payloads = {name: (work / name).read_bytes()
                        for name in ("native-run.json", "alignment-train.jsonl", "native-dev.jsonl")
                        if (work / name).is_file()}
            payloads["checkpoints/checkpoint_last.pt"] = b"not a model"
            with ZipFile(target / "snapshot.zip", "w") as output:
                for name, value in payloads.items():
                    output.writestr(name, value)
            files = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
        (target / "snapshot.json").write_text(json.dumps({
            "updates": step, "storage": "zip-v1",
            "identity": {"work": str(work), "command": native_command(work, method)},
            "files": files,
        }), encoding="utf-8")
    return target


def test_tracking_resumes_and_uploads_only_published_numeric_rows(tmp_path, monkeypatch):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup, logs = [tmp_path / name for name in ("work", "backup", "logs")]
    work.mkdir()
    (work / "native-run.json").write_text('{"contract_hash":"matching-plan"}', encoding="utf-8")
    (work / "alignment-train.jsonl").write_text(
        "".join(json.dumps({"step": step, "metric_schema": 2, "train/unit_ce_nats": 1.5, "alignment/lambda": 10.,
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
                               url="https://wandb.ai/local/test", entity="team", finish=lambda: None)

    monkeypatch.setitem(sys.modules, "wandb", sdk(init))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project", entity="",
                              group="exp/trial", method="mask_distillation", contract="matching-plan")
    done = Event()
    done.set()
    tool["report"](args, done)
    assert [row["update"] for row in points] == [1, 100, 100]
    assert all("experiment_run" not in row and "train/update/seconds" not in row for row in points)
    assert points[0]["train/unit_ce_nats"] == 1.5
    publish_receipt(work, backup, 102)
    with (work / "alignment-train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"step":102,"metric_schema":2,"train/unit_ce_nats":1.2}')
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
    ns.update(execute=lambda *args, **kwargs: calls.append((args, kwargs)), WANDB_WRITER_BUCKET="shared-lock")
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
    assert calls[-1][1]["env"]["AT_LOG_WANDB_DRAIN"] == "1"
    assert tracking[tracking.index("--writer-bucket") + 1] == "shared-lock"
    assert json.loads(tracking[tracking.index("--after-training") + 1])[-1] == "evaluate"


@pytest.mark.parametrize("bucket", ["", "shared"])
def test_recovery_notebook_uses_only_cpu_audit_and_wandb(tmp_path, monkeypatch, bucket):
    source, ns = namespace(tmp_path, "training")
    recovery = source["wandb-recovery"]
    calls, authenticated = [], []
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(
        userdata=SimpleNamespace(get=lambda name: "only-in-env"),
        auth=SimpleNamespace(authenticate_user=lambda: authenticated.append(True)),
    ))
    ns.update(execute=lambda *args, **kwargs: calls.append((args, kwargs)), log_event=lambda *args: None,
              WANDB_PROJECT="project", WANDB_ENTITY="entity", WANDB_WRITER_BUCKET=bucket,
              TRANSFER_EXPERIMENT_ID="fixed-trial", EXPERIMENT_ID="experiment-001")
    exec(recovery, ns)
    assert not calls
    enabled = recovery.replace("RECOVER_WANDB_METRICS = False", "RECOVER_WANDB_METRICS = True")
    with pytest.raises(ValueError, match="confirm"):
        exec(enabled, ns)
    assert not calls
    exec(enabled.replace("CONFIRM_WANDB_WRITERS_STOPPED = False", "CONFIRM_WANDB_WRITERS_STOPPED = True"), ns)
    assert len(calls) == 5
    assert "--audit-metrics-only" in calls[0][0]
    assert any("wandb==0.31.0" in args for args, _ in calls)
    assert ("google-cloud-storage==3.4.1" in calls[-2][0]) == bool(bucket)
    assert bool(authenticated) == bool(bucket)
    command, options = calls[-1]
    assert command[command.index("--group") + 1] == "experiment-001/fixed-trial"
    assert command[command.index("--method") + 1] == "distillation"
    assert {"--sync-metrics-only", "--recover-writer", "--confirm-writers-stopped"} <= set(command)
    assert ("--writer-bucket" in command) == bool(bucket)
    assert "only-in-env" not in command and options["env"]["WANDB_API_KEY"] == "only-in-env"
    assert options["env"]["WANDB_WRITER_BUCKET"] == bucket
    assert options["env"]["AT_LOG_WANDB_DRAIN"] == "1"


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
    publish_receipt(work, backup, 50000, checkpoint="c" * 64)
    original = {"update": 50000, "train/main_loss": 2.2, "wandb_train_update": 50000}
    history = [original.copy()]
    summary = original.copy()
    inits, metrics = [], {}
    identifier = hashlib.sha256(contract.encode()).hexdigest()[:24]
    remote = SimpleNamespace(id=identifier, entity="team", project="project", group="exp/trial", state="finished",
                             config={"condition": "mask_distillation"}, summary=summary)

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
    add_score(2000, "e" * 64, stale=True)
    add_score(5000, "e" * 64, wrong_plan=True)
    add_score(125000, "e" * 64)

    def log(values):
        history.append(values)
        summary.update({key: value for key, value in values.items() if key != "update"})

    def init(**kwargs):
        inits.append(kwargs)
        assert kwargs["id"] == identifier
        return SimpleNamespace(summary=summary, log=log, define_metric=lambda name, **kw: metrics.update({name: kw}),
                               url="https://wandb.ai/team/project", entity="team", finish=lambda: None)

    def lookup(path):
        assert path == f"team/project/{identifier}"
        return remote

    monkeypatch.setitem(sys.modules, "wandb", sdk(init, lookup))
    monkeypatch.setitem(tool["writer_lock"].__globals__, "gettempdir", lambda: str(tmp_path))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project", entity="",
                              group="exp/trial", method="mask_distillation", bleu_results=output, contract=contract,
                              command=["train"], after_training=None)
    with pytest.raises(RuntimeError, match="No local W&B ownership receipt"):
        tool["sync_saved_bleu"](args)
    assert len(inits) == 0 and history == [original]

    # The only active training reporter consumes the Drive queue, then
    # authorizes sync-only on its own runtime after closing.
    monkeypatch.setitem(tool["report"].__globals__, "run_process", lambda *a, **kw: None)
    tool["supervise"](args)
    assert summary["train/max_update"] == 50000
    remote.state = "running"
    with pytest.raises(RuntimeError, match="active"):
        tool["sync_saved_bleu"](args)
    remote.state = "finished"
    done = Event()
    done.set()
    add_score(1000, "e" * 64)  # This scoring receipt survived the training archive.
    tool["sync_saved_bleu"](args)
    tool["sync_saved_bleu"](args)
    assert history[0] == original and [row["update"] for row in history] == [50000, 25000, 1000]
    assert summary["train/max_update"] == 50000
    assert history[1]["eval/asr_bleu"] == 15. and history[1]["eval/aux_failures"] == 0
    assert metrics["train/*"] == metrics["eval/*"] == {"step_metric": "update"}
    assert [init["resume"] for init in inits] == ["must", "must", "must"]
    assert all("config" not in init and "name" not in init for init in inits)
    assert saved.read_bytes() == before

    from filelock import Timeout
    with tool["writer_lock"](contract), pytest.raises(Timeout):
        tool["sync_saved_bleu"](args)
    assert len(history) == 3
    # Each simulated VM can acquire its own local lock against the same
    # shared checkpoint and stale remote state. The other VM lacks ownership.
    with tool["writer_lock"](contract), monkeypatch.context() as other:
        other.setitem(tool["writer_lock"].__globals__, "gettempdir", lambda: str(tmp_path / "other-vm"))
        with tool["writer_lock"](contract):
            pass
        with pytest.raises(RuntimeError, match="No local W&B ownership receipt"):
            tool["sync_saved_bleu"](args)
    assert len(inits) == 3 and len(history) == 3
    # A VM that used to be the owner must not follow a stale local receipt
    # after a different reporter has used and closed the W&B run.
    summary[tool["REPORTER_SESSION"]] = "other-vm"
    with pytest.raises(RuntimeError, match="Another training reporter"):
        tool["sync_saved_bleu"](args)
    assert len(inits) == 3 and len(history) == 3
    # A completed score remains usable after archive retention prunes its source snapshot.
    (backup / "000000025000-fixture" / "snapshot.zip").unlink()
    points = tool["saved_bleu_points"](args, plan_sha, 50000, contract=contract)
    assert [step for step, _ in points] == [1000, 25000]
    (work / "alignment-train.jsonl").write_text('{"step":100000,"metric_schema":2,"train/unit_ce_nats":1.8}\n', encoding="utf-8")
    publish_receipt(work, backup, 100000)
    tool["report"](args, done)
    assert [row["update"] for row in history] == [50000, 25000, 1000, 100000]
    assert summary["train/max_update"] == 100000


def test_wandb_worker_stays_active_until_checkpoint_evaluation_finishes(monkeypatch, tmp_path):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    monkeypatch.setitem(tool["writer_lock"].__globals__, "gettempdir", lambda: str(tmp_path))
    active, calls = Event(), []
    owner = tool["writer_path"]("test", ".owner.json")
    owner.parent.mkdir()
    owner.write_text('{"session":"old"}', encoding="utf-8")

    def report(args, stop):
        assert owner.is_file()
        active.set()
        assert stop.wait(5)
        calls.append("flushed")

    def process(command, logs, name, env):
        assert active.wait(5)
        assert "WANDB_API_KEY" not in env
        assert "AT_LOG_WANDB_DRAIN" not in env
        calls.append(command[0])

    monkeypatch.setenv("WANDB_API_KEY", "fixture-secret")
    monkeypatch.setitem(tool["report"].__globals__, "report", report)
    monkeypatch.setitem(tool["report"].__globals__, "run_process", process)
    tool["supervise"](SimpleNamespace(contract="test", logs=tmp_path, command=["train"], after_training=["evaluate"]))
    assert calls == ["train", "evaluate", "flushed"]
    assert json.loads(owner.read_text(encoding="utf-8")) == {"session": "old"}


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
run.log({'update': 50000, 'train/main_loss': 2.2, 'wandb_train_update': 50000})
tool['set_training_progress'](run, 50000)
sent = {}
for step in (25000, 1000):
    tool['sync_bleu'](run, [(step, dict(checkpoint_sha256=str(step), aux_bleu=21.0,
                                     asr_bleu=15.0, aux_failures=0, asr_failures=0))], sent)
    assert run.summary['train/max_update'] == 50000
    assert run.summary.get('update') is None
run.log({'update': 100000, 'train/main_loss': 1.8, 'wandb_train_update': 100000})
tool['set_training_progress'](run, 100000)
assert run.summary['train/max_update'] == 100000
tool['set_training_progress'](run, 25000)  # retained remote progress stays monotonic
assert run.summary['train/max_update'] == 100000
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
    assert [(row["_step"], row["update"]) for row in history] == [(0, 50000), (1, 25000), (2, 1000), (3, 100000)]
    assert history[0]["train/main_loss"] == 2.2
    assert [row["eval/asr_bleu"] for row in history[1:3]] == [15., 15.]
    assert history[-1]["train/main_loss"] == 1.8


@pytest.mark.parametrize("prefix", ["INFO:{tag}:", "2026-11-18 00:00:00 | INFO | {tag} | "])
@pytest.mark.parametrize("method", ["distillation", "mask_distillation", "annealed_distillation"])
def test_backfill_historical_fairseq_training_and_dev(tmp_path, monkeypatch, capsys, prefix, method):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, other, backup, logs = [tmp_path / name for name in ("work", "other", "backup", "logs")]
    for folder in (work, other, backup, logs):
        folder.mkdir()
    contract = "a" * 64
    (work / "native-run.json").write_text(json.dumps({"contract_hash": contract, "transfer": {"method": method}}), encoding="utf-8")
    (work / "alignment-train.jsonl").write_text(
        '{"step":100,"train/main_loss":2.0,"alignment/lambda":10.0,"alignment/w_0":0.4}\n',
        encoding="utf-8",
    )
    (work / "native-dev.jsonl").write_text('{"step":100,"dev/main_loss":2.5}\n', encoding="utf-8")
    publish_receipt(work, backup, 100, method=method, archive=True)
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
    make_plan("matching-contract", 1, contract)
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
                               url="https://wandb.ai/local/test", entity="team", finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", sdk(init))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project",
                              entity="", group="exp/trial", method=method,
                              contract=contract)
    done = Event()
    done.set()
    tool["report"](args, done)
    marker = tool["TRAIN_WINDOW"]
    recovered, = (p for p in points if marker in p)
    assert recovered["update"] == 100
    omitted = {"alignment/tau", "alignment/invalid", "alignment/task_only_start"}
    for key in (k for k in values if k.startswith("alignment/") and k not in omitted | {"alignment/worst_layer"}):
        assert recovered[key.replace("alignment/", "alignment/window/", 1)] == float(values[key])
    assert recovered["alignment/window/worst_layer"] == 0  # argmax, not the old averaged 0.12
    assert recovered["train/window/main_loss_fairseq"] == 2.7
    assert recovered["train/window/loss_scale"] == 128
    assert not ({"alignment/window/" + key.removeprefix("alignment/") for key in omitted | {"alignment/foreign"}} & recovered.keys())
    full_dev, = (p for p in points if tool["DEV_WINDOW"] in p)
    assert full_dev["dev/full/loss"] == 3.3 and full_dev["dev/full/nll_loss"] == 2.9
    logical, = (p for p in points if tool["TRAIN_UPDATE"] in p)
    assert logical["alignment/update/w_0"] == 0.4 and logical["alignment/update/lambda"] == 10.
    assert not {"train/main_loss", "train/unit_ce_nats"} & logical.keys()
    capsys.readouterr()
    tool["audit_saved_metrics"](args)
    audit = json.loads(capsys.readouterr().out)
    assert audit["legacy_main_loss_rows"] == audit["window_unit_ce_rows"] == 1
    assert audit["legacy_main_loss_min"] == 2.0 and audit["verified_update_unit_ce_rows"] == 0

    # A later snapshot cannot authorize the tail of a different attempt.
    publish_receipt(work, backup, 200, method=method)
    tool["report"](args, done)
    assert [p["update"] for p in points if marker in p] == [100]
    make_log("resumed", expected, [train(200, **{"alignment/w_4": ".74"}), checkpoint(200)])
    tool["report"](args, done)
    assert [(p["update"], p["alignment/window/w_4"]) for p in points if marker in p] == [(100, .35), (200, .74)]
    with historical.open("a", encoding="utf-8") as stream:
        stream.write(checkpoint(200))
        stream.write(prefix.format(tag="train_inner") + '{"num_updates": "300", "alignment/w_4":')
    tool["report"](args, done)
    assert [p["update"] for p in points if marker in p] == [100, 200]


def test_sync_only_fills_gaps_resumes_after_connection_failure(tmp_path, monkeypatch):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup, logs = (tmp_path / name for name in ("work", "backup", "logs"))
    work.mkdir()
    logs.mkdir()
    contract = "b" * 64
    (work / "native-run.json").write_text(json.dumps({"contract_hash": contract,
        "transfer": {"method": "mask_distillation"}}), encoding="utf-8")
    (work / "alignment-train.jsonl").write_text("".join(
        json.dumps({"step": step, "metric_schema": 2, "train/unit_ce_nats": 1.8,
                    "train/optimization_loss_nats": 2.3}) + "\n"
        for step in (100, 200, 300, 400)), encoding="utf-8")
    publish_receipt(work, backup, 300, archive=True)
    marker = tool["TRAIN_UPDATE"]
    history = [{marker: 300, "update": 300, "train/unit_ce_nats": 1.8}]
    summary = {marker: 300, "train/max_update": 300, tool["REPORTER_SESSION"]: "session"}
    owner = dict(entity="entity", project="project", method="mask_distillation", session="session")
    monkeypatch.setitem(tool["writer_lock"].__globals__, "gettempdir", lambda: str(tmp_path))
    with tool["writer_lock"](contract):
        tool["writer_path"](contract, ".owner.json").write_text(json.dumps(owner), encoding="utf-8")
    def scan_history(*, page_size, use_cache):
        assert page_size == 1000 and use_cache is False
        return iter(history)
    remote = SimpleNamespace(id=tool["run_id"](contract), entity="entity", project="project",
                             config={"condition": "mask_distillation"}, state="finished", summary=summary,
                             scan_history=scan_history)
    fail = [True]
    opened = []
    def log(values):
        if values["update"] == 200 and fail[0]:
            fail[0] = False
            raise OSError("temporary connection")
        history.append(values)
        summary.update(values)
    def init(**kw):
        assert kw["id"] == tool["run_id"](contract) and kw["resume"] == "must"
        opened.append(kw)
        return SimpleNamespace(summary=summary, log=log, define_metric=lambda *a, **kw: None,
                               url="https://wandb.ai/entity/project", finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", sdk(init, lambda path: remote))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project", entity="entity",
                              method="mask_distillation")
    with pytest.raises(OSError, match="temporary connection"):
        tool["sync_saved_metrics"](args)
    tool["sync_saved_metrics"](args)
    tool["sync_saved_metrics"](args)
    assert [row["update"] for row in history] == [300, 100, 200, 300]
    assert history[-1] == {marker: 300, "update": 300, "train/update/optimization_loss_nats": 2.3}
    assert len(opened) == 2  # The final idempotent retry never opens a writer.
    assert summary[marker] == summary["train/max_update"] == 300
    # A conflicting metric must block all writes, including a genuinely missing metric at another update.
    history[0]["train/unit_ce_nats"] = 1.7
    del history[1]["train/update/optimization_loss_nats"]
    with pytest.raises(ValueError, match="metric conflict for train/unit_ce_nats at update 300"):
        tool["sync_saved_metrics"](args)
    assert len(opened) == 2 and len(history) == 4
    remote.state = "running"
    with pytest.raises(RuntimeError, match="active"):
        tool["sync_saved_metrics"](args)


def test_cold_metric_recovery_verifies_run_and_snapshot_and_blocks_other_vms(tmp_path, monkeypatch):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup, logs = (tmp_path / name for name in ("work", "backup", "logs"))
    work.mkdir()
    contract, method, bucket = "f" * 64, "mask_distillation", "test-writer-bucket"
    (work / "native-run.json").write_text(json.dumps({"contract_hash": contract, "transfer": {"method": method}}),
                                          encoding="utf-8")
    row = {"step": 100, "metric_schema": 2, "train/unit_ce_nats": 2.0, "train/optimization_loss_nats": 2.5}
    (work / "alignment-train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    publish_receipt(work, backup, 100, archive=True)
    for file in work.iterdir():
        file.unlink()
    work.rmdir()
    run_id = tool["run_id"](contract)
    history = [{"update": 100, "train/unit_ce_nats": 2.0}]
    remote = SimpleNamespace(id=run_id, entity="team", project="project", group="exp/trial", state="finished",
                             config={"condition": method}, summary={}, scan_history=lambda **kw: iter(history))
    opened, held = [], set()
    from contextlib import contextmanager
    from wandb_writer import WriterLease
    @contextmanager
    def cloud_lock(name, identity, **kwargs):
        assert name == bucket and identity == dict(entity="team", project="project", run_id=run_id,
                                                   contract_hash=contract, method=method)
        if name in held:
            raise RuntimeError("Shared W&B writer lock is occupied")
        held.add(name)
        try:
            yield WriterLease()
        finally:
            held.remove(name)
    def lookup(path):
        assert path == f"team/project/{run_id}"
        return remote
    def init(**kwargs):
        assert held == {bucket} and kwargs["resume"] == "must" and kwargs["id"] == run_id
        opened.append(kwargs)
        # A second VM has a different FileLock and no local owner file.
        with monkeypatch.context() as other:
            other.setitem(tool["writer_path"].__globals__, "gettempdir", lambda: str(tmp_path / "other-vm"))
            with pytest.raises(RuntimeError, match="lock is occupied"):
                tool["sync_saved_metrics"](args)
        return SimpleNamespace(summary=remote.summary, log=history.append, define_metric=lambda *a, **kw: None,
                               url="https://wandb.ai/team/project", finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", sdk(init, lookup))
    monkeypatch.setitem(tool["writer_path"].__globals__, "gettempdir", lambda: str(tmp_path / "cold-vm"))
    monkeypatch.setitem(tool["report"].__globals__, "shared_writer", cloud_lock)
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project", entity="team", group="exp/trial",
                              method=method, writer_bucket=bucket, recover_writer=True, contract=contract)
    owner = tool["writer_path"](contract, ".owner.json")
    assert not owner.exists()
    for name, bad, error in (("id", "wrong", "run identity"), ("group", "different", "run identity"),
                             ("project", "other", "run identity"), ("state", "running", "active")):
        prior = getattr(remote, name)
        setattr(remote, name, bad)
        with pytest.raises(RuntimeError, match=error):
            tool["sync_saved_metrics"](args)
        setattr(remote, name, prior)
    args.contract = "0" * 64
    with pytest.raises(ValueError, match="expected training contract"):
        tool["sync_saved_metrics"](args)
    args.contract = contract
    assert not opened and not held
    tool["sync_saved_metrics"](args)
    tool["sync_saved_metrics"](args)
    assert len(opened) == 1 and not held and not work.exists()
    assert history[-1] == {"update": 100, "train/update/optimization_loss_nats": 2.5, tool["TRAIN_UPDATE"]: 100}
    assert json.loads(owner.read_text())["session"] == remote.summary[tool["REPORTER_SESSION"]]
    assert remote.summary[tool["WRITER_BUCKET"]] == bucket
    args.recover_writer = False
    args.writer_bucket = None
    with pytest.raises(RuntimeError, match="bucket already bound"):
        tool["sync_saved_metrics"](args)


@pytest.mark.parametrize("existing_session", ["old-reporter", None])
def test_manual_recovery_restores_ownership_without_adding_history(tmp_path, monkeypatch, existing_session):
    from filelock import Timeout

    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup, logs = (tmp_path / name for name in ("work", "backup", "logs"))
    work.mkdir()
    contract, method = "d" * 64, "mask_distillation"
    (work / "native-run.json").write_text(json.dumps({"contract_hash": contract, "transfer": {"method": method}}),
                                          encoding="utf-8")
    (work / "alignment-train.jsonl").write_text(
        '{"step":100,"metric_schema":2,"train/unit_ce_nats":2.0}\n', encoding="utf-8")
    publish_receipt(work, backup, 100, archive=True)
    for path in work.iterdir():
        path.unlink()
    work.rmdir()
    globals_ = tool["writer_path"].__globals__
    monkeypatch.setitem(globals_, "gettempdir", lambda: str(tmp_path / "new-cpu"))
    session_key, run_id = tool["REPORTER_SESSION"], tool["run_id"](contract)
    history = [{"update": 100, "train/unit_ce_nats": 2.0}]
    opened, writes = [], []
    remote = SimpleNamespace(id=run_id, entity="team", project="project", group="exp/trial", state="finished",
                             config={"condition": method}, summary={session_key: existing_session})
    owner = tool["writer_path"](contract, ".owner.json")
    pending = tool["writer_path"](contract, ".pending.json")
    active_during_scan = [True]

    def scan_history(**options):
        assert options == {"page_size": 1000, "use_cache": False}
        with pytest.raises(Timeout):
            with tool["writer_lock"](contract):
                pass
        yield from history
        if active_during_scan[0]:
            remote.state = "running"

    def init(**kwargs):
        assert kwargs["resume"] == "must" and kwargs["id"] == run_id
        assert pending.is_file()
        opened.append(kwargs)
        return SimpleNamespace(url="https://wandb.ai/team/project", summary=remote.summary,
                               define_metric=lambda *a, **kw: None, log=writes.append, finish=lambda: None)

    remote.scan_history = scan_history
    monkeypatch.setitem(sys.modules, "wandb", sdk(init, lambda path: remote))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project", entity="team", group="exp/trial",
                              method=method, recover_writer=True, confirm_writers_stopped=False)
    with pytest.raises(RuntimeError, match="confirm-writers-stopped"):
        tool["sync_saved_metrics"](args)
    args.confirm_writers_stopped = True
    with tool["writer_lock"](contract):
        with pytest.raises(Timeout):
            tool["sync_saved_metrics"](args)
        pending.write_text("{}", encoding="utf-8")
    remote.summary[tool["WRITER_BUCKET"]] = "required"
    with pytest.raises(RuntimeError, match="bucket already bound"):
        tool["sync_saved_metrics"](args)
    del remote.summary[tool["WRITER_BUCKET"]]
    with pytest.raises(RuntimeError, match="active"):
        tool["sync_saved_metrics"](args)
    assert pending.exists() and not owner.exists() and not opened
    remote.state, active_during_scan[0] = "finished", False
    tool["sync_saved_metrics"](args)
    tool["sync_saved_metrics"](args)
    assert not pending.exists() and not work.exists() and not writes
    assert len(opened) == (0 if existing_session else 1)
    assert json.loads(owner.read_text(encoding="utf-8"))["session"] == remote.summary[session_key]

    args.recover_writer = False
    args.bleu_results = tmp_path / "saved-bleu"
    args.contract = contract
    monkeypatch.setitem(globals_, "saved_bleu_points", lambda *a, **kw: [(100, {
        "contract_hash": contract, "checkpoint_sha256": "b" * 64,
        "aux_bleu": 15, "asr_bleu": 10, "aux_failures": 0, "asr_failures": 0})])
    tool["sync_saved_bleu"](args)
    assert len(writes) == 1 and writes[0]["eval/asr_bleu"] == 10


@pytest.mark.parametrize("changed", [None, "state", "session", "group"])
def test_manual_recovery_rechecks_before_writing_missing_metrics(tmp_path, monkeypatch, changed):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup, logs = (tmp_path / name for name in ("work", "backup", "logs"))
    work.mkdir()
    contract, method = "b" * 64, "mask_distillation"
    (work / "native-run.json").write_text(json.dumps({"contract_hash": contract, "transfer": {"method": method}}),
                                          encoding="utf-8")
    (work / "alignment-train.jsonl").write_text(
        '{"step":100,"metric_schema":2,"train/unit_ce_nats":2.0}\n', encoding="utf-8")
    publish_receipt(work, backup, 100, archive=True)
    session_key = tool["REPORTER_SESSION"]
    remote = SimpleNamespace(id=tool["run_id"](contract), entity="team", project="project", group="exp/trial",
                             state="finished", config={"condition": method}, summary={session_key: "old"})

    def scan_history(**options):
        if changed == "state":
            remote.state = "running"
        elif changed == "session":
            remote.summary[session_key] = "new"
        elif changed == "group":
            remote.group = "other"
        return iter(())

    points = []

    def init(**options):
        assert changed is None and options["resume"] == "must"
        return SimpleNamespace(url="https://wandb.ai/team/project", summary=remote.summary,
                               define_metric=lambda *a, **kw: None, log=points.append, finish=lambda: None)

    remote.scan_history = scan_history
    monkeypatch.setitem(tool["writer_path"].__globals__, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "wandb", sdk(init, lambda path: remote))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, project="project", entity="team", group="exp/trial",
                              method=method, recover_writer=True, confirm_writers_stopped=True)
    owner = tool["writer_path"](contract, ".owner.json")
    if changed is None:
        tool["sync_saved_metrics"](args)
        assert owner.is_file()
        assert points == [{"update": 100, tool["TRAIN_UPDATE"]: 100, "train/unit_ce_nats": 2.0}]
    else:
        with pytest.raises(RuntimeError, match="active|Another W&B writer|run identity"):
            tool["sync_saved_metrics"](args)
        assert not owner.exists() and not points


def test_training_reporter_uses_bound_writer_bucket_until_finish(tmp_path, monkeypatch):
    from contextlib import contextmanager

    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    from wandb_writer import WriterLease

    work, backup, logs = (tmp_path / name for name in ("work", "backup", "logs"))
    work.mkdir()
    contract, method = "a" * 64, "mask_distillation"
    (work / "native-run.json").write_text(json.dumps({"contract_hash": contract}), encoding="utf-8")
    publish_receipt(work, backup, 100)
    summary = {tool["WRITER_BUCKET"]: "shared"}
    remote = SimpleNamespace(id=tool["run_id"](contract), state="finished", entity="team", project="project",
                             group="exp/trial", config={"condition": method}, summary=summary)
    observed = []
    @contextmanager
    def cloud_lock(bucket, identity, **kwargs):
        assert bucket == "shared" and identity["run_id"] == remote.id
        observed.append("acquired")
        lease = WriterLease()
        yield lease
        assert not lease.unfinished
        observed.append("released")
    def init(**options):
        assert observed == ["acquired"] and options["resume"] == "must"
        observed.append("opened")
        return SimpleNamespace(url="https://wandb.ai/team/project", entity="team", summary=summary,
                               define_metric=lambda *a, **kw: None, finish=lambda: observed.append("finished"))
    monkeypatch.setitem(sys.modules, "wandb", sdk(init, lambda path: remote))
    monkeypatch.setitem(tool["report"].__globals__, "shared_writer", cloud_lock)
    args = argparse.Namespace(work=work, backup=backup, logs=logs, contract=contract, method=method, entity="team",
                              project="project", group="exp/trial", writer_bucket=None)
    stop = Event()
    stop.set()
    with pytest.raises(RuntimeError, match="bucket already bound"):
        tool["report"](args, stop)
    args.writer_bucket = "shared"
    assert tool["report"](args, stop)["session"] == summary[tool["WRITER_TOKEN"]]
    assert observed == ["acquired", "opened", "finished", "released"]


def test_training_reporter_retries_preflight_and_sends_saved_metrics(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from requests.exceptions import Timeout

    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    from wandb_writer import WriterBusy, WriterLease

    work, backup, logs = (tmp_path / name for name in ("work", "backup", "logs"))
    work.mkdir()
    contract, method = "e" * 64, "mask_distillation"
    (work / "native-run.json").write_text(json.dumps({"contract_hash": contract}), encoding="utf-8")
    (work / "alignment-train.jsonl").write_text(
        '{"step":100,"metric_schema":2,"train/unit_ce_nats":1.6}\n', encoding="utf-8")
    publish_receipt(work, backup, 100)
    remote = SimpleNamespace(id=tool["run_id"](contract), state="finished", entity="team", project="project",
                             group="exp/trial", config={"condition": method}, summary={})
    outcomes = iter(["missing", "timeout", "running", "finished", "finished", "running", "finished", "finished"])
    waits, acquisitions, events, written = [], [], [], []
    stop = Event()

    def lookup(path):
        state = next(outcomes)
        if state == "missing":
            raise CommError(RunNotFoundError(path))
        if state == "timeout":
            raise CommError(Timeout("read timed out"))
        remote.state = state
        return remote

    @contextmanager
    def cloud_lock(*args, **kwargs):
        if not acquisitions:
            acquisitions.append("busy")
            raise WriterBusy("already held")
        acquisitions.append("acquired")
        try:
            yield WriterLease()
        finally:
            acquisitions.append("released")

    def init(**kwargs):
        events.append(kwargs["resume"])
        stop.set()
        return SimpleNamespace(url="https://wandb.ai/team/project", entity="team", summary=remote.summary,
                               define_metric=lambda *a, **kw: None, log=written.append,
                               finish=lambda: events.append("finished"))

    def wait(seconds):
        waits.append(seconds)
        assert len(waits) <= 6
        return stop.is_set()

    monkeypatch.setattr(stop, "wait", wait)
    globals_ = tool["report"].__globals__
    monkeypatch.setitem(globals_, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setitem(globals_, "shared_writer", cloud_lock)
    monkeypatch.setitem(sys.modules, "wandb", sdk(init, lookup))
    tool["store_owner"](contract, dict(entity="team", project="project", method=method, session="previous"))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, contract=contract, method=method, entity="team",
                              project="project", group="exp/trial", writer_bucket="shared")
    assert tool["report"](args, stop)["session"] == remote.summary[tool["REPORTER_SESSION"]]
    assert waits == [2, 4, 8, 16, 32, 15]
    assert acquisitions == ["busy", "acquired", "released", "acquired", "released"]
    assert events == ["must", "finished"]
    assert [row["update"] for row in written] == [100]


def test_pinned_sdk_transient_core_errors_are_distinguished_from_denials():
    wandb = pytest.importorskip("wandb")
    from wandb.proto.wandb_api_pb2 import ApiErrorResponse
    from wandb.sdk.lib.service.service_connection import WandbApiFailedError

    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    for status in (0, 429, 503, 401, 403, 404):
        underlying = WandbApiFailedError("client failure", ApiErrorResponse(http_status=status))
        assert tool["retryable_preflight"](wandb.errors.CommError("client failure", underlying), wandb) is (
            status in (0, 429, 503))


@pytest.mark.parametrize("failure", ["contract", "destination", "forbidden", "init", "finish"])
def test_reporter_leaves_uncertain_writer_for_explicit_recovery(tmp_path, monkeypatch, failure):
    from requests import HTTPError, Response, Timeout

    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup, logs = (tmp_path / name for name in ("work", "backup", "logs"))
    work.mkdir()
    contract, method = "e" * 64, "mask_distillation"
    (work / "native-run.json").write_text(json.dumps({"contract_hash": contract}), encoding="utf-8")
    publish_receipt(work, backup, 100)
    remote = SimpleNamespace(id=tool["run_id"](contract), state="finished", entity="team", project="project",
                             group="exp/trial", config={"condition": method}, summary={})
    init_calls = []
    stop = Event()
    monkeypatch.setattr(stop, "wait", lambda seconds: pytest.fail(f"retired writer retried after {seconds}s"))
    monkeypatch.setitem(tool["report"].__globals__, "gettempdir", lambda: str(tmp_path))

    def lookup(path):
        if failure == "forbidden":
            response = Response()
            response.status_code = 403
            raise CommError(HTTPError(response=response))
        if failure == "contract":
            remote.config["alignment_contract_hash"] = "wrong"
        return remote

    def fail():
        raise Timeout("connection state unknown")

    def init(**kwargs):
        init_calls.append(kwargs)
        if failure == "init":
            fail()
        stop.set()
        return SimpleNamespace(url="https://wandb.ai/team/project", entity="team", summary=remote.summary,
                               define_metric=lambda *a, **kw: None, finish=fail)

    monkeypatch.setitem(sys.modules, "wandb", sdk(init, lookup))
    args = argparse.Namespace(work=work, backup=backup, logs=logs, contract=contract, method=method, entity="team",
                              project="project", group="exp/trial")
    if failure == "destination":
        tool["store_owner"](contract, dict(entity="team", project="other", method=method, session="previous"))
    if failure in ("contract", "destination", "forbidden"):
        with pytest.raises((RuntimeError, CommError)):
            tool["report"](args, stop)
        assert not init_calls and not tool["writer_path"](contract, ".pending.json").exists()
    else:
        with pytest.raises(Timeout, match="state unknown"):
            tool["report"](args, stop)
        assert len(init_calls) == 1 and tool["writer_path"](contract, ".pending.json").exists()
        with pytest.raises(RuntimeError, match="unconfirmed"):
            tool["report"](args, stop)
        assert len(init_calls) == 1


def test_metric_audit_reads_verified_zip_on_cold_cpu_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    work, backup = tmp_path / "work", tmp_path / "backup"
    work.mkdir()
    plan = {"contract_hash": "c" * 64, "transfer": {"method": "mask_distillation"}}
    (work / "native-run.json").write_text(json.dumps(plan), encoding="utf-8")
    (work / "alignment-train.jsonl").write_text('{"step":100,"metric_schema":2,"train/unit_ce_nats":2.0}\n', encoding="utf-8")
    (work / "native-dev.jsonl").write_text('{"step":100,"dev/main_loss":3.0}\n', encoding="utf-8")
    receipt = publish_receipt(work, backup, 100, archive=True)
    for file in work.iterdir():
        file.unlink()
    work.rmdir()
    command = [sys.executable, "-S", str(root / "tools/wandb_transfer.py"), "--audit-metrics-only",
               "--work", str(work), "--backup", str(backup), "--logs", str(tmp_path / "logs"),
               "--method", "mask_distillation", "--project", "project"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    audit = json.loads(result.stdout)
    assert audit["verified_snapshot_members"] == ["native-run.json", "alignment-train.jsonl", "native-dev.jsonl"]
    assert audit["verified_update_unit_ce_rows"] == audit["verified_dev_rows"] == 1
    assert not work.exists()
    assert (receipt / "snapshot.zip").is_file()


def test_metric_snapshot_does_not_open_checkpoint_or_use_local_metric_copies(tmp_path, monkeypatch):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup = tmp_path / "work", tmp_path / "backup"
    work.mkdir()
    contract = "e" * 64
    (work / "native-run.json").write_text(json.dumps({"contract_hash": contract,
        "transfer": {"method": "mask_distillation"}}), encoding="utf-8")
    archived = b'{"step":100,"metric_schema":2,"train/unit_ce_nats":2.0}\n'
    train = work / "alignment-train.jsonl"
    train.write_bytes(archived)
    publish_receipt(work, backup, 100, archive=True)
    train.write_bytes(b"unverified local data")
    opened = []
    original = ZipFile.open
    def record_open(self, entry, *args, **kwargs):
        opened.append(entry.filename)
        return original(self, entry, *args, **kwargs)
    monkeypatch.setattr(ZipFile, "open", record_open)
    args = argparse.Namespace(work=work, backup=backup, method="mask_distillation", contract=contract)
    with tool["metric_snapshot"](args) as (_, _, _, local, members):
        assert (local / train.name).read_bytes() == archived
        assert opened == list(members) == ["native-run.json", train.name]
        assert not (local / "native-dev.jsonl").exists()
    assert train.read_bytes() == b"unverified local data"


def test_metric_backfill_recognizes_sparse_history_without_stream_markers():
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    key = "train/unit_ce_nats"
    loss = "train/update/optimization_loss_nats"
    stream = tool["TRAIN_UPDATE"]
    history = [{"update": 100, key: 2.0}, {"update": 200, key: None, loss: 3.0}]
    remote = SimpleNamespace(scan_history=lambda **kwargs: iter(history))
    points = [(stream, 100, {key: 2.0, loss: 3.0}), (stream, 200, {key: 2.1, loss: 3.0})]
    assert tool["missing_metrics"](remote, points) == {(stream, 100): {loss: 3.0}, (stream, 200): {key: 2.1}}
    with pytest.raises(ValueError, match="Conflicting source metric"):
        tool["missing_metrics"](remote, [*points, (stream, 100, {key: 3.5})])


def test_baseline_schema_v2_spec_maps_the_emitted_shared_series():
    root = Path(__file__).resolve().parents[1]
    tool = runpy.run_path(str(root / "tools/wandb_transfer.py"))
    spec = (root / "docs/WANDB_BASELINE_COMPARISON.md").read_text(encoding="utf-8")
    shared = tool["logical_train_metrics"]({"metric_schema": 2, "train/unit_ce_nats": 2.0,
                                             "train/learning_rate": 0.001, "alignment/task_loss": 3.0})
    reader = SimpleNamespace(rows=lambda *args: iter([(100, {"dev/loss": 1.5,
        **{f"dev/multitask_{task}_loss": 2.0 for task in ("source_letter", "target_letter", "decoder_target_ctc")}})]))
    _, _, dev = next(tool["fairseq_metric_rows"](reader, [], "contract", 100))
    table = [line.split("|")[2].strip().strip("`") for line in spec.splitlines() if line.startswith("| ")]
    assert {*shared, *dev, "dev/main_loss"} <= set(table)
    assert not {"train/main_loss", "train/update/optimization_loss_nats", "train/window/loss_fairseq"} & set(table)


@pytest.mark.parametrize("member", ["native-run.json", "alignment-train.jsonl", "native-dev.jsonl"])
def test_metric_modes_reject_bad_archived_member_before_network(tmp_path, member):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    work, backup = tmp_path / "work", tmp_path / "backup"
    work.mkdir()
    plan = {"contract_hash": "d" * 64, "transfer": {"method": "mask_distillation"}}
    (work / "native-run.json").write_text(json.dumps(plan), encoding="utf-8")
    for name in ("alignment-train.jsonl", "native-dev.jsonl"):
        (work / name).write_text('{"step":100}\n', encoding="utf-8")
    folder = publish_receipt(work, backup, 100, archive=True)
    archive = folder / "snapshot.zip"
    with ZipFile(archive) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    entries[member] += b" "  # A valid ZIP member whose bytes no longer match the receipt.
    with ZipFile(archive, "w") as dest:
        for name, value in entries.items():
            dest.writestr(name, value)
    args = argparse.Namespace(work=work, backup=backup, logs=tmp_path / "logs", method="mask_distillation")
    for mode in ("audit_saved_metrics", "sync_saved_metrics"):
        with pytest.raises(ValueError, match="Archived .* SHA-256 differs"):
            tool[mode](args)


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
