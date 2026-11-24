import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from test_benchmark_native import native_data
from test_benchmark_colab import benchmark_runtime
from test_t2tt_extension import equal_state
from test_training_notebook import namespace
from test_wandb_transfer import publish_receipt, sdk
from native_plan_fixture import fixture_plan
from alignment_transfer_s2ut import benchmark, metric_integrity_v3 as integrity
from alignment_transfer_s2ut.alignment.losses import coefficient_policy
from alignment_transfer_s2ut.io import ArtifactStore, digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from alignment_transfer_s2ut.s2ut.benchmark_extension import (
    loss_sums, METRIC_AUXILIARIES, trainer_hooks, trim_v3_metrics, v3_loss_measurements,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import transfer_metrics_v3 as v3


FIXTURE = Path(__file__).parent / "fixtures/wandb_metric_schema_v3.json"
TAU = .4


def full_layer_plan(root, folder, method):
    store = ArtifactStore(root / "teacher")
    for row in read_jsonl(root / "manifest.jsonl"):
        key = row["pair_id"]
        meta, packet = store.get(key)
        if "mt_attention_layers" not in packet:
            meta.update(all_layers=True, decoder_layers=6, layer_indices=list(range(6)))
            packet["mt_attention_layers"] = np.repeat(packet["mt_attention"][None], 6, axis=0)
            store.put(key, meta, packet, overwrite=True)
    plan = fixture_plan(root / "baseline.json", folder, method, "t2tt",
                        root / "manifest.jsonl", root / "units", root / "teacher", all_layers=True)
    if method == "mask_distillation":
        plan["transfer"]["mask_lambda"] = 10.0
    plan["transfer"]["coefficient_policy"] = coefficient_policy(plan["transfer"], plan["training_budget"])
    plan["contract_hash"] = digest({key: value for key, value in plan.items() if key != "contract_hash"})
    return plan


@pytest.fixture
def tiny_sampler(monkeypatch):
    try:
        __import__("fairseq.data.data_utils_fast")
    except ImportError:
        from fairseq.data.fairseq_dataset import FairseqDataset
        monkeypatch.setattr(FairseqDataset, "batch_by_size", lambda self, indices, **kw: [[int(i)] for i in indices])


def fixture_logs(alignment):
    fixture = json.loads(FIXTURE.read_bytes())
    logs = []
    for sample, weighted_alignment in zip(fixture["microbatches"], alignment, strict=True):
        task = sample["main_sum"] + sum(fixture["weights"][key] * loss for key, loss in sample["aux_sum"].items())
        logs.append(dict(native_main_loss_sum=torch.tensor(sample["main_sum"]), sample_size=sample["main_count"],
            nll_loss=torch.tensor(sample["nll_sum"]), ntokens=sample["main_count"], nsentences=1,
            native_task_loss_sum=task, loss=task + weighted_alignment, native_alignment_loss_sum=weighted_alignment,
            multitask={key: dict(loss=torch.tensor(loss), sample_size=sample["aux_count"][key],
                                loss_weight=fixture["weights"][key]) for key, loss in sample["aux_sum"].items()}))
    return fixture, logs


@pytest.mark.parametrize("alignment", [(0., 0.), (3., 6.)])
def test_shared_benchmark_fixture_uses_production_reducer(alignment):
    fixture, logs = fixture_logs(alignment)
    measurement = v3_loss_measurements([loss_sums(logs, training=True)], "train")
    expected = fixture["expected_nats"]
    expected["total_loss"] += sum(alignment) / 10
    for name, value in expected.items():
        assert measurement["train/" + name] == pytest.approx(value)
    assert measurement["train/unit_nll"] == fixture["expected_nll_nats"]
    assert measurement["alignment/task_loss"] == measurement["train/task_loss"]
    assert measurement["alignment/weighted_loss"] == sum(alignment) / 10
    v3.check_raw(measurement, "train")
    with pytest.raises(ValueError, match="alignment objective diagnostics"):
        v3.check_raw({**measurement, "alignment/weighted_loss": -1}, "train")
    logs[-1]["loss"] = float("nan")
    with pytest.raises(ValueError, match="invalid raw"):
        loss_sums(logs, training=True)


@pytest.mark.parametrize("method", v3.METHODS)
@pytest.mark.parametrize("native_data", [6], indirect=True)
def test_v3_observes_real_native_training_and_dev_without_changing_updates(native_data, monkeypatch, method, tiny_sampler):
    from fairseq import tasks, utils
    from fairseq_cli import train

    root = native_data.root
    monkeypatch.setattr(integrity, "DEV_SAMPLES", 1)
    results = []
    for schema in (2, 3):
        if schema == 3:
            monkeypatch.setenv("S2ST_TRANSFER_METRIC_SCHEMA", "3")
        folder = root / f"metric-{schema}"
        plan = full_layer_plan(root, folder, method)
        write_json(folder / "native-run.json", plan)
        cfg = benchmark.native_config(plan["config"])
        utils.set_torch_seed(cfg.common.seed)
        task = tasks.setup_task(cfg.task)
        for name, weight in METRIC_AUXILIARIES.items():
            task.multitask_tasks[name].args.config["loss_weight"] = weight
        task.load_dataset("train")
        task.load_dataset("dev")
        data = task.dataset("train")
        samples = [data.collater([data[i]]) for i in range(2)]
        with trainer_hooks(plan["contract_hash"], {"mode": "unit_loss_report_only", "interval": 1}):
            trainer = train.Trainer(cfg, task, task.build_model(cfg.model), task.build_criterion(cfg.criterion))
            objectives, gradients = [], []
            trainer.criterion.register_forward_hook(lambda model, args, value: objectives.append(float(value[0])))
            for param in trainer.model.parameters():
                if param.requires_grad:
                    param.register_hook(lambda value: gradients.append(value.detach().clone()))
            trainer.train_step(samples)
            rng = torch.get_rng_state().clone()
            iterator = SimpleNamespace(epoch=1, state_dict=lambda: {"epoch": 1, "iterations_in_epoch": 1},
                                       end_of_epoch=lambda: False)
            values, _ = train.validate_and_save(cfg, trainer, task, iterator, ["dev"], False)
            assert torch.equal(torch.get_rng_state(), rng)
            state = benchmark.load_native(folder / "checkpoints/checkpoint_last.pt")
            results.append((objectives, gradients, trainer.get_lr(), values, state["model"],
                            state["last_optimizer_state"], state["optimizer_history"],
                            state["extra_state"]["alignment_selection"]))
            row, = read_jsonl(folder / "alignment-train.jsonl")
            dev, = read_jsonl(folder / "native-dev.jsonl")
            if schema == 3:
                assert "measurement_error" not in row, row
                assert "measurement_error" not in dev, dev
                assert row["train/total_loss"] == pytest.approx(row["train/task_loss"] + row["alignment/weighted_loss"])
                assert row["alignment/weighted_loss"] > 0
                assert dev["dev/main_loss"] == pytest.approx(values[0] * torch.log(torch.tensor(2.)).item())
                assert dev["sample_count"] == 1 and "weighted_alignment" not in dev["loss_sums"]
    for old, new in zip(*results, strict=True):
        equal_state(old, new)


@pytest.mark.parametrize("native_data", [6], indirect=True)
def test_native_v3_pause_and_restore_keep_400k_contract(native_data, monkeypatch, tiny_sampler):
    from fairseq.data.audio.data_cfg import SingleTaskConfig

    monkeypatch.setenv("S2ST_TRANSFER_METRIC_SCHEMA", "3")
    monkeypatch.setattr(SingleTaskConfig, "get_loss_weight", lambda self, step: METRIC_AUXILIARIES[self.task_name])
    root = native_data.root
    method = "distillation"
    folder = root / ("research-v3" + v3.SCOPE) / method
    plan = full_layer_plan(root, folder, method)
    plan["training_budget"] = v3.MAX_UPDATE
    plan["config"]["optimization"]["max_update"] = v3.MAX_UPDATE
    plan["selection_policy"] = {"mode": "unit_loss_report_only", "interval": 1000}
    plan["contract_hash"] = digest({key: value for key, value in plan.items() if key != "contract_hash"})
    assert benchmark.train_native(plan, folder, resume=True, stop_after_update=2)["updates"] == 2
    checkpoint = folder / "checkpoints/checkpoint_last.pt"
    saved = benchmark.load_native(checkpoint)
    assert saved["cfg"]["optimization"]["max_update"] == v3.MAX_UPDATE
    assert saved["extra_state"]["alignment_contract_hash"] == plan["contract_hash"]
    from alignment_transfer_s2ut.s2ut import benchmark_extension

    before = file_hash(checkpoint)
    def broken_measurement(*_, **kwargs):
        raise ValueError("required auxiliary loss is missing")

    with monkeypatch.context() as failure:
        failure.setattr(benchmark_extension, "loss_sums", broken_measurement)
        with pytest.raises(ValueError, match="train measurement failed at update 3.*auxiliary"):
            benchmark.train_native(plan, folder, resume=True, stop_after_update=4)
    assert file_hash(checkpoint) == before
    originals = folder / "alignment-train.jsonl"
    assert [row["step"] for row in read_jsonl(originals)] == [1, 2]
    assert read_json(folder / "metric-stop.json")["step"] == 3

    make_row = benchmark_extension.training_metric_row

    def nonfinite_alignment(result, *args, **kwargs):
        return make_row({**result, "alignment/w_1": torch.tensor(float("nan"))}, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(benchmark_extension, "training_metric_row", nonfinite_alignment)
        with pytest.raises(ValueError, match="nonfinite or invalid metrics.*w_1"):
            benchmark.train_native(plan, folder, resume=True, stop_after_update=4)
    assert file_hash(checkpoint) == before

    valid = read_jsonl(originals)
    write_jsonl(originals, valid[1:], overwrite=True)
    with pytest.raises(ValueError, match="Incomplete Schema v3 train originals through update 2"):
        benchmark.train_native(plan, folder, resume=True, stop_after_update=4)
    assert file_hash(checkpoint) == before
    write_jsonl(originals, valid, overwrite=True)
    assert benchmark.train_native(plan, folder, resume=True, stop_after_update=4)["updates"] == 4
    rows = read_jsonl(folder / "alignment-train.jsonl")
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    assert all(row["metric_schema"] == 3 and "measurement_error" not in row for row in rows)
    assert not (folder / "native-dev.jsonl").exists()
    with pytest.raises(ValueError, match="reserved"):
        benchmark.train_native(plan, root / ("research-v3" + v3.OLD_SCOPE) / method, resume=True)


def row_for(update, contract, split="train", *, method="distillation"):
    _, logs = fixture_logs((1., 2.) if split == "train" else (0., 0.))
    if split == "dev":
        logs[0]["nsentences"], logs[1]["nsentences"] = 500, 953
    row = v3_loss_measurements([loss_sums(logs, training=split == "train")], split)
    if split == "train":
        mask = method == "mask_distillation"
        row.update({"train/learning_rate": .0005, "alignment/tau": TAU,
                    "alignment/lambda": 10.0 if mask else 1.0, "alignment/loss": .03 if mask else .3,
                    "alignment/teacher_entropy": 1., "alignment/student_entropy": 1.1,
                    "alignment/kl_mean": .3, "alignment/kl_worst": .3, "alignment/worst_layer": 0})
        for layer in range(6):
            for name, value in {"D": .03 if mask else .3, "w": 1 / 6, "valid_rows": 12,
                                "kl_layer": .3, "worst_layer_fraction": int(layer == 0)}.items():
                row[f"alignment/{name}_{layer}"] = value
            if mask:
                for name, value in {"region_mass": .97, "coverage": .4, "selected_frames": 24}.items():
                    row[f"alignment/{name}_{layer}"] = value
        if mask:
            row["alignment/top_p"] = .8
    return dict(row, step=update, metric_schema=3, alignment_contract_hash=contract)


@pytest.mark.parametrize("name,value,detail", [
    ("tau", -.4, "tau"), ("tau", .6, "tau"), ("w_0", 1.4, "w weights"),
    ("w_0", .3, "w weights"), ("valid_rows_0", 1.5, "valid_rows"),
    ("valid_rows_0", -1, "valid_rows"), ("valid_rows_0", 0, "w weights"),
    ("worst_layer", 6, "worst_layer"), ("worst_layer", 1, "worst_layer"),
    ("D_5", .5, "worst_layer"), ("worst_layer_fraction_0", .8, "worst_layer_fraction"),
    ("loss", .1, "weighted_loss"), ("region_mass_2", 1.5, "region_mass"),
    ("coverage_2", -.01, "coverage"), ("selected_frames_2", -.01, "selected_frames"),
])
def test_alignment_audit_rejects_invalid_diagnostics(tmp_path, name, value, detail):
    method, contract = "mask_distillation", "a" * 64
    originals = tmp_path / "alignment-train.jsonl"
    rows = [row_for(i, contract, method=method) for i in range(1, 4)]
    rows[-1]["alignment/" + name] = value  # Not a W&B comparison point.
    write_jsonl(originals, rows)
    _, report = v3.coverage(originals, 3, method, contract, "train", tau=TAU)
    assert not report["complete"] and "update 3" in report["invalid_examples"][0]
    assert detail in report["invalid_examples"][0]


def test_alignment_accepts_fp32_rounding_and_respects_task_only():
    method, contract = "mask_distillation", "a" * 64
    row = row_for(3, contract, method=method)
    row.update({"alignment/tau": float(np.float32(TAU)), "alignment/w_0": float(np.float32(1 / 6)),
                "alignment/loss": float(np.float32(.03)), "alignment/coverage_0": 1 + 1e-7})
    v3.validate_row(row, method, contract, "train", tau=TAU)
    values = v3.metric_values(row_for(320001, contract), "train")
    values.update({"alignment/update/lambda": 0., "alignment/update/task_only_phase": 1,
                   "alignment/update/task_only_start": 320000, "alignment/update/annealing_progress": 1.})
    with pytest.raises(ValueError, match="weighted_loss"):
        v3.check_alignment(values, "annealed_distillation", 320001, TAU)
    values["alignment/update/weighted_loss"] = 0.
    v3.check_alignment(values, "annealed_distillation", 320001, TAU)


def test_failure_fallback_cannot_publish_originals_ahead_of_checkpoint(tmp_path):
    contract = "a" * 64
    plan = {"contract_hash": contract, "transfer": {"method": "distillation",
            "calibrated": {"tau_kl": TAU}, "coefficient_policy": {"tau": TAU}}}
    write_jsonl(tmp_path / "alignment-train.jsonl", [row_for(1, contract), row_for(2, contract)])
    with pytest.raises(ValueError, match="ends at 2, expected 1"):
        integrity.checkpoint_originals(tmp_path, 1, plan, tail_only=True)
    integrity.checkpoint_originals(tmp_path, 2, plan, tail_only=True)


def test_full_original_audit_exact_points_and_1100_resume(tmp_path):
    contract = "f" * 64
    train = tmp_path / "alignment-train.jsonl"
    dev = tmp_path / "native-dev.jsonl"
    rows = [row_for(step, contract) for step in range(1, 1101)]
    train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dev.write_text(json.dumps(row_for(1000, contract, "dev")) + "\n", encoding="utf-8")
    assert v3.segment_targets(0, 1000) == [1000]
    assert v3.segment_targets(1000, 1100) == [1000, 1100]
    assert v3.segment_targets(1100, 0) == list(v3.MILESTONES[1:])
    assert v3.coverage(dev, 1100, "distillation", contract, "dev", tau=TAU)[1]["complete"]
    logged, result = v3.coverage(train, 1100, "distillation", contract, "train", tau=TAU)
    assert result["complete"] and set(logged) == {1, *range(100, 1101, 100)}
    trim_v3_metrics(tmp_path, 1000)
    with train.open("a", encoding="utf-8") as stream:
        for row in rows[1000:]:
            stream.write(json.dumps(row) + "\n")
    assert v3.coverage(train, 1100, "distillation", contract, "train", tau=TAU)[1]["complete"]
    rows[99].pop("train/main_loss")
    rows[199]["alignment/D_2"] = float("inf")
    rows[300]["step"] = 300  # Equal number of rows, one duplicate and one missing update.
    train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    bad = v3.coverage(train, 1100, "distillation", contract, "train", tau=TAU)[1]
    assert not bad["complete"] and bad["missing_examples"] == [301]
    assert len(bad["invalid_examples"]) == 3
    dev.write_text(json.dumps(row_for(999, contract, "dev")) + "\n", encoding="utf-8")
    assert not v3.coverage(dev, 1100, "distillation", contract, "dev", tau=TAU)[1]["complete"]


def test_v3_identity_and_transport_validation(tmp_path):
    method = "mask_distillation"
    args = argparse.Namespace(metric_schema=3, entity=v3.ENTITY, project=v3.PROJECT, group=v3.GROUP, method=method,
        work=tmp_path / ("research-v3" + v3.SCOPE) / method,
        backup=tmp_path / ("research-v3-" + method + v3.SCOPE))
    v3.verify_destination(args)
    key = v3.namespace("contract", args)
    assert key == v3.namespace("contract", args) and key != "contract"
    assert v3.binding("contract", args)["metric_schema"] == 3
    args.metric_schema = 2
    with pytest.raises(ValueError, match="Legacy telemetry"):
        v3.verify_destination(args)
    args.metric_schema, args.project = 3, "direct-s2st"
    with pytest.raises(ValueError, match="Schema v3"):
        v3.verify_destination(args)
    assert "alignment/update/top_p" in v3.required_metrics(method, 100)
    assert "alignment/update/task_only_start" not in v3.required_metrics("annealed_distillation", 320000)
    assert "alignment/update/task_only_start" in v3.required_metrics("annealed_distillation", 320001)

    source = {"snapshot": [("train", 100, {"train/main_loss": 2.0})]}
    target = SimpleNamespace(summary={"train": 100}, scan_history=lambda **kw: iter(()))
    with pytest.raises(RuntimeError, match="ACK_PENDING"):
        v3.reconcile_delivery(target, source, 100)
    assert v3.remote_audit(target, source, 100)["status"] == "ACK_PENDING"
    target.scan_history = lambda **kw: iter(({"update": 101, "train/main_loss": 2.0},))
    assert v3.remote_audit(target, source, 101)["status"] == "CONFLICT"
    target.scan_history = lambda **kw: iter(({"update": 100, "train/main_loss": 2.0},) * 2)
    with pytest.raises(ValueError, match="duplicate"):
        v3.reconcile_delivery(target, source, 100)
    target.scan_history = lambda **kw: iter(({"update": 100, "train/main_loss": 2.2},))
    with pytest.raises(ValueError, match="conflict"):
        v3.reconcile_delivery(target, source, 100)
    target.scan_history = lambda **kw: iter(({"update": 100, "train/main_loss": 2.0},))
    assert v3.remote_audit(target, source, 100)["status"] == "VERIFIED"


def test_v3_bleu_requires_three_values_per_qualified_subset():
    scores = {"policy": v3.BLEU_POLICY, "failure_policy": "generation-only-v1", "generation_failures": {}}
    ids = [str(i) for i in range(500)]
    for name, count in (("aux", 500), ("asr", 100)):
        scores.update({f"{name}_pair_ids": ids[:count], f"{name}_samples": count, f"{name}_failures": 0,
                       f"{name}_bleu": 0.0, f"{name}_scoring": {"signature": v3.SACREBLEU}})
        scores["generation_failures"][name] = []
    v3.check_bleu(scores)
    scores.pop("asr_samples")
    with pytest.raises(KeyError, match="asr_samples"):
        v3.check_bleu(scores)


def test_pinned_runtime_bleu_matches_registered_v3_policy(benchmark_runtime):
    from direct_s2st.s2ut.paper_recipe import bleu_policy, MILESTONES

    assert bleu_policy() == v3.BLEU_POLICY
    assert MILESTONES == v3.MILESTONES


def test_v3_cold_snapshot_sync_uses_bound_run_id_and_is_idempotent(tmp_path, monkeypatch):
    from test_wandb_transfer import CommError, RunNotFoundError

    reporter = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    method, contract = "distillation", "a" * 64
    work = tmp_path / ("research-v3" + v3.SCOPE) / method
    backup = tmp_path / ("research-v3-" + method + v3.SCOPE)
    work.mkdir(parents=True)
    write_json(work / "native-run.json", {"contract_hash": contract, "transfer": {
        "method": method, "calibrated": {"tau_kl": TAU}, "coefficient_policy": {"tau": TAU}}})
    original = work / "alignment-train.jsonl"
    original.write_text("".join(json.dumps(row_for(i, contract)) + "\n" for i in range(1, 101)), encoding="utf-8")
    receipt = publish_receipt(work, backup, 100, method=method, archive=True)
    before = (receipt / "snapshot.zip").read_bytes()
    original.unlink()
    (work / "native-run.json").unlink()
    logs = tmp_path / "logs"
    args = argparse.Namespace(work=work, backup=backup, logs=logs, method=method, metric_schema=3,
        project=v3.PROJECT, entity=v3.ENTITY, group=v3.GROUP, contract=contract, audit_v3="snapshot", final=True,
        recover_writer=True, confirm_writers_stopped=True, bleu_results=None)
    identity = reporter["run_id"](v3.namespace(contract, args))
    history, opened, summary = [], [], {}
    binding = v3.binding(contract, args)
    remote = SimpleNamespace(id=identity, state="finished", config=binding, entity=v3.ENTITY, project=v3.PROJECT,
                             group=v3.GROUP, summary=summary, scan_history=lambda **kw: iter(history))

    def lookup(path):
        assert path == f"{v3.ENTITY}/{v3.PROJECT}/{identity}"
        if not opened:
            raise CommError(RunNotFoundError(path))
        return remote

    def log(row):
        history.append(row)
        summary.update(row)

    def init(**kw):
        assert kw["id"] == identity and kw["resume"] == ("must" if opened else "never")
        assert kw.get("config") == (None if opened else binding)
        opened.append(kw["id"])
        return SimpleNamespace(log=log, summary=summary, finish=lambda: None, define_metric=lambda *a, **kw: None,
                               url="https://wandb.ai/example")

    monkeypatch.setitem(sys.modules, "wandb", sdk(init, lookup))
    monkeypatch.setitem(reporter["writer_path"].__globals__, "gettempdir", lambda: str(tmp_path / "runtime"))
    assert reporter["audit_v3"](args) is True
    assert not opened
    reporter["sync_saved_metrics"](args)
    assert [row["update"] for row in history] == [1, 100]
    assert summary["train/target_update"] == 400000 and summary["train/durable_update"] == 100
    owner = reporter["writer_path"](v3.namespace(contract, args), ".owner.json")
    assert owner.is_file()
    owner.unlink()  # A fresh runtime has only the immutable archives and the remote run.
    reporter["sync_saved_metrics"](args)
    assert owner.is_file() and opened == [identity] and len(history) == 2
    args.audit_v3 = "wandb"
    assert reporter["audit_v3"](args) is True
    assert (receipt / "snapshot.zip").read_bytes() == before

    # The high-water mark is at 100. A historical metric disappeared at 1.
    key = "train/main_loss"
    measured = history[0].pop(key)
    with pytest.raises(RuntimeError, match="ACK_PENDING"):
        reporter["sync_saved_metrics"](args)
    assert opened == [identity]

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    clock = [2000]
    monkeypatch.setattr(reporter["gap_observation"].__globals__["time"], "time", lambda: clock[0])
    args.record_v3_gaps = first
    assert not reporter["audit_v3"](args)
    args.record_v3_gaps, args.compare_v3_gaps = second, first
    clock[0] += 600
    assert not reporter["audit_v3"](args)
    args.repair_v3_gaps = (first, second)
    args.confirm_writers_stopped = False
    with pytest.raises(ValueError, match="explicit confirmation"):
        reporter["sync_saved_metrics"](args)
    args.confirm_writers_stopped = True

    history.append({"update": 1, "train/task_loss": history[0]["train/task_loss"]})
    with pytest.raises(ValueError, match="duplicate"):
        reporter["sync_saved_metrics"](args)
    history.pop()
    history[0][key] = measured + 1.0
    with pytest.raises(ValueError, match="conflict"):
        reporter["sync_saved_metrics"](args)
    history[0].pop(key)
    scans = []

    def becomes_visible(**options):
        assert options["use_cache"] is False
        scans.append(True)
        if len(scans) == 2:
            history[0][key] = measured
        return iter(history)

    remote.scan_history = becomes_visible
    with pytest.raises(RuntimeError, match="became visible"):
        reporter["sync_saved_metrics"](args)
    assert opened == [identity]
    remote.scan_history = lambda **_: iter(history)
    history[0].pop(key)
    secondary = history[0].pop("train/task_loss")  # Single observation is insufficient.
    reporter["sync_saved_metrics"](args)
    appended = history[-1]
    assert appended == {"update": 1, key: measured, reporter["TRAIN_UPDATE"]: 1}
    assert summary[reporter["TRAIN_UPDATE"]] == 100
    assert opened == [identity, identity]
    reporter["sync_saved_metrics"](args)
    assert len(history) == 3 and opened == [identity, identity]
    history[0]["train/task_loss"] = secondary
    args.record_v3_gaps = None
    assert reporter["audit_v3"](args)


@pytest.mark.parametrize("damaged", [False, True])
def test_reporter_failure_leaves_originals_verifiable(tmp_path, monkeypatch, damaged):
    reporter = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_transfer.py"))
    scope = reporter["supervise"].__wrapped__.__globals__
    method, contract = "distillation", "f" * 64
    work = tmp_path / ("research-v3" + v3.SCOPE) / method
    backup = tmp_path / ("research-v3-" + method + v3.SCOPE)
    work.mkdir(parents=True)
    args = argparse.Namespace(work=work, backup=backup, logs=tmp_path / "logs", method=method, metric_schema=3,
        project=v3.PROJECT, entity=v3.ENTITY, group=v3.GROUP, contract=contract, audit_v3="snapshot", final=True,
        originals_only=True, expected_update=1, command=["trainer"], after_training_audit=["audit"],
        after_training=["bleu"])
    events, commands = [], []

    def disconnected(*_):
        raise TimeoutError("WANDB unavailable")

    def run(command, *_, **__):
        commands.extend(command)
        if command == ["trainer"]:
            write_json(work / "native-run.json", {"contract_hash": contract, "transfer": {
                "method": method, "calibrated": {"tau_kl": TAU}, "coefficient_policy": {"tau": TAU}}})
            write_jsonl(work / "alignment-train.jsonl", [] if damaged else [row_for(1, contract)])
            publish_receipt(work, backup, 1, method=method, archive=True)
        elif command == ["audit"] and not reporter["audit_v3"](args):
            raise ValueError("archived originals failed audit")

    monkeypatch.setitem(scope, "report", disconnected)
    monkeypatch.setitem(scope, "run_process", run)
    monkeypatch.setitem(scope, "log_event", lambda *args: events.append(args))
    monkeypatch.setitem(scope, "gettempdir", lambda: str(tmp_path / "runtime"))
    if damaged:
        with pytest.raises(ValueError, match="archived originals failed audit"):
            reporter["supervise"](args)
        assert commands == ["trainer", "audit"]
    else:
        reporter["supervise"](args)
        assert commands == ["trainer", "audit", "bleu"]
    assert any(event[1] == "failed" for event in events)


@pytest.mark.parametrize("latest,stop,audit_failure", [(0, 1000, False), (1000, 1100, False), (1000, 1100, True)])
def test_colab_v3_plan_and_pause_route_without_changing_budget(tmp_path, latest, stop, audit_failure):
    source, ns = namespace(tmp_path, "training")
    exec(source["settings"].replace("TRANSFER_METRIC_PROFILE = 'legacy'", "TRANSFER_METRIC_PROFILE = 'v3'"), ns)
    assert ns["TRANSFER_EXPERIMENT_ID"] == "transfer-fixed-v2" and ns["CALIBRATION_EXPERIMENT_ID"] == "transfer-main-v1"
    ns.update(PREPARE_AUDIO_PACKS=False, RESEARCH_ROOT=tmp_path / ("research-v3" + v3.SCOPE),
              CALIBRATION=tmp_path / "calibration.json", TEACHER_ROOT=tmp_path / "teacher",
              FULL_STOP_AFTER_UPDATE=stop, RESEARCH_CONDITIONS=[("distillation", "KL", True)], os=os)
    write_json(ns["CALIBRATION"], {})
    calls, train = [], []
    plan = dict(training_budget=400000, initialization="seeded", selection_policy={}, contract_hash="a" * 64,
                transfer=dict(calibration_hash="fixture", coefficient_policy={}))

    def execute_json(*args, **options):
        if args[1] == "-m":
            return plan
        assert args[2] == "inventory" and args[-2:] == ("--stop-after-update", stop)
        return {"targets": v3.segment_targets(latest, stop)}

    def execute(*args, **_):
        calls.append(args)
        if audit_failure and "--originals-only" in args:
            raise ValueError("incomplete Snapshot originals")

    def run(stage, module, *args, **kwargs):
        train.append((args, kwargs))
        if audit := kwargs.get("after_training_audit"):
            execute(*audit)

    ns.update(prepare_wandb=lambda: (Path("/sdk"), "secret"), execute_json=execute_json, execute=execute,
              run=run)
    if audit_failure:
        with pytest.raises(ValueError, match="incomplete Snapshot"):
            exec(source["transfer-training"], ns)
        assert [argv[-1] for argv, _ in train] == [1000]
        return
    exec(source["transfer-training"], ns)
    assert "--verify-v3-plan" in calls[0]
    assert [argv[-1] for argv in calls[1:] if "--originals-only" in argv] == v3.segment_targets(latest, stop)
    assert [argv[-1] for argv, _ in train] == v3.segment_targets(latest, stop)
    assert all("--max-update" not in argv and options["env"]["S2ST_TRANSFER_METRIC_SCHEMA"] == "3"
               for argv, options in train)
    assert [argv[-1] for argv, options in train if "after_training" in options] == [1000]
