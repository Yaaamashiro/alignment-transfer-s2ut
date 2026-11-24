import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

from test_benchmark_native import native_data
from test_benchmark_colab import benchmark_runtime
from test_t2tt_extension import equal_state
from test_training_notebook import namespace
from test_wandb_transfer import publish_receipt, sdk
from native_plan_fixture import fixture_plan
from alignment_transfer_s2ut import benchmark
from alignment_transfer_s2ut.io import digest, read_jsonl, write_json
from alignment_transfer_s2ut.s2ut.benchmark_extension import (
    loss_sums, METRIC_AUXILIARIES, trainer_hooks, trim_v3_metrics, v3_loss_measurements,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import transfer_metrics_v3 as v3


FIXTURE = Path(__file__).parent / "fixtures/wandb_metric_schema_v3.json"


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
def test_v3_observes_real_native_training_and_dev_without_changing_updates(native_data, monkeypatch, method, tiny_sampler):
    from fairseq import tasks, utils
    from fairseq_cli import train

    root = native_data.root
    results = []
    for schema in (2, 3):
        if schema == 3:
            monkeypatch.setenv("S2ST_TRANSFER_METRIC_SCHEMA", "3")
        folder = root / f"metric-{schema}"
        plan = fixture_plan(root / "baseline.json", folder, method, "t2tt",
                            root / "manifest.jsonl", root / "units", root / "teacher")
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


def test_native_v3_pause_and_restore_keep_400k_contract(native_data, monkeypatch, tiny_sampler):
    from fairseq.data.audio.data_cfg import SingleTaskConfig

    monkeypatch.setenv("S2ST_TRANSFER_METRIC_SCHEMA", "3")
    monkeypatch.setattr(SingleTaskConfig, "get_loss_weight", lambda self, step: METRIC_AUXILIARIES[self.task_name])
    root = native_data.root
    method = "distillation"
    folder = root / ("research-v3" + v3.SCOPE) / method
    plan = fixture_plan(root / "baseline.json", folder, method, "t2tt",
                        root / "manifest.jsonl", root / "units", root / "teacher")
    plan["training_budget"] = v3.MAX_UPDATE
    plan["config"]["optimization"]["max_update"] = v3.MAX_UPDATE
    plan["selection_policy"] = {"mode": "unit_loss_report_only", "interval": 1000}
    plan["contract_hash"] = digest({key: value for key, value in plan.items() if key != "contract_hash"})
    assert benchmark.train_native(plan, folder, resume=True, stop_after_update=2)["updates"] == 2
    checkpoint = folder / "checkpoints/checkpoint_last.pt"
    saved = benchmark.load_native(checkpoint)
    assert saved["cfg"]["optimization"]["max_update"] == v3.MAX_UPDATE
    assert saved["extra_state"]["alignment_contract_hash"] == plan["contract_hash"]
    assert benchmark.train_native(plan, folder, resume=True, stop_after_update=4)["updates"] == 4
    rows = read_jsonl(folder / "alignment-train.jsonl")
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    assert all(row["metric_schema"] == 3 and "measurement_error" not in row for row in rows)
    assert not (folder / "native-dev.jsonl").exists()
    with pytest.raises(ValueError, match="reserved"):
        benchmark.train_native(plan, root / ("research-v3" + v3.OLD_SCOPE) / method, resume=True)


def row_for(update, contract, split="train"):
    _, logs = fixture_logs((1., 2.) if split == "train" else (0., 0.))
    if split == "dev":
        logs[0]["nsentences"], logs[1]["nsentences"] = 500, 953
    row = v3_loss_measurements([loss_sums(logs, training=split == "train")], split)
    if split == "train":
        for name in v3.required_metrics("distillation", update):
            key = name.replace("alignment/update/", "alignment/")
            row.setdefault(key, 0.0)
        row["alignment/lambda"] = 1.0
    return dict(row, step=update, metric_schema=3, alignment_contract_hash=contract)


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
    assert v3.coverage(dev, 1100, "distillation", contract, "dev")[1]["complete"]
    logged, result = v3.coverage(train, 1100, "distillation", contract, "train")
    assert result["complete"] and set(logged) == {1, *range(100, 1101, 100)}
    trim_v3_metrics(tmp_path, 1000)
    with train.open("a", encoding="utf-8") as stream:
        for row in rows[1000:]:
            stream.write(json.dumps(row) + "\n")
    assert v3.coverage(train, 1100, "distillation", contract, "train")[1]["complete"]
    rows[99].pop("train/main_loss")
    rows[199]["alignment/D_2"] = float("inf")
    rows[300]["step"] = 300  # Equal number of rows, one duplicate and one missing update.
    train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    bad = v3.coverage(train, 1100, "distillation", contract, "train")[1]
    assert not bad["complete"] and bad["missing_examples"] == [301]
    assert len(bad["invalid_examples"]) == 3
    dev.write_text(json.dumps(row_for(999, contract, "dev")) + "\n", encoding="utf-8")
    assert not v3.coverage(dev, 1100, "distillation", contract, "dev")[1]["complete"]


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
    write_json(work / "native-run.json", {"contract_hash": contract, "transfer": {"method": method}})
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
        assert kw["id"] == identity and kw["resume"] == "never" and kw["config"] == binding
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


@pytest.mark.parametrize("latest,stop", [(0, 1000), (1000, 1100)])
def test_colab_v3_plan_and_pause_route_without_changing_budget(tmp_path, latest, stop):
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

    ns.update(prepare_wandb=lambda: (Path("/sdk"), "secret"), execute_json=execute_json,
              execute=lambda *args, **opt: calls.append(args),
              run=lambda stage, module, *args, **kwargs: train.append((args, kwargs)))
    exec(source["transfer-training"], ns)
    assert len(calls) == 1 and "--verify-v3-plan" in calls[0]
    assert [argv[-1] for argv, _ in train] == v3.segment_targets(latest, stop)
    assert all("--max-update" not in argv and options["env"]["S2ST_TRANSFER_METRIC_SCHEMA"] == "3"
               for argv, options in train)
    assert [argv[-1] for argv, options in train if "after_training" in options] == [1000]
