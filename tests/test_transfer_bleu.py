import csv
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from alignment_transfer_s2ut import colab_runtime
from alignment_transfer_s2ut.io import digest, file_hash, write_json
from test_benchmark_colab import benchmark_runtime
from test_training_notebook import namespace


TOOL = Path(__file__).resolve().parents[1] / "tools" / "transfer_bleu.py"
bleu = SimpleNamespace(**runpy.run_path(str(TOOL)))


@pytest.fixture
def published(tmp_path, benchmark_runtime):
    work = tmp_path / "research" / "distillation"
    backup = tmp_path / "drive/stages" / "distillation"
    work.mkdir(parents=True)
    (work / "checkpoints").mkdir()
    plan = {"transfer": {"method": "distillation"}, "implementation_hash": "original-code"}
    plan["contract_hash"] = digest(plan)
    write_json(work / bleu.PLAN, plan)
    identity = {"work": str(work), "benchmark_revision": colab_runtime.BENCHMARK_REVISION,
                "command": ["python", "-m", "alignment_transfer_s2ut.benchmark", "train",
                            "--method", "distillation", "--output", str(work)]}
    hashes = {}
    for step in (1000, 10000, 36000):
        torch.save({"optimizer_history": [{"num_updates": step}], "model": {"weight": torch.ones(1)},
                    "last_optimizer_state": {"loss_scale": 128},
                    "extra_state": {"alignment_contract_hash": plan["contract_hash"], "train_iterator": {"epoch": 2},
                                    "alignment_rng": torch.get_rng_state()}}, work / bleu.CHECKPOINT)
        hashes[step] = file_hash(work / bleu.CHECKPOINT)
        benchmark_runtime.publish_packed(work, backup, identity, step)
    return work, backup, plan, hashes


def test_inventory_and_exact_readonly_zip_restore(published, tmp_path):
    work, backup, plan, hashes = published
    found = bleu.snapshots(backup, work, "distillation", file_hash(work / bleu.PLAN))
    assert bleu.inventory(found, (1000, 5000, 10000, 50000)) == {
        "latest": 36000, "available": [1000, 10000], "missing_past": [5000], "online": [50000]}
    scratch = tmp_path / "evaluation"
    scratch.mkdir()
    restored_plan, state = bleu.verified_state(work, found[1000], scratch, 1000)
    assert restored_plan == plan
    assert state["optimizer_history"][-1]["num_updates"] == 1000
    assert state["last_optimizer_state"]["loss_scale"] == 128
    assert file_hash(work / bleu.CHECKPOINT) == hashes[36000]
    assert list(scratch.iterdir()) == []
    with pytest.raises(ValueError, match="requested update"):
        bleu.verified_state(work, found[1000], scratch, 5000)
    with pytest.raises(ValueError, match="current training plan"):
        bleu.snapshots(backup, work, "distillation", "0" * 64)
    listing = subprocess.check_output([sys.executable, str(TOOL), "inventory", "--work", str(work),
                                       "--backup", str(backup), "--method", "distillation"], text=True)
    assert json.loads(listing)["missing_past"] == [5000, 25000]


def test_evaluation_resume_and_csv_recovery(published, tmp_path, monkeypatch):
    work, backup, plan, hashes = published
    found = bleu.snapshots(backup, work, "distillation")
    from direct_s2st.s2ut.bleu_monitor import score, selected_rows
    from direct_s2st.s2ut.paper_recipe import bleu_policy

    policy = bleu_policy()
    ids = [r["pair_id"] for r in selected_rows(
        [{"pair_id": str(i), "split": "dev"} for i in range(600)], 500, 1)]

    def evaluate_model(plan, state, scratch, update, policy, dev_sha):
        def result(ids_for_metric):
            references = {i: "these are four words" for i in ids_for_metric}
            predictions = {i: {"hypothesis": references[i], "status": "success"} for i in references}
            predictions[ids_for_metric[0]]["status"] = "failed"
            value = score(predictions, references)
            assert value["samples"] == len(ids_for_metric) and 0 < value["bleu"] < 100
            return value

        assert state["optimizer_history"][-1]["num_updates"] == 1000
        aux, asr = result(ids), result(ids[:100])
        return {"update": update, "status": "completed_with_failures", "identity": {"ids": ids},
                **{f"{kind}_{key}": values[key] for kind, values in (("aux", aux), ("asr", asr))
                   for key in ("bleu", "failures", "samples")}}

    monkeypatch.setitem(bleu.evaluate.__globals__, "score_checkpoint", evaluate_model)
    args = SimpleNamespace(work=work, backup=backup, baseline_backup=None, scratch=tmp_path / "evaluation",
                           output=tmp_path / "drive/transfer-bleu", logs=tmp_path / "logs",
                           dev_sha="a" * 64, method="distillation", mode="backfill")
    bleu.evaluate(args, found[1000], policy)
    target, = args.output.rglob("result.json")
    before = target.read_bytes()
    result = json.loads(before)
    assert result["method"] == "distillation" and result["aux_pair_ids"] == ids
    assert result["asr_pair_ids"] == ids[:100]
    assert result["checkpoint_sha256"] == hashes[1000] and result["contract_hash"] == plan["contract_hash"]
    assert result["aux_failures"] == result["asr_failures"] == 1
    target.with_suffix(".csv").unlink()
    monkeypatch.setitem(bleu.evaluate.__globals__, "score_checkpoint",
                        lambda *args: pytest.fail("persisted score should be reused"))
    args.mode = "online"
    bleu.evaluate(args, found[1000], policy)
    assert target.read_bytes() == before
    with target.with_suffix(".csv").open(newline="", encoding="utf-8") as table:
        assert next(csv.DictReader(table))["checkpoint_sha256"] == hashes[1000]
    bleu.evaluate(args, found[1000], policy)


def test_wandb_sync_keeps_checkpoint_axes_and_does_not_log_twice(tmp_path, monkeypatch):
    output = tmp_path / "drive"
    for update, mode in ((1000, "backfill"), (5000, "backfill"), (50000, "online")):
        row = dict(contract_hash="contract", update=update, mode=mode, status="completed",
                   checkpoint_sha256=str(update), aux_bleu=22., asr_bleu=12., aux_failures=0, asr_failures=1)
        row["result_hash"] = bleu.result_hash(row)
        write_json(output / f"mask_distillation/contract/{update}/result.json", row)

    class Run:
        summary = {}
        url = "fixture"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def define_metric(self, *args, **kwargs):
            if args == ("eval/*",):
                assert kwargs == {"step_metric": "eval/checkpoint_update"}

        def log(self, fields):
            self.summary.update(fields)
            logged.append(fields)

    logged = []
    run = Run()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run, Settings=dict))
    args = SimpleNamespace(output=output, logs=tmp_path, method="mask_distillation", contract="contract",
                           mode="backfill", project="p", entity="", group="g")
    bleu.sync_wandb(args)
    bleu.sync_wandb(args)
    assert [row["eval/checkpoint_update"] for row in logged] == [1000, 5000]
    assert logged[0]["eval/aux_bleu"] == 22. and logged[0]["eval/asr_failures"] == 1


def test_notebook_trains_then_evaluates_each_exact_online_milestone(tmp_path):
    cells, ns = namespace(tmp_path, "training")
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {})
    calls = []

    def execute_json(*args, name):
        if name.startswith("plan-"):
            return dict(training_budget=400000, initialization="fresh native seed", selection_policy={},
                        contract_hash="fixture", transfer={"calibration_hash": "cal", "coefficient_policy": {}})
        assert name.startswith("bleu-inventory-") and "--plan-sha" in args
        return {"online": [50000, 100000, 200000, 400000]}

    def train(stage, module, action, *args, **kwargs):
        assert module == "alignment_transfer_s2ut.benchmark" and action == "train"
        assert "--resume" in args and "--max-update" not in args
        step = args[args.index("--stop-after-update") + 1] if "--stop-after-update" in args else 400000
        calls.append(("train", step))

    def execute(*args, name):
        assert args[2] == "evaluate"
        calls.append(("eval", args[args.index("--update") + 1]))

    ns.update(RESEARCH_CONDITIONS=[("distillation", "KL", True)], TRANSFER_RUN_MODE="full",
              PREPARE_AUDIO_PACKS=False,
              CALIBRATION=calibration, MASK_TOP_P=.8, BASELINE=tmp_path / "baseline.json",
              MANIFEST=tmp_path / "manifest.jsonl", TEACHER_ROOT=tmp_path / "teacher",
              execute_json=execute_json, execute=execute, run=train)
    exec(cells["transfer-training"], ns)
    assert calls == [(phase, step) for step in (50000, 100000, 200000, 400000) for phase in ("train", "eval")]


def test_backfill_notebook_invokes_evaluation_only(tmp_path):
    cells, ns = namespace(tmp_path, "evaluate")
    calls = []
    ns.update(BACKFILL_TRANSFER_BLEU=True, TRANSFER_BLEU_UPDATE=25000, COMPARE_MASK_DISTILLATION=True,
              execute=lambda *args, **kwargs: calls.append(args))
    exec(cells["bleu-backfill"], ns)
    for cell in ("registration", "restore-speech-artifacts", "restore-models"):
        exec(cells[cell], ns)
    command, = calls
    assert command[2:5] == ("backfill", "--method", "mask_distillation")
    assert command[-2:] == ("--update", 25000)
