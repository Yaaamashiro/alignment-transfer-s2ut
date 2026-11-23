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
                                       "--backup", str(backup), "--method", "distillation",
                                       "--allow-new", "--plan-sha", file_hash(work / bleu.PLAN)], text=True)
    assert json.loads(listing)["missing_past"] == [5000, 25000]


def test_inventory_allows_a_fresh_run_only_with_explicit_opt_in(tmp_path, benchmark_runtime):
    from direct_s2st.s2ut.paper_recipe import MILESTONES

    work = tmp_path / "research/distillation"
    backup = tmp_path / "drive/distillation"
    command = [sys.executable, str(TOOL), "inventory", "--work", str(work), "--backup", str(backup),
               "--method", "distillation", "--plan-sha", "a" * 64]
    result = subprocess.check_output([*command, "--allow-new"], text=True)
    assert json.loads(result) == {"latest": 0, "available": [], "missing_past": [], "online": list(MILESTONES)}
    assert not work.exists() and not backup.exists()
    with pytest.raises(FileNotFoundError, match="No published transfer snapshots"):
        bleu.snapshots(backup, work, "distillation")
    work.mkdir(parents=True)
    backup.mkdir(parents=True)
    assert bleu.snapshots(backup, work, "distillation", allow_new=True) == {}


@pytest.mark.parametrize("relative", [
    "research/distillation/checkpoints/checkpoint_last.pt",
    "research/distillation.interrupted-1/native-run.json",
    "drive/distillation/000000000001-partial/snapshot.zip",
])
def test_inventory_refuses_fresh_start_when_training_files_remain(tmp_path, relative):
    artifact = tmp_path / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_text("preserve training evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to start from update 0"):
        bleu.snapshots(tmp_path / "drive/distillation", tmp_path / "research/distillation",
                       "distillation", allow_new=True)
    assert artifact.read_text(encoding="utf-8") == "preserve training evidence"


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
        failures = {kind: [{"pair_id": ids[0], "error": "GenerationFailure: no EOS"}] for kind in ("aux", "asr")}
        return {"update": update, "status": "completed_with_failures", "identity": {"ids": ids},
                "generation_failures": failures,
                **{f"{kind}_{key}": values[key] for kind, values in (("aux", aux), ("asr", asr))
                   for key in ("bleu", "failures", "samples")}}

    monkeypatch.setitem(bleu.evaluate.__globals__, "score_checkpoint", evaluate_model)
    args = SimpleNamespace(work=work, backup=backup, baseline_backup=None, scratch=tmp_path / "evaluation",
                           output=tmp_path / "drive/transfer-bleu", logs=tmp_path / "logs",
                           dev_sha="a" * 64, method="distillation")
    legacy_scope = digest({"plan": found[1000][1]["files"][bleu.PLAN],
                           "policy": policy, "common_dev_sha": args.dev_sha})
    legacy = args.output / "distillation" / legacy_scope / "000001000/result.json"
    write_json(legacy, {"status": "completed_with_failures", "aux_failures": 0, "asr_failures": 1})
    legacy_bytes = legacy.read_bytes()
    bleu.evaluate(args, found[1000], policy)
    target, = [path for path in args.output.rglob("result.json") if path != legacy]
    before = target.read_bytes()
    result = json.loads(before)
    assert result["method"] == "distillation" and result["aux_pair_ids"] == ids
    assert "mode" not in result
    assert result["asr_pair_ids"] == ids[:100]
    assert result["checkpoint_sha256"] == hashes[1000] and result["contract_hash"] == plan["contract_hash"]
    assert result["aux_failures"] == result["asr_failures"] == 1
    target.with_suffix(".csv").unlink()
    monkeypatch.setitem(bleu.evaluate.__globals__, "score_checkpoint",
                        lambda *args: pytest.fail("persisted score should be reused"))
    bleu.evaluate(args, found[1000], policy)
    assert target.read_bytes() == before
    with target.with_suffix(".csv").open(newline="", encoding="utf-8") as table:
        assert next(csv.DictReader(table))["checkpoint_sha256"] == hashes[1000]
    bleu.evaluate(args, found[1000], policy)
    assert legacy.read_bytes() == legacy_bytes and not legacy.with_suffix(".csv").exists()


@pytest.mark.parametrize("fault", ["asr", "vocoder", "source", "unknown_decoder", "timeout", "cuda"])
def test_benchmark_failure_classification_and_retry(published, tmp_path, monkeypatch, fault):
    from direct_s2st.s2ut import bleu_monitor as monitor
    from direct_s2st.s2ut.paper_recipe import bleu_policy
    from alignment_transfer_s2ut.io import read_jsonl, write_jsonl

    work, backup, _, hashes = published
    snapshot = bleu.snapshots(backup, work, "distillation")[1000]
    data = tmp_path / "data/s2ut/fairseq"
    data.mkdir(parents=True)
    reference = "this is a complete english sentence"
    rows = [{"pair_id": str(i), "split": "dev", "en_text": reference} for i in range(3)]
    write_jsonl(data.parents[1] / "common/dev.jsonl", rows)
    (data / "dev.tsv").write_text("id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n" +
                                 "".join(f"{i}\tx.wav\t3\t0 1\t2\n" for i in range(3)), encoding="utf-8")
    ids = [row["pair_id"] for row in monitor.selected_rows(rows, 3)]
    first, retry, bad = map(int, ids)
    seen = []
    fail = True

    def fail_at(stage, index):
        if stage == "asr":
            seen.append(index)
        if fail and index == retry:
            if stage == "aux" and fault == "cuda":
                raise torch.cuda.OutOfMemoryError("temporary CUDA fault")
            if stage == "aux" and fault == "unknown_decoder":
                raise ValueError("unexpected decoder configuration error")
            if stage == fault:
                raise OSError(f"temporary {stage} fault")

    class Dataset:
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            fail_at("source", index)
            return index

        def collater(self, items):
            return dict(id=torch.tensor(items), net_input=dict(
                src_tokens=torch.full((1, 3, 80), float(items[0])), src_lengths=torch.tensor([3])))

    def sample_index(sample):
        return int(sample["net_input"]["src_tokens"][0, 0, 0])

    def aux(model, task, sample, maximum):
        index = sample_index(sample)
        fail_at("aux", index)
        if index == bad:
            raise ValueError("auxiliary decoding reached max_aux_tokens without EOS")
        return reference

    class Vocoder:
        def dict(self, code):
            return code

        def dur_predictor(self, code):
            fail_at("vocoder", int(code[0, 0]))
            return torch.zeros_like(code, dtype=torch.float32)

        def __call__(self, **kwargs):
            return torch.full((1, 1, 12), 0.1)

    def asr(path):
        fail_at("asr", int(path.stem))
        return reference

    def generate(models, sample):
        index = sample_index(sample)
        return [[{"tokens": torch.tensor([index] if index == bad else [index, 100])}]]

    model = torch.nn.Linear(1, 1)
    dictionary = type("Dictionary", (), {"eos": lambda self: 100, "__getitem__": lambda self, key: str(key)})()
    task = SimpleNamespace(args=SimpleNamespace(data=str(data)), dataset=lambda split: Dataset(),
                           target_dictionary=dictionary, build_generator=lambda *args: SimpleNamespace(generate=generate))
    policy = dict(bleu_policy(), aux_samples=3, asr_samples=3, asr_updates=[1000])
    monkeypatch.setenv("S2ST_S2UT_BLEU_POLICY", json.dumps(policy))
    monkeypatch.setattr(monitor, "decode_characters", aux)
    monkeypatch.setattr(monitor, "load_audio_models", lambda *args: (Vocoder(), asr))
    if fault == "timeout":
        ticks = iter((0., 0., 0.))
        monkeypatch.setattr(monitor, "time", SimpleNamespace(monotonic=lambda: next(ticks, 2000.)))
    def score(plan, state, scratch, step, policy, checksum):
        return bleu.benchmark_scores(monitor, model, task, scratch, step)

    monkeypatch.setitem(bleu.evaluate.__globals__, "score_checkpoint", score)

    args = SimpleNamespace(work=work, backup=backup, baseline_backup=None, scratch=tmp_path / "eval",
                           output=tmp_path / "drive/transfer-bleu", logs=tmp_path / "logs",
                           dev_sha="a" * 64, method="distillation")
    with pytest.raises((OSError, ValueError, torch.cuda.OutOfMemoryError), match="temporary|unexpected|time budget"):
        bleu.evaluate(args, snapshot, policy)
    assert not list(args.output.rglob("result.*"))
    assert not list(args.scratch.rglob("result.json"))
    attempt, = args.output.rglob("failed-*.json")
    assert json.loads(attempt.read_bytes())["checkpoint_sha256"] == hashes[1000]
    journal, = args.scratch.rglob("asr.jsonl")
    assert [row["pair_id"] for row in read_jsonl(journal)] == [str(first)]
    fail = False
    if fault == "timeout":
        monkeypatch.setattr(monitor, "time", SimpleNamespace(monotonic=lambda: 0.))
    bleu.evaluate(args, snapshot, policy)
    target, = args.output.rglob("result.json")
    before = {path: path.read_bytes() for path in (target, target.with_suffix(".csv"))}
    result = json.loads(before[target])
    expected = monitor.score({i: {"status": "failed" if i == str(bad) else "success", "hypothesis": reference}
                              for i in ids}, dict.fromkeys(ids, reference))
    assert result["status"] == "completed_with_failures"
    for kind in ("aux", "asr"):
        assert result[f"{kind}_samples"] == 3 and result[f"{kind}_failures"] == 1
        assert result[f"{kind}_bleu"] == expected["bleu"]
        assert [row["pair_id"] for row in result["generation_failures"][kind]] == [str(bad)]
    assert seen.count(first) == 1 and seen[-1] == retry
    bleu.evaluate(args, snapshot, policy)
    assert {path: path.read_bytes() for path in before} == before
    assert file_hash(work / bleu.CHECKPOINT) == hashes[36000]


@pytest.mark.parametrize("fresh", [True, False], ids=["fresh", "resume"])
@pytest.mark.parametrize("tracked", [True, False], ids=["tracked", "untracked"])
def test_notebook_trains_then_evaluates_each_exact_online_milestone(tmp_path, fresh, tracked):
    cells, ns = namespace(tmp_path, "training")
    assert ns["TRANSFER_ONLINE_BLEU"]
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {})
    calls = []
    steps = [50000, 100000, 200000, 400000]
    if fresh:
        steps = [1000, 5000, 10000, 25000, *steps]

    def execute_json(*args, name):
        if name.startswith("plan-"):
            return dict(training_budget=400000, initialization="fresh native seed", selection_policy={},
                        contract_hash="fixture", transfer={"calibration_hash": "cal", "coefficient_policy": {}})
        assert name.startswith("bleu-inventory-") and "--plan-sha" in args and "--allow-new" in args
        return {"online": steps}

    def train(stage, module, action, *args, **kwargs):
        assert module == "alignment_transfer_s2ut.benchmark" and action == "train"
        assert "--resume" in args and "--max-update" not in args
        step = args[args.index("--stop-after-update") + 1] if "--stop-after-update" in args else 400000
        calls.append(("train", step))
        if tracked:
            assert kwargs["wandb_tracking"][2:] == ("fixture", "distillation")
            evaluation = kwargs["after_training"]
            assert evaluation[2] == "evaluate" and evaluation[evaluation.index("--update") + 1] == step
            calls.append(("eval", step))

    def execute(*args, name):
        assert not tracked, "The training reporter must own the entire segment"
        assert args[2] == "evaluate"
        calls.append(("eval", args[args.index("--update") + 1]))

    ns.update(RESEARCH_CONDITIONS=[("distillation", "KL", True)], TRANSFER_RUN_MODE="full",
              PREPARE_AUDIO_PACKS=False,
              CALIBRATION=calibration, MASK_TOP_P=.8, BASELINE=tmp_path / "baseline.json",
              MANIFEST=tmp_path / "manifest.jsonl", TEACHER_ROOT=tmp_path / "teacher",
              execute_json=execute_json, execute=execute, run=train, ENABLE_WANDB=tracked,
              prepare_wandb=lambda: ("tracker-python", "fixture-key"))
    exec(cells["transfer-training"], ns)
    assert calls == [(phase, step) for step in steps for phase in ("train", "eval")]


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


def test_sync_only_notebook_skips_evaluation_and_cuda(tmp_path, monkeypatch):
    cells, ns = namespace(tmp_path, "evaluate")
    calls = []
    colab = SimpleNamespace(userdata=SimpleNamespace(get=lambda name: "fixture-key"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    ns.update(SYNC_SAVED_TRANSFER_BLEU=True, EVALUATE_S2UT=False, ENABLE_WANDB=True,
              COMPARE_MASK_DISTILLATION=True, execute=lambda *args, **kwargs: calls.append((args, kwargs)))
    for cell in ("preflight", "restore-inputs", "bleu-backfill"):
        exec(cells[cell], ns)
    cmd, options = calls[-1]
    assert cmd[2:5] == ("--sync-bleu-only", "--method", "mask_distillation")
    assert "--bleu-results" in cmd and "--baseline-backup" not in cmd
    assert options["env"]["WANDB_API_KEY"] == "fixture-key"
    assert not any("--cuda" in cmd or "backfill" in cmd for cmd, _ in calls)
