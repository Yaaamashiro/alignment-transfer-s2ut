"""Kokoro release, verified WAV ZIP, and pinned native preparation integration."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

from alignment_transfer_s2ut import benchmark, benchmark_data, colab_runtime, kokoro_runtime
from alignment_transfer_s2ut.data.manifest import TEXT_IDENTITY, load_manifest, text_manifest_hash, verified_teacher_subset
from alignment_transfer_s2ut.io import ArtifactStore, digest, file_hash, local_path, read_jsonl, write_json, write_jsonl
from test_benchmark_colab import benchmark_runtime  # noqa: F401
from test_notebook_commands import comparison_paths, validate_command


@pytest.mark.fairseq
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_kokoro_release_to_native_recipe(tmp_path, benchmark_runtime, monkeypatch, capsys, device):
    import torch
    pytest.importorskip("fairseq")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Kokoro native FP16 smoke requires CUDA")
    monkeypatch.delenv("CORPUS_ROOT", raising=False)
    monkeypatch.delenv("S2ST_INPUT_MAP", raising=False)
    from direct_s2st import artifacts
    from direct_s2st.s2ut import paper_recipe
    from direct_s2st.s2ut.multitask import validate_prepared
    import importlib

    extraction = importlib.import_module("direct_s2st.s2ut.extract_units")
    corpus, data = tmp_path / "corpus", tmp_path / "data"
    source, summary = corpus / "accepted.jsonl", corpus / "summary.json"
    common, units, prepared = data / "common", data / "s2ut/units", data / "s2ut/fairseq"
    rows = []
    for i, split in enumerate(("train", "train", "dev", "test")):
        row = dict(pair_id=f"p{i}", split=split, corpus="jparacrawl", qc_status="accepted",
                   ja_text=f"これは文{i}です", en_text=f"Sentence {i}.")
        if split != "test":
            for lang in ("ja", "en"):
                relative = Path("audio/16k") / lang / "shard-001" / f"p{i}.wav"
                wav = corpus / relative
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav, np.zeros(8000 + 320 * i), 16000, subtype="PCM_16")
                row.update({f"{lang}_wav_16k": "/old-vm/" + relative.as_posix(),
                            f"{lang}_sha256": file_hash(wav), f"{lang}_duration": (8000 + 320 * i) / 16000})
        rows.append(row)
    write_jsonl(source, rows)
    release = {"complete": True, "accepted_pairs": 4,
               **{split: {"accepted": n} for split, n in (("train", 2), ("dev", 1), ("test", 1))}}
    write_json(summary, release)
    lock = benchmark_data.import_kokoro(source, summary, corpus, common)
    assert lock["total_pairs"] == 3 and lock["release_pairs"] == 4
    assert read_jsonl(common / "test.jsonl") == []
    assert read_jsonl(common / "train.jsonl")[0]["source_id"] == "p0"
    assert benchmark_data.import_kokoro(source, summary, corpus, common, resume=True) == lock
    packs = tmp_path / "packs"
    assert colab_runtime.stage_kokoro(common, corpus, packs, tmp_path / "cpu", True)["status"] == "PACKED"
    (corpus / "audio").rename(corpus / "offline")
    staged = colab_runtime.stage_kokoro(common, corpus, packs, tmp_path / "gpu")
    assert staged["files"] == 6
    monkeypatch.setenv("S2ST_INPUT_MAP", staged["input_map"])
    monkeypatch.setenv("CORPUS_ROOT", str(corpus))

    def extract(wav):
        from direct_s2st.drive_staging import local_path
        frames = sf.info(local_path(wav)).frames
        return [i % 3 for i in range((frames - 400) // 320 + 1)]

    monkeypatch.setattr(artifacts, "download_artifact", lambda *a, **kw: None)
    monkeypatch.setattr(extraction, "extractor_from_config", lambda *a: extract)
    assert kokoro_runtime.prepare_units(common, units, tmp_path / "cache", 2) == {"shards": 2}
    monkeypatch.setattr(extraction, "extractor_from_config", lambda *a: pytest.fail("reran a committed shard"))
    kokoro_runtime.prepare_units(common, units, tmp_path / "cache", 2)
    prepared_lock = kokoro_runtime.prepare_s2ut(common, units, prepared)
    assert prepared_lock["splits"] == {"train": 2, "dev": 1, "test": 0}
    with pytest.raises(ValueError, match="empty split"):
        validate_prepared(prepared)
    assert benchmark_data.export_manifest(common, prepared, tmp_path / "speech.jsonl")
    assert benchmark_data.import_units(tmp_path / "speech.jsonl", units, prepared, tmp_path / "align-units")["pairs"] == 3
    _, packet = ArtifactStore(tmp_path / "align-units").get("p1")
    assert len(packet["original_units"]) == (8320 - 400) // 320 + 1

    def run_session(config, *, work, backup, **kwargs):
        from direct_s2st.s2ut import fairseq_train

        assert config["command"][2:4] == ["alignment_transfer_s2ut.kokoro_runtime", "_fairseq"]
        assert config["command"][config["command"].index("--max-tokens") + 1] == "20000"
        assert config["command"][config["command"].index("--update-freq") + 1] == "4"
        assert "--fp16" in config["command"] and kwargs["total"] == 400000
        assert kwargs["_staged"] and kwargs["restore_only"] and not kwargs["auto_resume"]
        substitutions = dict(run_root=str(work), updates="400000", python=sys.executable)
        worker = [arg.format(**substitutions) for arg in config["command"]]
        with monkeypatch.context() as mp:
            mp.setattr(sys, "argv", worker[2:])
            mp.setenv("S2ST_S2UT_PAPER_MONITOR", json.dumps(paper_recipe.report_policy()))
            mp.setenv("S2ST_S2UT_BLEU_POLICY", json.dumps(paper_recipe.bleu_policy()))
            def validate_worker_recipe():
                _, (_, metadata) = fairseq_train.prepare_research_metadata(sys.argv)
                assert metadata["checkpoint_selection"] == "teacher_forced_dev_unit_loss"
                assert metadata["max_updates"] == 400000
            mp.setattr(fairseq_train, "main", validate_worker_recipe)
            kokoro_runtime.main()
        assert (work / "s2ut-recipe-lock.json").is_file()
        best = work / "checkpoints/checkpoint_best.pt"
        best.parent.mkdir(parents=True)
        best.touch()
        return {"durable_updates": 400000}

    monkeypatch.setattr(paper_recipe, "preflight_dependencies", lambda: None)
    monkeypatch.setattr(benchmark_runtime, "run_session", run_session)
    assert kokoro_runtime.baseline_session(data, tmp_path / "run", tmp_path / "backup", True)["durable_updates"] == 400000
    verify_kokoro_comparison(tmp_path, prepared, device, capsys, benchmark_runtime)


def verify_kokoro_comparison(root, prepared, device, capsys, runtime):
    import torch
    from fairseq import tasks, utils
    from fairseq.trainer import Trainer
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    from native_plan_fixture import fixture_plan
    from test_benchmark_native import tiny_fairseq_config

    torch.set_num_threads(1)
    identity = benchmark.prepared_identity(prepared)
    assert benchmark.tsv(prepared / "test.tsv", allow_empty=True) == []
    with pytest.raises(ValueError, match="empty/duplicate"):
        benchmark.tsv(prepared / "test.tsv")
    for split in ("train", "dev"):
        tsv = prepared / f"{split}.tsv"
        content = tsv.read_bytes()
        tsv.write_bytes(content.splitlines(keepends=True)[0])
        with pytest.raises(ValueError, match=f"empty/duplicate.*{split}.tsv"):
            benchmark.prepared_identity(prepared)
        tsv.write_bytes(content)

    checkpoint = root / "tiny-baseline/checkpoints/checkpoint_best.pt"
    checkpoint.parent.mkdir(parents=True)
    # Keep all pinned auxiliary attachment layers while shrinking model widths.
    cfg = tiny_fairseq_config(prepared, checkpoint.parent, decoder_layers=3, encoder_layers=8, device=device)
    lock = {"recipe": {"fixture": True}, "command": ["direct_s2st.s2ut.fairseq_train", str(prepared)],
            "data": identity["files"]}
    fingerprint = benchmark.benchmark_digest(lock)
    lock["recipe"]["training_fingerprint"] = fingerprint
    with colab_runtime.native_inputs():
        utils.set_torch_seed(cfg.common.seed)
        task = tasks.setup_task(cfg.task)
        task.load_dataset("train")
        dataset = task.dataset("train")
        batches = [dataset.collater([dataset[i]]) for i in range(len(dataset))]
        stock = Trainer(cfg, task, task.build_model(cfg.model), task.build_criterion(cfg.criterion))
        for step in (1, 2):
            stock.train_step(batches)
            assert stock.get_num_updates() == step
        stock.consolidate_optimizer()
        stock.save_checkpoint(str(checkpoint), {"s2st_recipe_fingerprint": fingerprint})
    del stock
    recipe, metadata, baseline = root / "tiny-recipe.json", root / "tiny-metadata.json", root / "baseline.json"
    write_json(recipe, lock)
    write_json(metadata, {"training_fingerprint": fingerprint, "fairseq_revision": benchmark.REVISION,
                         "dataset_fingerprint": identity["data_lock"], "completed_updates": 2})
    record = benchmark.register_baseline(checkpoint, prepared, recipe, metadata, baseline)
    assert benchmark.read_record(baseline) == record

    manifest, units, teacher = root / "speech.jsonl", root / "align-units", root / "teacher"
    unit_store, teacher_store = ArtifactStore(units), ArtifactStore(teacher)
    spec = dict(all_layers=True, decoder_layers=3, layer_indices=[0, 1, 2], heads=None)
    for row in load_manifest(manifest):
        pair = row["pair_id"]
        meta, packet = unit_store.get(pair)
        t, s = len(row["en_text"]), len(row["ja_text"])
        attention = np.zeros((3, t, s), dtype=np.float32)
        attention[0, :, 0] = attention[1, :, s // 2] = attention[2, :, -1] = 1
        source_duration = sf.info(local_path(row["ja_audio"])).duration
        source_edges = np.linspace(0, source_duration, s + 1)
        target_edges = np.linspace(0, float(packet["duration"]), t + 1)
        teacher_store.put(pair, {**meta, **spec, "attention_spec_hash": digest(spec),
            "stage": "teacher_alignment", "s2ut_architecture_hash": digest(record["backbone"]),
            "units_hash": unit_store.hash(pair), "backend": "fairseq", "mt_checkpoint_hash": "mt-fixture",
            "tokenizer_hash": "tokenizer-fixture", "ctc_config_hash": "ctc-fixture"},
            {"mt_attention_layers": attention, "mt_attention": attention[-1],
             "source_spans": np.column_stack((source_edges[:-1], source_edges[1:])),
             "target_spans": np.column_stack((target_edges[:-1], target_edges[1:])),
             "source_duration": np.asarray(source_duration), "unit_spans": packet["unit_spans"],
             "reduced_units": packet["reduced_units"]})
    run = root / "transfer"
    fixture_plan(baseline, run, manifest=manifest, units=units, teacher=teacher, all_layers=True)
    capsys.readouterr()
    plans = {}
    for method in ("distillation", "annealed_distillation", "mask_distillation"):
        benchmark.main(["plan", "--baseline", str(baseline), "--output", str(run), "--method", method,
                        "--manifest", str(manifest), "--units", str(units), "--teacher", str(teacher),
                        "--calibration", str(run.with_suffix(".calibration.json"))])
        plan = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert plan["transfer"]["method"] == method and plan["training_budget"] == 2
        assert plan["prepared"] == identity and plan["transfer"]["all_layers"]
        plans[method] = plan
    plan = plans["annealed_distillation"]
    write_json(run / "native-run.json", plan)
    cfg = benchmark.native_config(plan["config"])
    with colab_runtime.native_inputs(), trainer_hooks(plan["contract_hash"], plan["selection_policy"]):
        task = tasks.setup_task(cfg.task)
        task.load_dataset("train")
        dataset = task.dataset("train")
        batches = [dataset.collater([dataset[i]]) for i in range(len(dataset))]
        native = train.Trainer(cfg, task, task.build_model(cfg.model), task.build_criterion(cfg.criterion))
        if device == "cuda":
            sample, _ = native._prepare_sample(batches[0])
            assert sample["net_input"]["src_tokens"].dtype == torch.float16
            assert sample["alignment_prior"].dtype == torch.float32
        native.train_step(batches)
        assert native.get_num_updates() == 1
        last = run / "checkpoints/checkpoint_last.pt"
        last.parent.mkdir()
        native.consolidate_optimizer()
        native.save_checkpoint(str(last), {})
        del native
        backup = root / "tiny-transfer-backups"
        snapshot_identity = {"contract_hash": plan["contract_hash"]}
        runtime.publish_packed(last.parent, backup, snapshot_identity, 1)
        snapshot = runtime.latest(backup, snapshot_identity)
        assert snapshot is not None and snapshot[1]["updates"] == 1
        restored = root / "tiny-transfer-restored"
        runtime.restore(snapshot, restored)
        resumed = train.Trainer(cfg, task, task.build_model(cfg.model), task.build_criterion(cfg.criterion))
        resumed.load_checkpoint(str(restored / last.name))
        assert resumed.get_num_updates() == 1
        resumed.train_step(batches)
        assert resumed.get_num_updates() == 2
        resumed.consolidate_optimizer()
        resumed.save_checkpoint(str(last), {})
        resumed.save_checkpoint(str(last.with_name("checkpoint_best.pt")), {})
    assert benchmark.train_native(plan, run, resume=True)["optimizer_steps"] == 0
    command = [sys.executable, "-m", "alignment_transfer_s2ut.benchmark", "train",
               "--output", str(run), "--method", "annealed_distillation"]
    identity = colab_runtime.identity_for(command, run)
    durable = root / "comparison-backup"
    runtime.publish_packed(run, durable, identity, 2)
    run.rename(root / "previous-vm")
    with pytest.raises(ValueError, match="identity differs"):
        colab_runtime.restore_native_training(run, durable, "distillation")
    restored = colab_runtime.restore_native_training(run, durable, "annealed_distillation")
    assert restored["updates"] == 2 and restored["status"] == "RESTORED"
    assert benchmark.register_transfer(run, run / "checkpoints/checkpoint_best.pt",
                                      root / "restored-transfer.json")["training_budget"] == 2


def test_teacher_subset_binds_original_train_and_dev_without_mutating_speech(tmp_path):
    rows = [
        dict(pair_id="a", split="train", ja_text="あ", en_text="A"),
        dict(pair_id="b", split="train", ja_text="い", en_text="B"),
        dict(pair_id="c", split="dev", ja_text="う", en_text="C"),
    ]
    source = tmp_path / "text.jsonl"
    write_jsonl(source, rows)
    saved = dict(manifest_identity=TEXT_IDENTITY, train_hash=text_manifest_hash(rows[:2]), dev_hash=text_manifest_hash(rows[2:]))
    selected = [{**rows[1], "ja_audio": "selected.wav"}, rows[2]]
    assert verified_teacher_subset(saved, selected, source) == text_manifest_hash(rows)
    assert selected[0]["ja_audio"] == "selected.wav"
    for field, value in (("split", "dev"), ("en_text", "b"), ("pair_id", "unknown")):
        with pytest.raises(ValueError, match="speech pair"):
            verified_teacher_subset(saved, [{**selected[0], field: value}], source)
    with pytest.raises(ValueError, match="original dev"):
        verified_teacher_subset({**saved, "dev_hash": "wrong"}, selected, source)


@pytest.mark.fairseq
def test_saved_sentencepiece_teacher_restores_and_extracts_only_speech_rows(tmp_path, benchmark_runtime):
    pytest.importorskip("fairseq")
    from alignment_transfer_s2ut.cli import parser_for
    from alignment_transfer_s2ut.smoke_pipeline import TINY_CONFIG
    from alignment_transfer_s2ut.training import extract_attention, train

    source, speech = tmp_path / "original/text.jsonl", tmp_path / "speech.jsonl"
    config = tmp_path / "model.json"
    write_json(config, TINY_CONFIG)
    rows = [
        dict(pair_id="a", split="train", ja_text="あい あ", en_text="a word"),
        dict(pair_id="b", split="train", ja_text="いあ い", en_text="other word"),
        dict(pair_id="c", split="dev", ja_text="あいい", en_text="a new word"),
    ]
    write_jsonl(source, rows)
    write_jsonl(speech, rows[1:])
    work = tmp_path / "training"
    command = ["--manifest", str(source), "--output", str(work), "--config", str(config),
               "--backend", "fairseq", "--tokenizer", "sentencepiece", "--max-updates", "2",
               "--source-vocab-size", "32", "--target-vocab-size", "32"]
    train(parser_for("train_t2tt").parse_args(command), "text")
    identity = dict(benchmark_revision=colab_runtime.BENCHMARK_REVISION, work=str(work),
                    command=["python", "-m", "alignment_transfer_s2ut.train_t2tt", *command])
    snapshot = benchmark_runtime.publish_packed(work, tmp_path / "snapshots", identity, 2)
    restored = tmp_path / "restored"
    evidence = kokoro_runtime.restore_teacher(snapshot, restored, 2, source, speech, config)
    assert evidence["training_updates"] == 2 and evidence["pairs"] == 2
    checkpoint = restored / "checkpoint_best.pt"
    target = tmp_path / "attention"
    args = ["--manifest", str(speech), "--checkpoint", str(checkpoint), "--output", str(target), "--all-layers"]
    parser = parser_for("extract_t2tt_attention")
    with pytest.raises(ValueError, match="train split"):
        extract_attention(parser.parse_args(args))
    result = extract_attention(parser.parse_args([*args, "--teacher-training-manifest", str(source)]))
    assert result["pairs"] == 2
    assert not ArtifactStore(target).path("a").exists()
    metadata, arrays = ArtifactStore(target).get("b")
    assert metadata["teacher_source_text_hash"] == text_manifest_hash(rows)
    assert metadata["manifest_hash"] == text_manifest_hash(rows[1:])
    assert arrays["attention_layers"].shape[0] == TINY_CONFIG["backbone"]["decoder_layers"]


def test_kokoro_attention_notebook_uses_verified_source_manifest(tmp_path):
    ns = comparison_paths(tmp_path, all_layers=True, KOKORO=True)
    commands = []
    def check(*args, **kwargs):
        validate_command(args[2], args[3:])
        commands.append(args)
        return {}
    ns.update(PYTHON="python", RUN_TEACHER_ARTIFACTS=True, config=dict(teacher_source_manifest="source.jsonl", teacher_snapshot="snapshot",
              teacher_updates=200000), KOKORO_TEACHER_MANIFEST_OVERRIDE="", KOKORO_TEACHER_SNAPSHOT_OVERRIDE="",
              MANIFEST=tmp_path/"speech.jsonl", BACKBONE_CONFIG=tmp_path/"backbone.json", execute_json=check,
              run=lambda name, module, *args, **kwargs: check("python", "-m", module, *args))
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           "notebooks/legacy/benchmark_comparison.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"] if "name='verify-kokoro-teacher'" in "".join(c["source"]))
    exec(source, ns)
    assert [c[2].rsplit(".", 1)[-1] for c in commands] == [
        "text_manifest", "kokoro_runtime", "extract_t2tt_attention", "align_speech_text"]
    assert "--all-layers" in commands[2] and "--teacher-training-manifest" in commands[2]
