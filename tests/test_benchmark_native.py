"""Real pinned fairseq Trainer updates with a separately trained baseline."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from test_benchmark_colab import benchmark_runtime
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("fairseq")
from alignment_transfer_s2ut import benchmark as b
from alignment_transfer_s2ut.io import ArtifactStore, digest, file_hash, read_json, write_json, write_jsonl
from alignment_transfer_s2ut.data.manifest import prepare_manifest, load_manifest, manifest_hash

pytestmark = pytest.mark.fairseq


def yaml_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


@pytest.fixture
def native_data(tmp_path, request):
    from fairseq import options, tasks, utils
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
    from fairseq.trainer import Trainer
    torch.set_num_threads(1)
    root, data = tmp_path, tmp_path / "prepared"
    data.mkdir()
    rows = []
    for i, split in enumerate(("train", "train", "dev", "test")):
        row = dict(pair_id=str(i), split=split, ja_text="あい" if i % 2 == 0 else "いあ", en_text="ab" if i % 2 == 0 else "ba")
        # Unique pairs across splits, while preserving a common character vocabulary.
        row["ja_text"] += "あ" * i
        for language in ("ja", "en"):
            path = root / f"{i}-{language}.wav"
            sf.write(path, .1 * np.sin(2 * np.pi * (200 + 30 * i) * np.arange(8000 + 160 * i) / 16000), 16000)
            row[language + "_audio"] = str(path)
        rows.append(row)
    write_jsonl(root / "input.jsonl", [r for r in rows if r["split"] != "test"])
    prepare_manifest(root / "input.jsonl", root / "manifest.jsonl")
    yaml_file(data / "config.yaml", dict(input_channels=1, input_feat_per_channel=80,
                                       transforms={"*": ["utterance_cmvn"]}))
    multi = {}
    for name, field, symbols in (("source_letter", "ja_text", ["あ", "い"]),
                                  ("target_letter", "en_text", ["a", "b"]),
                                  ("decoder_target_ctc", "en_text", ["a", "b"])):
        folder = data / name
        folder.mkdir()
        (folder / "dict.txt").write_text("".join(f"{c} 1\n" for c in symbols), encoding="utf-8")
        multi[name] = dict(data=str(folder), dict=str(folder / "dict.txt"),
            decoder_type="ctc" if name == "decoder_target_ctc" else "transformer", loss_weight=1.,
            **({"decoder_layer": 1, "zero_infinity": False} if name == "decoder_target_ctc" else {"encoder_layer": 1,
               "decoder_args": dict(decoder_layers=1, decoder_embed_dim=16, decoder_ffn_embed_dim=32, decoder_attention_heads=2, dropout=0.)}))
        for split in ("train", "dev", "test"):
            (folder / (split + ".tsv")).write_text("id\ttgt_text\n" + "".join(
                r["pair_id"] + "\t" + " ".join(r[field]) + "\n" for r in rows if r["split"] == split), encoding="utf-8")
    yaml_file(data / "config_multitask.yaml", multi)
    (data / "dict.txt").write_text("".join(f"{i} 1\n" for i in range(100)), encoding="utf-8")
    (data / "ctc.model").write_bytes(b"fixture-vocabulary-provenance")
    write_json(data / "ctc-tokenizer.json", {"model_sha256": file_hash(data / "ctc.model")})
    write_json(data / "data-lock.json", {"fixture": True, "target": "reduced_units"})
    for split in ("train", "dev", "test"):
        (data / (split + ".tsv")).write_text("id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n" + "".join(
            f"{r['pair_id']}\t{r['ja_audio']}\t{50 + int(r['pair_id'])}\t0 1 0 1 0 1\t6\n" for r in rows if r["split"] == split), encoding="utf-8")
    argv = [str(data), "--task", "speech_to_speech", "--target-is-code", "--target-code-size", "100",
        "--criterion", "speech_to_unit", "--arch", "s2ut_transformer_fisher", "--config-yaml", "config.yaml",
        "--multitask-config-yaml", "config_multitask.yaml", "--train-subset", "train", "--valid-subset", "dev",
        "--encoder-layers", "1", "--decoder-layers", str(getattr(request,"param",1)), "--encoder-embed-dim", "16", "--decoder-embed-dim", "16",
        "--encoder-ffn-embed-dim", "32", "--decoder-ffn-embed-dim", "32", "--encoder-attention-heads", "2",
        "--decoder-attention-heads", "2", "--conv-channels", "32", "--dropout", "0", "--attention-dropout", "0",
        "--relu-dropout", "0", "--share-decoder-input-output-embed", "--max-target-positions", "3000",
        "--optimizer", "adam", "--adam-betas", "(0.9,0.98)", "--lr", "0.0005", "--lr-scheduler", "inverse_sqrt",
        "--warmup-updates", "1", "--warmup-init-lr", "1e-7", "--max-update", "2", "--max-tokens", "20000", "--update-freq", "2",
        "--num-workers", "0", "--cpu", "--save-dir", str(root / "baseline/checkpoints"), "--save-interval-updates", "1"]
    cfg = convert_namespace_to_omegaconf(options.parse_args_and_arch(options.get_training_parser(), input_args=argv))
    utils.set_torch_seed(cfg.common.seed)
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    initial = deepcopy(model.state_dict())
    trainer = Trainer(cfg, task, model, criterion)
    task.load_dataset("train")
    dataset = task.dataset("train")
    samples = [dataset.collater([dataset[i]]) for i in range(2)]
    trainer.train_step(samples)
    trainer.train_step(samples)
    identity = b.prepared_identity(data)
    recipe = {"recipe": {"fixture": True}, "command": ["direct_s2st.s2ut.fairseq_train", str(data)], "data": identity["files"]}
    fingerprint = b.benchmark_digest(recipe)
    recipe["recipe"]["training_fingerprint"] = fingerprint
    write_json(root / "recipe.json", recipe)
    write_json(root / "metadata.json", {"training_fingerprint": fingerprint, "fairseq_revision": b.REVISION,
        "dataset_fingerprint": identity["data_lock"], "completed_updates": 2})
    (root / "baseline/checkpoints").mkdir(parents=True)
    trainer.consolidate_optimizer()
    trainer.save_checkpoint(str(root / "baseline/checkpoints/checkpoint_best.pt"), {"s2st_recipe_fingerprint": fingerprint})
    record = b.register_baseline(root / "baseline/checkpoints/checkpoint_best.pt", data, root / "recipe.json",
                                 root / "metadata.json", root / "baseline.json")
    from alignment_transfer_s2ut.config import BackboneConfig
    from alignment_transfer_s2ut.units.timing import frame_spans, reduce_units, run_spans
    canonical = load_manifest(root / "manifest.jsonl")
    unit_store, teacher_store = ArtifactStore(root / "units"), ArtifactStore(root / "teacher")
    config = b.backbone(record["config"])
    for row in canonical:
        wave, sr = sf.read(row["en_audio"])
        count = (len(wave) - 400) // 320 + 1
        original = np.repeat([0, 1, 0, 1, 0, 1], [4, 4, 4, 4, 4, count - 20])
        reduced, runs = reduce_units(original)
        frames = frame_spans(count, sr, len(wave) / sr, [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2])
        meta = {"split": row["split"], "manifest_hash": manifest_hash(canonical), "unit_config_hash": "unit-fixture",
                "ja_audio_hash": row["ja_sha256"], "en_audio_hash": row["en_sha256"],
                "text_hash": digest([row["ja_text"], row["en_text"]])}
        unit_store.put(row["pair_id"], meta, {"reduced_units": reduced, "unit_spans": run_spans(runs, frames),
                                            "duration": np.asarray(len(wave) / sr)})
        teacher_store.put(row["pair_id"], {**meta, "s2ut_architecture_hash": digest(config.asdict()),
            "backend": "fairseq", "units_hash": unit_store.hash(row["pair_id"]), "mt_checkpoint_hash": "mt-fixture",
            "tokenizer_hash": "tokenizer-fixture", "ctc_config_hash": "ctc-fixture"},
            {"mt_attention": np.ones((len(row["en_text"]), len(row["ja_text"]))),
             "source_spans": np.array([[0., len(wave) / sr]] * len(row["ja_text"])),
             "target_spans": np.array([[0., len(wave) / sr]] * len(row["en_text"])),
             "source_duration": np.asarray(len(wave) / sr), "unit_spans": run_spans(runs, frames), "reduced_units": reduced})
    return Namespace(root=root, cfg=cfg, record=record, task=task, samples=samples, initial=initial)


def test_native_registration_plan_updates_and_teacher_free_validation(native_data):
    from fairseq import tasks, utils
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks, AlignmentCriterion
    root = native_data.root
    plan = b.make_plan(root / "baseline.json", root / "transfer", "annealed_guidance", "t2tt",
                       root / "manifest.jsonl", root / "units", root / "teacher", anneal_steps=1)
    assert b.scientific_config(plan["config"]) == native_data.record["scientific_config"]
    assert plan["config"]["checkpoint"]["restore_file"] == "checkpoint_last.pt"
    assert plan["config"]["common"]["log_file"] == str(root / "transfer/train.log")
    write_json(root / "transfer/native-run.json", plan)
    cfg = b.native_config(plan["config"])
    utils.set_torch_seed(cfg.common.seed)
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    assert all(torch.equal(v, model.state_dict()[key]) for key, v in native_data.initial.items())
    criterion = task.build_criterion(cfg.criterion)
    assert isinstance(criterion, AlignmentCriterion)
    task.load_dataset("train")
    dataset = task.dataset("train")
    samples = [dataset.collater([dataset[i]]) for i in range(2)]
    with trainer_hooks(plan["contract_hash"]):
        trainer = train.Trainer(cfg, task, model, criterion)
        trainer.train_step(samples)
        assert trainer.get_num_updates() == 1 and model.transfer_context is None
        trainer.train_step(samples)
        assert trainer.get_num_updates() == 2
        folder = root / "transfer/checkpoints"
        folder.mkdir()
        for name in ("checkpoint_last.pt", "checkpoint_best.pt"):
            trainer.consolidate_optimizer()
            trainer.save_checkpoint(str(folder / name), {"train_iterator": {"epoch": 1, "iterations_in_epoch": 2}})
        expected_rng = torch.get_rng_state().clone()
        # Native optimizer/RNG loading uses the same fairseq Trainer path.
        restarted = train.Trainer(cfg, task, task.build_model(cfg.model), task.build_criterion(cfg.criterion))
        restarted.load_checkpoint(str(folder / "checkpoint_last.pt"))
        assert restarted.get_num_updates() == 2 and torch.equal(torch.get_rng_state(), expected_rng)
        assert all(torch.equal(v, restarted.model.state_dict()[key]) for key, v in model.state_dict().items())
        # Validation works without a prior, and never loads the teacher dataset.
        shutil.rmtree(root / "teacher")
        task.load_dataset("dev")
        dev = task.dataset("dev")
        assert "alignment_prior" not in dev.collater([dev[0]])
        trainer.valid_step(dev.collater([dev[0]]))
    transferred = b.register_transfer(root / "transfer", folder / "checkpoint_best.pt", root / "transfer.json")
    assert transferred["training_budget"] == 2
    assert b.compare_records([root / "baseline.json", root / "transfer.json"])["fair_comparison"]
    assert b.train_native(plan, root / "transfer", resume=True)["optimizer_steps"] == 0
    # Both external and transfer records use the stock native incremental generator,
    # without any training labels, reference text or teacher artifact at generation.
    from alignment_transfer_s2ut.inference.native import infer_native
    source = load_manifest(root / "manifest.jsonl", split="dev", check_audio=False)[0]
    write_jsonl(root / "source-only.jsonl", [{"pair_id": source["pair_id"], "ja_audio": source["ja_audio"]}])
    for record_name in ("baseline", "transfer"):
        args = Namespace(record=root / (record_name + ".json"), input=root / "source-only.jsonl",
            output=root / (record_name + "-predictions.jsonl"), units_only=True, vocoder_config=None,
            device="cpu", beam_size=2, max_len_a=0., max_len_b=8, length_penalty=1., resume=False, overwrite=False)
        predictions = infer_native(args)
        assert predictions[0]["inference_beta"] == 0 and predictions[0]["units"]
        args.resume = True
        assert infer_native(args) == predictions
        args.beam_size = 3
        with pytest.raises(ValueError, match="resume contract"):
            infer_native(args)


def test_native_beta_zero_matches_stock_trainer_update(native_data):
    from fairseq import tasks, utils
    from fairseq.trainer import Trainer as StockTrainer
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root = native_data.root
    plan = b.make_plan(root / "baseline.json", root / "zero", "guidance", "t2tt",
                       root / "manifest.jsonl", root / "units", root / "teacher", beta=0.)
    write_json(root / "zero/native-run.json", plan)
    cfg = b.native_config(plan["config"])
    task = tasks.setup_task(cfg.task)
    utils.set_torch_seed(cfg.common.seed)
    model = task.build_model(cfg.model)
    task.load_dataset("train")
    ds = task.dataset("train")
    samples = [ds.collater([ds[i]]) for i in range(2)]
    stock_model = native_data.task.build_model(native_data.cfg.model)
    stock_model.load_state_dict(native_data.initial)
    stock = StockTrainer(native_data.cfg, native_data.task, stock_model, native_data.task.build_criterion(native_data.cfg.criterion))
    stock.train_step(native_data.samples)
    with trainer_hooks(plan["contract_hash"]):
        guided = train.Trainer(cfg, task, model, task.build_criterion(cfg.criterion))
        guided.train_step(samples)
        assert all(torch.equal(v, model.state_dict()[key]) for key, v in stock_model.state_dict().items())


def test_native_benchmark_runtime_freezes_complete_training_state(native_data, benchmark_runtime, monkeypatch):
    from fairseq import tasks, checkpoint_utils
    from fairseq_cli import train
    from alignment_transfer_s2ut.colab_runtime import benchmark_training, native_inputs
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root = native_data.root
    stage = root / "pending"
    stage.mkdir()
    monkeypatch.setenv("S2ST_TRAIN_STAGING", str(stage))
    monkeypatch.setenv("S2ST_TRAIN_OPTIMIZE", "1")
    monkeypatch.setenv("S2ST_TRAIN_CACHE", str(root / "cache"))
    monkeypatch.setenv("S2ST_TRAIN_CACHE_GB", "0")
    monkeypatch.setenv("S2ST_BACKUP_MIN_SECONDS", "600")
    monkeypatch.setenv("S2ST_TRAIN_READERS", "1")
    monkeypatch.setenv("S2ST_TRAIN_ADAPTIVE_BATCH", "0")
    plan = b.make_plan(root / "baseline.json", root / "runtime", "guidance", "t2tt",
                       root / "manifest.jsonl", root / "units", root / "teacher", beta=0.)
    write_json(root / "runtime/native-run.json", plan)
    cfg = b.native_config(plan["config"])
    iterator = Namespace(state_dict=lambda: {"epoch": 1, "iterations_in_epoch": 1},
                         epoch=1, end_of_epoch=lambda: False)
    with native_inputs(), benchmark_training(), trainer_hooks(plan["contract_hash"]):
        task = tasks.setup_task(cfg.task)
        task.load_dataset("train")
        ds = task.dataset("train")
        samples = [ds.collater([ds[i]]) for i in range(2)]
        trainer = train.Trainer(cfg, task, task.build_model(cfg.model), task.build_criterion(cfg.criterion))
        trainer.train_step(samples)
        checkpoint_utils.save_checkpoint(cfg.checkpoint, trainer, iterator, None)
        ready = list(stage.glob("*/ready.json"))
        assert len(ready) == 1 and read_json(ready[0])["updates"] == 1
        frozen = ready[0].parent / "files"
        assert (frozen / "native-run.json").is_file()
        saved = b.load_native(frozen / "checkpoints/checkpoint_last.pt")
        assert saved["extra_state"]["alignment_contract_hash"] == plan["contract_hash"]
        assert "s2st_performance" in saved["extra_state"] and saved["last_optimizer_state"]
        from alignment_transfer_s2ut.colab_runtime import inspect_checkpoint
        assert inspect_checkpoint(frozen / "checkpoints/checkpoint_last.pt", "s2ut") == 1
        assert read_json(frozen / "gradient-audit-rank-0.json")["status"] == "PASS"


def test_native_staged_metadata_and_audio_need_no_original_payload(native_data, benchmark_runtime):
    from direct_s2st.drive_staging import register, active_map
    from direct_s2st.preparation import file_stamp
    from alignment_transfer_s2ut.colab_runtime import native_inputs
    from fairseq import tasks
    root = native_data.root
    mapping = {}
    data = root / "prepared"
    for source in data.rglob("*"):
        if source.is_file():
            target = root / "staged" / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            register(mapping, source, target, file_stamp(source))
    rows = load_manifest(root / "manifest.jsonl")
    for row in rows:
        for key in ("ja_audio", "en_audio"):
            source = Path(row[key])
            target = root / "staged-audio" / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            register(mapping, source, target, file_stamp(source))
            assert source.resolve().is_relative_to(root.resolve())
            source.rename(source.with_suffix(".offline"))
    assert data.resolve().is_relative_to(root.resolve())
    data.rename(root / "prepared-offline")
    with active_map(mapping, root / "mapping"), native_inputs():
        assert b.prepared_identity(data) == native_data.record["prepared"]
        task = tasks.setup_task(b.native_config(native_data.record["config"]).task)
        task.load_dataset("train")
        ds = task.dataset("train")
        sample = ds.collater([ds[0]])
        assert sample["net_input"]["src_tokens"].shape[-1] == 80
        assert set(sample["multitask"]) == {"source_letter", "target_letter", "decoder_target_ctc"}


def test_native_report_only_selection_and_full_dev(native_data, monkeypatch):
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root = native_data.root
    plan = b.make_plan(root / "baseline.json", root / "reported", method="baseline")
    cfg = b.native_config(plan["config"])
    task = native_data.task
    trainer_state = {"epoch": 1, "iterations_in_epoch": 1}
    iterator = Namespace(state_dict=lambda: trainer_state, epoch=1, end_of_epoch=lambda: False)
    with trainer_hooks(plan["contract_hash"], {"mode": "unit_loss_report_only", "interval": 1}):
        model = task.build_model(cfg.model)
        trainer = train.Trainer(cfg, task, model, task.build_criterion(cfg.criterion))
        trainer.train_step(native_data.samples)
        def validate(*args):
            # Main loss deliberately disagrees with an imagined total loss ranking.
            trainer.validation_main.extend([(12., 4), (3., 2)])
            torch.rand(3)
            return [999.]
        monkeypatch.setattr(train, "validate", validate)
        original_rng = torch.get_rng_state().clone()
        values, stop = train.validate_and_save(cfg, trainer, task, iterator, ["dev"], False)
        assert not stop and values[0] == pytest.approx(2.5 / np.log(2))
        assert torch.equal(original_rng, torch.get_rng_state())
        assert trainer.selection_state["best_update"] == 1
        assert (root / "reported/checkpoints/checkpoint_best.pt").is_file()


def test_native_comparison_rejects_unknown_differences(native_data):
    root = native_data.root
    other = deepcopy(native_data.record)
    other["method"] = "annealed_guidance"
    other["scientific_config"]["optimizer"]["adam_betas"] = "(0.9,0.999)"
    other["record_hash"] = digest({k: v for k, v in other.items() if k != "record_hash"})
    write_json(root / "different.json", other)
    result = b.compare_records([root / "baseline.json", root / "different.json"])
    assert not result["fair_comparison"] and "scientific_config" in result["differences"]


def test_comparison_requires_matching_generation_and_checkpoint(native_data):
    root = native_data.root
    other = deepcopy(native_data.record)
    other["method"] = "annealed_guidance"
    other["record_hash"] = digest({k: v for k, v in other.items() if k != "record_hash"})
    write_json(root / "transfer.json", other)
    contract = {"manifest_hash": "dev", "inference_config": {"beam_size": 10}, "vocoder_hash": "vocoder",
                "backend": "fairseq", "asr_config_hash": "asr", "unit_artifacts_hash": "units"}
    for name in ("baseline", "transfer"):
        write_json(root / (name + "-metrics.json"), {"checkpoint_hash": other["checkpoint_hash"],
            "evaluation_contract": {**contract, "predictions_hash": name}})
    records = [root / "baseline.json", root / "transfer.json"]
    evaluations = [root / "baseline-metrics.json", root / "transfer-metrics.json"]
    assert b.compare_records(records, evaluations)["fair_comparison"]
    value = read_json(evaluations[1])
    value["evaluation_contract"]["inference_config"]["beam_size"] = 5
    write_json(evaluations[1], value, overwrite=True)
    assert "evaluation" in b.compare_records(records, evaluations)["differences"]
    value["checkpoint_hash"] = "different model"
    write_json(evaluations[1], value, overwrite=True)
    with pytest.raises(ValueError, match="another checkpoint"):
        b.compare_records(records, evaluations)


def test_fp16_overflow_retry_preserves_batch_rng_and_buffers():
    from alignment_transfer_s2ut.s2ut.benchmark_extension import completed_step
    model = torch.nn.BatchNorm1d(2)
    scaler = Namespace(loss_scale=128.)
    optimizer = Namespace(scaler=scaler, step=lambda: None, zero_grad=lambda: None)
    trainer = Namespace(cfg=Namespace(common=Namespace(fp16=True)), optimizer=optimizer, model=model)
    updates, observed = [0], []
    trainer.get_num_updates = lambda: updates[0]
    samples = [object()]
    def step(values):
        assert values is samples
        observed.append(torch.rand(4))
        model.num_batches_tracked.add_(1)
        if len(observed) == 1:
            scaler.loss_scale /= 2
            return None
        optimizer.step()
        updates[0] += 1
        return {"completed": True}
    assert completed_step(trainer, samples, step) == {"completed": True}
    assert torch.equal(*observed) and model.num_batches_tracked.item() == 1
    def ambiguous(values):
        optimizer.step()
        scaler.loss_scale /= 2
    with pytest.raises(RuntimeError, match="refusing to skip"):
        completed_step(trainer, samples, ambiguous)


def test_registration_rejects_changed_data_and_recipe(native_data):
    root = native_data.root
    assert b.read_record(root / "baseline.json")["checkpoint_updates"] == 2
    with pytest.raises(ValueError, match="outside registered"):
        b.make_plan(root / "baseline.json", root / "prepared/transfer", method="baseline")
    metadata = read_json(root / "metadata.json")
    metadata["training_fingerprint"] = "wrong"
    write_json(root / "bad-metadata.json", metadata)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        b.register_baseline(native_data.record["checkpoint"], root / "prepared", root / "recipe.json",
                            root / "bad-metadata.json", root / "bad.json")
    with (root / "prepared/source_letter/dict.txt").open("a", encoding="utf-8") as stream:
        stream.write("新 1\n")
    with pytest.raises(ValueError, match="prepared data changed"):
        b.read_record(root / "baseline.json")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="native FP16 Trainer requires CUDA")
def test_native_fp16_trainer_and_fp32_prior(native_data):
    from fairseq import tasks
    from fairseq_cli import train
    from alignment_transfer_s2ut.s2ut.benchmark_extension import trainer_hooks
    root = native_data.root
    plan = b.make_plan(root / "baseline.json", root / "cuda", "annealed_guidance", "t2tt",
                       root / "manifest.jsonl", root / "units", root / "teacher", anneal_steps=1)
    write_json(root / "cuda/native-run.json", plan)
    cfg = b.native_config(plan["config"])
    # This is a CUDA connection test, not a parity comparison with the CPU fixture.
    cfg.common.cpu, cfg.common.fp16 = False, True
    task = tasks.setup_task(cfg.task)
    task.load_dataset("train")
    ds = task.dataset("train")
    samples = [ds.collater([ds[i]]) for i in range(2)]
    with trainer_hooks(plan["contract_hash"]):
        trainer = train.Trainer(cfg, task, task.build_model(cfg.model), task.build_criterion(cfg.criterion))
        prepared, _ = trainer._prepare_sample(samples[0])
        assert prepared["alignment_prior"].dtype == torch.float32
        assert prepared["net_input"]["src_tokens"].dtype == torch.float16
        trainer.train_step(samples)
        trainer.train_step(samples)
        assert trainer.get_num_updates() == 2 and all(torch.isfinite(p).all() for p in trainer.model.parameters())


@pytest.mark.parametrize("inline_name", [False, True])
def test_full_manifest_and_saved_unit_import(native_data, inline_name):
    from alignment_transfer_s2ut.benchmark_data import export_manifest, import_units
    root = native_data.root
    common, unit_root = root / "common", root / "native-units"
    common.mkdir()
    unit_root.mkdir()
    rows = load_manifest(root / "manifest.jsonl", check_audio=False)
    for split in ("train", "dev"):
        write_jsonl(common / (split + ".jsonl"), [r for r in rows if r["split"] == split])
    # No test manifest or audio is requested during teacher preparation.
    write_json(common / "dataset-lock.json", {"source": "fixed corpus"})
    identity = {"hubert_layer": 6, "hubert_revision": "a" * 40, "kmeans_clusters": 100, "kmeans_sha256": "b" * 64}
    records = []
    for row in rows:
        count = (sf.info(row["en_audio"]).frames - 400) // 320 + 1
        original = np.repeat([0, 1, 0, 1, 0, 1], [4, 4, 4, 4, 4, count - 20]).tolist()
        reduced = [0, 1, 0, 1, 0, 1]
        records.append({"pair_id": row["pair_id"], "split": row["split"], **identity, "units_storage": "inline-v1",
            "units_original": original, "units_reduced": reduced, "unit_count_original": count, "unit_count_reduced": 6,
            "units_sha256": b.benchmark_digest(dict(original=original, reduced=reduced))})
    manifest = unit_root / ("manifest.inline.shard-00000-of-00001.jsonl" if inline_name else "manifest.shard-00000-of-00001.jsonl")
    write_jsonl(manifest, records)
    write_json(unit_root / "unit-lock.json", identity)
    write_json(root / "prepared/data-lock.json", {"target": "reduced_units", "unit_configuration": identity,
        "common_dataset_lock_sha256": file_hash(common / "dataset-lock.json"),
        "unit_manifest_sha256": {str(manifest): file_hash(manifest)}}, overwrite=True)
    canonical = root / "exported/manifest.jsonl"
    assert len(export_manifest(common, root / "prepared", canonical)) == 3
    assert import_units(canonical, unit_root, root / "prepared", root / "imported")["pairs"] == 3
    assert import_units(canonical, unit_root, root / "prepared", root / "imported", resume=True)["pairs"] == 3
    # A modified native target must not silently become a different experiment.
    path = root / "prepared/train.tsv"
    path.write_text(path.read_text(encoding="utf-8").replace("0 1 0 1 0 1", "1 0 1 0 1 0"), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from benchmark training targets"):
        import_units(canonical, unit_root, root / "prepared", root / "bad-import")
