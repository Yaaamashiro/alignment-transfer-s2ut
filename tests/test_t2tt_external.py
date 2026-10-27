"""Frozen historical 8C text evaluation, without remote data or model assets."""
import copy
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignment_transfer_s2ut import colab_runtime
from alignment_transfer_s2ut import evaluate_t2tt_external as external
from alignment_transfer_s2ut.io import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from alignment_transfer_s2ut.tokenization.character import SPECIAL, EOS


@pytest.fixture
def evidence(tmp_path_factory, monkeypatch):
    # Keep nested ZIP cache paths below legacy Windows MAX_PATH in local QA.
    tmp_path = tmp_path_factory.mktemp("ext")
    configured = os.environ.get("S2ST_BENCHMARK_TEST_ROOT")
    if configured:
        monkeypatch.syspath_prepend(str(Path(configured) / "src"))
        monkeypatch.setenv("PYTHONPATH", str(Path(configured) / "src") + os.pathsep +
                           str(Path(__file__).resolve().parents[1] / "src"))
    pytest.importorskip("direct_s2st")
    colab_runtime.runtime.cache_clear()
    bleu, bdigest = external.benchmark_functions()
    samples = [dict(sample_id=f"flores:devtest:{i}", flores_id=str(i), flores_split="devtest",
        fleurs_id=str(i + 500), fleurs_split="test", ja_text=f"日本語 {i}🙂",
        en_reference=f"THE old book is on table {i}!") for i in range(300)]
    for row in samples:
        row.update(ja_text_sha256=hashlib.sha256(row["ja_text"].encode()).hexdigest(),
                   en_text_sha256=hashlib.sha256(row["en_reference"].encode()).hexdigest())
    audit = dict(selected=300, selected_samples=samples, datasets=[
        dict(repository="google/fleurs", revision="a" * 40),
        dict(repository="openlanguagedata/flores_plus", revision="b" * 40)])
    plan = dict(purpose=external.PURPOSE, selected=samples, dataset=audit,
                implementation={"evaluation/bleu.py": hashlib.sha256(
                    Path(inspect.getfile(bleu)).read_bytes().replace(b"\r\n", b"\n")).hexdigest()})
    benchmark = tmp_path / "benchmark"
    root = benchmark / "diagnostics/flores-fleurs-cascade" / bdigest(plan)[:24]
    write_json(root / "identity.json", plan)
    write_json(root / "dataset-audit.json", audit)
    write_jsonl(root / "per_sample.jsonl", reversed(samples))
    write_json(root / "metrics.json", dict(purpose=external.PURPOSE, identity_sha256=bdigest(plan),
                                          metrics=dict(samples=300), per_sample=samples))
    yield SimpleNamespace(base=tmp_path, benchmark=benchmark, root=root, samples=samples)
    colab_runtime.runtime.cache_clear()


@pytest.fixture
def evaluation(evidence, monkeypatch):
    root = evidence.base
    fixed = root / "external-input/fixed-8c.json"
    external.prepare_fixed_set(evidence.benchmark, fixed)
    checkpoint = root / "t2tt/checkpoint_best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"frozen checkpoint test fixture")
    predictions = root / "text-evaluation/dev-text-predictions.jsonl"
    write_jsonl(predictions, [dict(hypothesis="dev only")])
    from sacrebleu import __version__
    dev = root / "text-evaluation/dev-text.json"
    write_json(dev, dict(split="dev", checkpoint_hash=file_hash(checkpoint),
        predictions_hash=file_hash(predictions), text_bleu=3.14,
        evaluation_contract=dict(implementation_hash=file_hash(external.text.__file__),
            decoding=dict(beam_size=5, max_chars=1000, length_penalty=1.),
            metric=dict(name="sacrebleu", version=__version__, tokenize="13a", lowercase=False,
                        smooth_method="exp", effective_order=False))))
    saved = dict(modality="text", step=19000, backend="smoke",
        backbone=dict(max_source_positions=1024, max_target_positions=1024),
        source_vocabulary=SPECIAL + list("日本語0123456789"),
        target_vocabulary=SPECIAL + list("the old book is on table.!".replace(" ", "")))
    # Fit a distinct vocabulary (repeated letters are deliberately removed).
    saved["target_vocabulary"] = SPECIAL + sorted(set("theoldbkisna.!"))
    monkeypatch.setattr(external, "load_checkpoint", lambda _: saved)
    monkeypatch.setattr(external, "checkpoint_model", lambda *_: SimpleNamespace(eval=lambda: object()))
    calls = []
    def generate(model, source, lengths, **kwargs):
        assert lengths.tolist() == [source.size(1)]
        assert kwargs == dict(beam_size=5, max_chars=1000, length_penalty=1.)
        # Generator receives tensors/source only; no reference or reference length.
        assert source.dim() == 2 and source.dtype.is_floating_point is False
        calls.append(source.tolist())
        return [saved["target_vocabulary"].index("t"), EOS], True
    monkeypatch.setattr(external.text, "generate_text", generate)
    args = SimpleNamespace(fixed_set=fixed, checkpoint=checkpoint, dev_evaluation=dev,
        output=root / "text-evaluation/flores-fleurs/flores-fleurs-text.json", device="cpu", resume=True)
    return args, calls


def test_exact_saved_set_deterministic_order_and_read_only_inputs(evidence):
    before = {p: file_hash(p) for p in evidence.root.rglob("*") if p.is_file()}
    output = evidence.base / "frozen.json"
    result = external.prepare_fixed_set(evidence.benchmark, output)
    packet = read_json(output)
    assert result["samples"] == 300
    assert {r["sample_id"] for r in packet["samples"]} == {r["sample_id"] for r in evidence.samples}
    assert [r["sample_id"] for r in packet["samples"]] == sorted(r["sample_id"] for r in evidence.samples)
    assert external.prepare_fixed_set(evidence.benchmark, output) == result
    assert all(file_hash(p) == sha for p, sha in before.items())


@pytest.mark.parametrize("change", ["duplicate", "missing", "reference", "text_sha", "bleu", "identity"])
def test_reject_changed_or_incomplete_saved_evidence(evidence, change):
    path = evidence.root / "per_sample.jsonl"
    rows = read_jsonl(path)
    if change == "duplicate":
        rows[-1] = rows[0]
    elif change == "missing":
        rows.pop()
    elif change == "reference":
        rows[0]["en_reference"] = "forged"
    elif change == "text_sha":
        rows[0]["ja_text_sha256"] = "0" * 64
    else:
        path = evidence.root / "identity.json"
        plan = read_json(path)
        if change == "bleu":
            plan["implementation"]["evaluation/bleu.py"] = "0" * 64
        else:
            plan["selected"].pop()
        write_json(path, plan, overwrite=True)
        with pytest.raises(ValueError):
            external.saved_set(evidence.root)
        return
    write_jsonl(path, rows, overwrite=True)
    with pytest.raises(ValueError):
        external.saved_set(evidence.root)


def test_missing_set_never_downloads_or_selects(tmp_path):
    with pytest.raises(ValueError, match="No completed saved 8C"):
        external.prepare_fixed_set(tmp_path, tmp_path / "frozen.json")


def test_two_completed_sets_require_explicit_identity(evidence):
    import shutil
    other = evidence.root.with_name("another")
    shutil.copytree(evidence.root, other)
    with pytest.raises(ValueError, match="Multiple completed"):
        external.prepare_fixed_set(evidence.benchmark, evidence.base / "fixed.json")
    assert external.prepare_fixed_set(evidence.benchmark, evidence.base / "fixed.json", evidence.root.name)["samples"] == 300


def test_no_reference_leakage_scores_oov_and_completed_partial_resume(evidence, evaluation, monkeypatch):
    args, calls = evaluation
    result = external.evaluate(args)
    assert len(calls) == 300 and result["checkpoint_step"] == 19000  # actual best, not final budget
    rows = read_jsonl(Path(result["predictions"]))
    assert {r["sample_id"] for r in rows} == {r["sample_id"] for r in evidence.samples}
    assert all(r["reference"] == next(s["en_reference"] for s in evidence.samples if s["sample_id"] == r["sample_id"]) for r in rows)
    assert result["source_unk_characters"] == 300 and result["samples_with_unk"] == 300
    assert result["source_characters"] == sum(len(r["ja_text"]) for r in evidence.samples)
    assert result["top_unknown_characters"] == [dict(character="🙂", count=300)]
    assert result["eos_reached_ratio"] == 1
    assert all(result[k] == v for k, v in external.score_predictions(copy.deepcopy(rows)).items())
    bleu, _ = external.benchmark_functions()
    assert result["benchmark_normalized_bleu"] == bleu.corpus_bleu(
        [bleu.normalize_english(r["hypothesis"]) for r in rows], [bleu.normalize_english(r["reference"]) for r in rows])
    cache = next((args.output.parent / (args.output.stem + "-cache")).glob("*.json"))
    assert "reference" not in read_json(cache) and "en_reference" not in read_json(cache)
    monkeypatch.setattr(external, "checkpoint_model", lambda *_: pytest.fail("resume loaded model"))
    assert external.evaluate(args) == result
    args.output.unlink()
    assert external.evaluate(args) == result
    assert len(calls) == 300
    # Mutating references cannot alter source-only generation or decoder settings.
    packet = read_json(args.fixed_set)
    for row in packet["samples"]:
        row["en_reference"] = "entirely different English reference length " * 30
        row["en_text_sha256"] = hashlib.sha256(row["en_reference"].encode()).hexdigest()
    packet["source_sample_set_hash"] = digest(packet["samples"])
    write_json(args.fixed_set, packet, overwrite=True)
    args.output = evidence.base / "different-reference/result.json"
    monkeypatch.setattr(external, "checkpoint_model", lambda *_: SimpleNamespace(eval=lambda: object()))
    external.evaluate(args)
    assert calls[:300] == calls[300:]


def test_resume_tampering_and_checkpoint_selection(evaluation):
    args, _ = evaluation
    result = external.evaluate(args)
    rows = read_jsonl(result["predictions"])
    rows.pop()
    write_jsonl(result["predictions"], rows, overwrite=True)
    with pytest.raises(ValueError, match="resume contract/predictions"):
        external.evaluate(args)
    args.checkpoint = args.checkpoint.with_name("checkpoint_last.pt")
    with pytest.raises(ValueError, match="checkpoint_best"):
        external.evaluate(args)


def test_dev_generation_contract_cannot_be_tuned_for_external_data(evaluation):
    args, calls = evaluation
    dev = read_json(args.dev_evaluation)
    dev["evaluation_contract"]["decoding"]["beam_size"] = 10
    write_json(args.dev_evaluation, dev, overwrite=True)
    with pytest.raises(ValueError, match="default generation/BLEU"):
        external.evaluate(args)
    assert not calls


def test_partial_cache_detects_modified_payload(evaluation):
    args, calls = evaluation
    external.evaluate(args)
    args.output.unlink()
    cache = next((args.output.parent / (args.output.stem + "-cache")).glob("*.json"))
    packet = read_json(cache)
    packet["hypothesis"] = "modified text"
    write_json(cache, packet, overwrite=True)
    with pytest.raises(ValueError, match="cache contract/payload"):
        external.evaluate(args)
    assert len(calls) == 300


@pytest.mark.parametrize("missing", ["metrics", "metrics_and_predictions", "argument"])
def test_absent_optional_dev_metrics_does_not_block_external_evaluation(evaluation, missing):
    args, calls = evaluation
    dev_path = args.dev_evaluation
    dev_predictions = dev_path.with_name(dev_path.stem + "-predictions.jsonl")
    dev_path.unlink()
    if missing != "metrics":
        dev_predictions.unlink()
    if missing == "argument":
        args.dev_evaluation = None
    result = external.evaluate(args)
    assert len(calls) == 300 and result["samples"] == 300
    assert result["generation"] == dict(beam_size=5, max_chars=1000, length_penalty=1.)
    assert result["dev_evaluation_available"] is False
    assert result["evaluation_contract"]["dev_evaluation_hash"] is None
    assert not dev_path.exists()  # No implicit in-domain generation/training.
    assert external.evaluate(args, completed_only=True) == result
    assert external.evaluate(args) == result and len(calls) == 300


def test_present_dev_with_missing_predictions_is_rejected(evaluation):
    args, calls = evaluation
    args.dev_evaluation.with_name(args.dev_evaluation.stem + "-predictions.jsonl").unlink()
    with pytest.raises(ValueError, match="dev evaluation must be verified"):
        external.evaluate(args)
    assert not calls


@pytest.mark.parametrize("payload", [None, [], {}])
def test_invalid_present_dev_metrics_never_fall_back_to_absent(evaluation, payload):
    args, calls = evaluation
    write_json(args.dev_evaluation, payload, overwrite=True)
    with pytest.raises(ValueError, match="dev evaluation"):
        external.evaluate(args)
    assert not calls


def test_output_stem_cannot_overwrite_existing_dev_predictions(evaluation):
    args, calls = evaluation
    dev_predictions = args.dev_evaluation.with_name(args.dev_evaluation.stem + "-predictions.jsonl")
    before = file_hash(dev_predictions)
    # A different extension still produces exactly the dev predictions filename.
    args.output = args.dev_evaluation.with_suffix(".jsonl")
    with pytest.raises(ValueError, match="overlaps protected"):
        external.evaluate(args)
    assert not calls and file_hash(dev_predictions) == before


def test_verify_is_read_only_never_generates_and_rechecks_dev_inputs(evaluation, monkeypatch):
    args, calls = evaluation
    with pytest.raises(ValueError, match="completed external evaluation required"):
        external.evaluate(args, completed_only=True)
    assert not calls and not args.output.parent.exists()
    result = external.evaluate(args)
    hashes = {p: file_hash(p) for p in args.output.parent.rglob("*") if p.is_file()}
    monkeypatch.setattr(external, "checkpoint_model", lambda *_: pytest.fail("verification loaded model"))
    assert external.evaluate(args, completed_only=True) == result
    assert len(calls) == 300 and all(file_hash(p) == sha for p, sha in hashes.items())
    dev_predictions = args.dev_evaluation.with_name(args.dev_evaluation.stem + "-predictions.jsonl")
    dev_predictions.write_text('changed dev predictions', encoding="utf-8")
    with pytest.raises(ValueError, match="dev evaluation must be verified"):
        external.evaluate(args, completed_only=True)
    assert len(calls) == 300


def test_normalization_preserves_historical_rules(evidence):
    # Exercise the actual pinned implementation, including its apostrophe rule.
    from direct_s2st.evaluation import bleu
    predictions = [dict(hypothesis="ＴＨＥ cat's, old BOOK!", reference="the cat's old book")]
    scores = external.score_predictions(predictions)
    assert scores["benchmark_normalized_bleu"] == 100.
    assert scores["text_bleu"] < scores["benchmark_normalized_bleu"]


def test_real_stage_zip_restores_partial_cache_then_skips_completed(evidence, evaluation, monkeypatch):
    """Use real pinned ZIP publication/restoration; only neural compute is stubbed."""
    import subprocess
    import sys
    args, _ = evaluation
    counter = evidence.base / "generation-calls.txt"
    script = evidence.base / "evaluate-fixture.py"
    script.write_text('''import os,sys
from pathlib import Path
from types import SimpleNamespace
from alignment_transfer_s2ut import evaluate_t2tt_external as e
from alignment_transfer_s2ut.tokenization.character import SPECIAL,EOS
args=SimpleNamespace(fixed_set=sys.argv[1],checkpoint=sys.argv[2],dev_evaluation=sys.argv[3],
    output=sys.argv[4],device='cpu',resume=True)
e.load_checkpoint=lambda _:dict(modality='text',step=19000,backend='smoke',
    backbone=dict(max_source_positions=1024,max_target_positions=1024),
    source_vocabulary=SPECIAL+list('日本語0123456789'),target_vocabulary=SPECIAL+['t'])
e.checkpoint_model=lambda *_:SimpleNamespace(eval=lambda:object())
calls=0
def generate(*a,**kw):
    global calls
    calls+=1
    if os.environ.get('EXTERNAL_TEST_INTERRUPT')=='1' and calls>100: raise SystemExit(7)
    with Path(os.environ['EXTERNAL_TEST_COUNTER']).open('a') as f: f.write('generated\\n')
    return [5,EOS],True
e.text.generate_text=generate
e.evaluate(args)
''', encoding="utf-8")
    command = [sys.executable, str(script), str(args.fixed_set), str(args.checkpoint),
               str(args.dev_evaluation), str(args.output), "--output", str(args.output)]
    backup = evidence.base / "drive/stages/t2tt-flores-fleurs"
    monkeypatch.setenv("S2ST_BACKUP_MAX_GB", "1")
    monkeypatch.setenv("EXTERNAL_TEST_COUNTER", str(counter))
    monkeypatch.setenv("EXTERNAL_TEST_INTERRUPT", "1")
    with pytest.raises(subprocess.CalledProcessError):
        colab_runtime.stage_session(command, args.output.parent, backup)
    assert len(counter.read_text().splitlines()) == 100
    assert not args.output.exists()
    args.output.parent.rename(evidence.base / "lost-partial-vm")
    monkeypatch.setenv("EXTERNAL_TEST_INTERRUPT", "0")
    colab_runtime.stage_session(command, args.output.parent, backup)
    assert len(counter.read_text().splitlines()) == 300
    original = read_json(args.output)
    args.output.parent.rename(evidence.base / "lost-completed-vm")
    assert colab_runtime.stage_session(command, args.output.parent, backup)["reused"] is True
    assert read_json(args.output) == original and len(counter.read_text().splitlines()) == 300
    assert {r["sample_id"] for r in read_jsonl(original["predictions"])} == {
        r["sample_id"] for r in evidence.samples}


@pytest.mark.parametrize("has_dev", [True, False])
@pytest.mark.parametrize("extended", [False, True])
def test_notebook_external_cell_uses_independent_stage_and_displays_existing_dev(evidence, evaluation, monkeypatch, has_dev, extended):
    import json
    import sys
    import types
    from unittest.mock import patch
    from test_notebook_commands import validate_command, comparison_paths
    args, _ = evaluation
    settings = comparison_paths(evidence.base, extended)
    if extended:
        settings["T2TT_CHECKPOINT"].parent.mkdir()
        args.checkpoint.rename(settings["T2TT_CHECKPOINT"])
        settings["T2TT_DEV_OUTPUT"].parent.mkdir(parents=True, exist_ok=True)
        args.dev_evaluation.rename(settings["T2TT_DEV_OUTPUT"])
        args.dev_evaluation.with_name(args.dev_evaluation.stem + "-predictions.jsonl").rename(
            settings["T2TT_DEV_OUTPUT"].with_name(settings["T2TT_DEV_OUTPUT"].stem + "-predictions.jsonl"))
        args.checkpoint, args.dev_evaluation, args.output = (
            settings["T2TT_CHECKPOINT"], settings["T2TT_DEV_OUTPUT"], settings["T2TT_EXTERNAL_OUTPUT"])
    if not has_dev:
        args.dev_evaluation.unlink()
        args.dev_evaluation.with_name(args.dev_evaluation.stem + "-predictions.jsonl").unlink()
    notebook = json.loads((Path(__file__).resolve().parents[1] /
        "notebooks/colab/benchmark_comparison.ipynb").read_text(encoding="utf-8"))
    source = "".join(next(c for c in notebook["cells"] if c.get("id") == "external-text-evaluate")["source"])
    calls, displays, skips = [], [], []
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = lambda rows: rows
    display_module = types.ModuleType("IPython.display")
    display_module.display = lambda frame: displays.append(frame)
    def prepare(*command, **kwargs):
        if command[2] == "alignment_transfer_s2ut.t2tt_extension":
            validate_command(command[2], command[3:])
            calls.append("status")
            return dict(completed_updates=200000)
        assert command[:3] == ("python", "-m", "alignment_transfer_s2ut.evaluate_t2tt_external")
        validate_command(command[2], command[3:])
        if command[3] == "restore-inputs":
            calls.append("restore")
            values = list(map(str, command))
            return external.restore_inputs(values[values.index("--work") + 1], values[values.index("--backup") + 1],
                int(values[values.index("--expected-updates") + 1]), values[values.index("--dev-backup") + 1],
                values[values.index("--dev-output") + 1])
        if command[3] == "prepare":
            return external.prepare_fixed_set(evidence.benchmark, args.fixed_set)
        assert command[3] == "verify" and kwargs["name"] == "verify-external-text"
        calls.append("verify")
        return external.evaluate(args, completed_only=True)
    def run(name, module, *arguments, **kwargs):
        assert name == settings["T2TT_EXTERNAL_STAGE"] and kwargs == {"gpu": True}
        validate_command(module, arguments)
        assert str(args.output) == str(arguments[arguments.index("--output") + 1])
        calls.append(name)
        return external.evaluate(args)
    namespace = dict(ROOT=str(evidence.base), BENCHMARK=str(evidence.benchmark), Path=Path, json=json,
                     PERSISTENT=str(evidence.base / "drive"), T2TT_UPDATES=20000,
                     PYTHON="python", LOGS=evidence.base / "logs", execute_json=prepare, run=run,
                     log_event=lambda *a: skips.append(a))
    namespace.update(settings)
    with patch.dict(sys.modules, {"pandas": pandas, "IPython": types.ModuleType("IPython"),
                                 "IPython.display": display_module}):
        exec(source, namespace)
        assert not calls and skips[0][1] == "skipped"
        exec(source.replace("RUN_EXTERNAL_TEXT = False", "RUN_EXTERNAL_TEXT = True"), namespace)
    assert calls == ["restore", *(["status"] if extended else []), settings["T2TT_EXTERNAL_STAGE"], "verify"]
    if has_dev:
        assert displays[0][0]["native text BLEU"] == 3.14
    else:
        assert len(displays[0]) == 1 and displays[0][0]["Dataset"] == "External FLEURS/FLORES"
        assert any(s[0] == "in-domain-dev-comparison" and s[1] == "skipped" for s in skips)
    assert len(displays[1]) == 10 and all("Reference EN" in row for row in displays[1])


@pytest.mark.parametrize("backend", ["smoke", "fairseq"])
def test_real_checkpoint_and_unmodified_generator_evaluate_all_300(evidence, backend):
    """Synthetic finite weights; real loader, tokenizer, model and beam search."""
    import torch
    from alignment_transfer_s2ut import training
    from alignment_transfer_s2ut.cli import parser_for
    from alignment_transfer_s2ut.config import BackboneConfig
    from alignment_transfer_s2ut.data.manifest import manifest_hash
    from alignment_transfer_s2ut.smoke_pipeline import TINY_CONFIG
    if backend == "fairseq":
        pytest.importorskip("fairseq")
    torch.set_num_threads(1)
    config = BackboneConfig(**TINY_CONFIG["backbone"])
    source = SPECIAL + list("日本語0123456789")
    target = SPECIAL + list("abcde")
    training.seed_all(7)
    model = training.build_model(config, "text", source, target, backend)
    norm = model.final_norm if backend == "smoke" else model.network.decoder.layer_norm
    projection = model.projection if backend == "smoke" else model.network.decoder.output_projection
    # Make this synthetic decoder emit EOS promptly, while running every real
    # encoder/decoder and the same default beam-5 generator without mocks.
    with torch.no_grad():
        norm.weight.zero_()
        norm.bias.fill_(1.)
        projection.weight.fill_(-20.)
        projection.weight[EOS].fill_(20.)
    root = evidence.base
    manifest_rows = [dict(pair_id=str(i), split=split, ja_text="日本語" + str(i), en_text="abcde" + str(i),
                          ja_audio=str(root / f"not-needed-{i}-ja.wav"), en_audio=str(root / f"not-needed-{i}-en.wav"))
                     for i, split in enumerate(("train", "dev"))]
    manifest = root / "manifest.jsonl"
    write_jsonl(manifest, manifest_rows)
    checkpoint = root / "t2tt/checkpoint_best.pt"
    checkpoint.parent.mkdir()
    torch.save(dict(modality="text", backend=backend, step=123, backbone=config.asdict(),
        source_vocabulary=source, target_vocabulary=target, model=model.state_dict(), recipe=dict(seed=7),
        train_hash=manifest_hash(manifest_rows[:1]), dev_hash=manifest_hash(manifest_rows[1:])), checkpoint)
    dev = root / "text-evaluation/dev-text.json"
    dev_args = parser_for("evaluate_t2tt").parse_args(list(map(str, [
        "--manifest", manifest, "--checkpoint", checkpoint, "--output", dev])))
    external.text.evaluate_t2tt(dev_args)
    fixed = root / "fixed-8c.json"
    external.prepare_fixed_set(evidence.benchmark, fixed)
    args = SimpleNamespace(fixed_set=fixed, checkpoint=checkpoint, dev_evaluation=dev,
        output=root / "text-evaluation/external/flores-fleurs-text.json", resume=True, device="cpu")
    result = external.evaluate(args)
    predictions = read_jsonl(result["predictions"])
    assert result["samples"] == len(predictions) == 300
    assert result["checkpoint_step"] == 123 and result["eos_reached_ratio"] == 1.
    assert all(r["hypothesis"] == "" and r["eos_reached"] for r in predictions)
    assert result["source_unk_characters"] == 300 and result["samples_with_unk"] == 300
    assert external.evaluate(args, completed_only=True) == result


@pytest.fixture
def saved_teacher(evidence):
    import torch
    root = evidence.base
    frozen = root / "frozen-training"
    frozen.mkdir()
    torch.save(dict(modality="text", step=19000, model=dict(weight=torch.ones(2))), frozen / "checkpoint_best.pt")
    torch.save(dict(modality="text", step=20000, model=dict(weight=torch.ones(2))), frozen / "checkpoint_last.pt")
    work = root / "t2tt"
    backup = root / "drive/stages/t2tt"
    identity = dict(work=str(work), benchmark_revision=colab_runtime.BENCHMARK_REVISION,
        inputs={str(root / "missing-manifest.jsonl"): "a" * 64},
        command=["python", "-m", "alignment_transfer_s2ut.train_t2tt", "--manifest", str(root / "missing-manifest.jsonl"),
                 "--output", str(work), "--max-updates", "20000"])
    bench = colab_runtime.runtime()
    snapshot = bench.publish_packed(frozen, backup, identity, 20000)
    dev_output = root / "text-evaluation/dev-text.json"
    dev_backup = root / "drive/stages/t2tt-dev"
    dev_frozen = root / "frozen-dev"
    write_jsonl(dev_frozen / "dev-text-predictions.jsonl", [dict(hypothesis="test fixture only")])
    write_json(dev_frozen / "dev-text.json", dict(checkpoint_hash=file_hash(frozen / "checkpoint_best.pt"),
        predictions_hash=file_hash(dev_frozen / "dev-text-predictions.jsonl"), text_bleu=3.14))
    dev_identity = dict(work=str(dev_output.parent), benchmark_revision=colab_runtime.BENCHMARK_REVISION,
        inputs={str(work / "checkpoint_best.pt"): file_hash(frozen / "checkpoint_best.pt")},
        command=["python", "-m", "alignment_transfer_s2ut.evaluate_t2tt", "--output", str(dev_output)])
    write_json(dev_frozen / "stage-complete.json", dict(identity=digest(dev_identity)))
    dev_snapshot = bench.publish_packed(dev_frozen, dev_backup, dev_identity, 1)
    return SimpleNamespace(root=root, work=work, backup=backup, frozen=frozen, snapshot=snapshot,
        identity=identity, dev_output=dev_output, dev_backup=dev_backup, dev_snapshot=dev_snapshot)


def test_restore_saved_teacher_and_dev_without_training_or_clobbering_external(saved_teacher, monkeypatch):
    saved = saved_teacher
    monkeypatch.setattr(colab_runtime, "train_session", lambda *a, **kw: pytest.fail("restore started training"))
    external_result = saved.dev_output.parent / "flores-fleurs/retained.json"
    write_json(external_result, dict(retain=True))
    write_json(saved.work / "unfinished.json", dict(retain=True))
    before = {p: file_hash(p) for p in (saved.root / "drive").rglob("*") if p.is_file()}
    result = external.restore_inputs(saved.work, saved.backup, 20000, saved.dev_backup, saved.dev_output)
    assert result["checkpoint_restored"] and result["dev_restored"] and result["completed_updates"] == 20000
    assert file_hash(saved.work / "checkpoint_best.pt") == file_hash(saved.frozen / "checkpoint_best.pt")
    assert external.load_checkpoint(saved.work / "checkpoint_best.pt")["step"] == 19000
    assert read_json(external_result) == dict(retain=True)
    assert list(saved.root.glob("t2tt.interrupted-*/unfinished.json"))
    assert read_json(saved.dev_output)["checkpoint_hash"] == file_hash(saved.work / "checkpoint_best.pt")
    assert not (saved.root / "missing-manifest.jsonl").exists()
    assert all(file_hash(p) == sha for p, sha in before.items())
    second = external.restore_inputs(saved.work, saved.backup, 20000, saved.dev_backup, saved.dev_output)
    assert not second["checkpoint_restored"] and not second["dev_restored"]


def test_restore_requires_existing_teacher_never_starts_new_run(evidence, monkeypatch):
    monkeypatch.setattr(colab_runtime, "train_session", lambda *a, **kw: pytest.fail("restore started training"))
    with pytest.raises(FileNotFoundError, match="no training was started"):
        external.restore_inputs(evidence.base / "t2tt", evidence.base / "no-backup", 20000)
    assert not (evidence.base / "t2tt/checkpoint_best.pt").exists()


@pytest.mark.parametrize("change", ["incomplete", "corrupt_zip", "wrong_stage", "no_best"])
def test_restore_rejects_invalid_teacher_without_last_checkpoint_fallback(saved_teacher, change):
    saved = saved_teacher
    marker = saved.snapshot / "snapshot.json"
    state = read_json(marker)
    if change == "incomplete":
        state["updates"] = 19900
        write_json(marker, state, overwrite=True)
    elif change == "corrupt_zip":
        with (saved.snapshot / "snapshot.zip").open("ab") as stream:
            stream.write(b"corrupt")
    elif change == "wrong_stage":
        state["identity"]["command"][2] = "alignment_transfer_s2ut.train_s2ut"
        write_json(marker, state, overwrite=True)
    else:
        (saved.frozen / "checkpoint_best.pt").unlink()
        state["identity"]["command"].append("--fixture-newer")
        colab_runtime.runtime().publish_packed(saved.frozen, saved.backup, state["identity"], 20001)
        # Keep the expected complete budget, but publish no best checkpoint.
        newer = max(saved.backup.glob("*/snapshot.json"))
        new_state = read_json(newer)
        new_state["updates"] = 20000
        write_json(newer, new_state, overwrite=True)
    with pytest.raises(ValueError):
        external.restore_inputs(saved.work, saved.backup, 20000)
    assert not (saved.work / "checkpoint_best.pt").exists()


def test_restore_does_not_overwrite_different_local_dev_report(saved_teacher):
    saved = saved_teacher
    write_json(saved.dev_output, dict(existing="different"))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        external.restore_inputs(saved.work, saved.backup, 20000, saved.dev_backup, saved.dev_output)
    assert read_json(saved.dev_output) == dict(existing="different")
