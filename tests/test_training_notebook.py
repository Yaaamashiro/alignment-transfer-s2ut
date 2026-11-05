"""Execute the primary training/evaluation notebook entry points without GPUs."""
import ast
import json
from pathlib import Path
from types import ModuleType

import pytest

from alignment_transfer_s2ut.io import write_json
from test_notebook_commands import validate_command

ROOT = Path(__file__).resolve().parents[1]


def cells(name):
    notebook = json.loads((ROOT / f"notebooks/colab/{name}.ipynb").read_text(encoding="utf-8"))
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}


def namespace(tmp_path, name):
    source = cells(name)
    ns = {"Path": Path, "json": json}
    exec(source["settings"], ns)
    exec(source["paths"], ns)
    ns.update(ROOT=str(tmp_path), PERSISTENT=str(tmp_path / "drive"), LOGS=tmp_path / "logs",
              config={"teacher_source_manifest": str(tmp_path / "original.jsonl")})
    ns["log_event"] = lambda *a, **kw: None
    return source, ns


def test_primary_notebooks_have_canonical_logging_and_no_raw_subprocesses():
    canonical = ast.parse((ROOT / "src/alignment_transfer_s2ut/process_logging.py").read_text(encoding="utf-8"))
    expected = {n.name: ast.dump(n) for n in canonical.body if isinstance(n, ast.FunctionDef)}
    for name in ("training", "evaluate"):
        source = cells(name)
        actual = {n.name: ast.dump(n) for n in ast.parse(source["process-logging"]).body if isinstance(n, ast.FunctionDef)}
        assert all(actual[key] == value for key, value in expected.items())
        for identity, code in source.items():
            if identity == "process-logging":
                continue
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert not (isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess")


def test_training_cpu_pack_mode_skips_all_model_training(tmp_path):
    source, ns = namespace(tmp_path, "training")
    ns.update(NEED_TEACHER=True, NEED_TEXT=True, RUN_BASELINE=True, RUN_T2TT=True)
    calls = []
    ns.update(execute=lambda *a, **kw: calls.append(a), execute_json=lambda *a, **kw: calls.append(a))
    for identity in ("text-paths", "text-input", "text-train", "text-restore", "text-evaluate",
                     "text-examples", "baseline", "registration", "speech-artifacts", "teacher-paths",
                     "teacher-extraction", "teacher-projection", "calibration", "transfer-training",
                     "register-transfers", "layer-diagnostics"):
        exec(source[identity], ns)
    assert calls == []


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_audio_pack_workers_reach_only_cpu_publication(tmp_path, monkeypatch, workers):
    source, ns = namespace(tmp_path, 'training')
    monkeypatch.setenv('S2ST_INPUT_MAP', '')
    ns.update(AUDIO_PACK_WORKERS=workers, CORPUS_ROOT=tmp_path/'corpus')
    calls = []
    def execute_json(*args, **kw):
        validate_command(args[2], args[3:])
        calls.append(list(map(str, args)))
        return {'input_map': str(tmp_path/'input-map.json')}
    ns['execute_json'] = execute_json
    exec(source['helpers'], ns)
    ns['corpus_stage_audio'](create=True)
    assert calls[-1][calls[-1].index('--audio-pack-workers') + 1] == str(workers)
    assert '--create-audio' in calls[-1]
    ns['corpus_stage_audio'](create=False)
    assert '--audio-pack-workers' not in calls[-1] and '--create-audio' not in calls[-1]


def test_primary_notebooks_install_the_tested_audio_pack_runtime():
    from alignment_transfer_s2ut.colab_runtime import BENCHMARK_RUNTIME_REVISION
    for name in ('training', 'evaluate'):
        ns = {'Path': Path}
        exec(cells(name)['settings'], ns)
        exec(cells(name)['paths'], ns)
        assert ns['BENCHMARK_RUNTIME_REVISION'] == BENCHMARK_RUNTIME_REVISION


@pytest.mark.parametrize("tokenizer", ["character", "sentencepiece"])
def test_training_text_stage_preserves_its_independent_paths_and_commands(tmp_path, tokenizer):
    source, ns = namespace(tmp_path, "training")
    ns.update(PREPARE_AUDIO_PACKS=False, NEED_TEXT=True, T2TT_TOKENIZER=tokenizer,
              T2TT_SOURCE_MANIFEST_OVERRIDE=str(tmp_path / "text.jsonl"), RUN_T2TT=True)
    exec(source["text-paths"], ns)
    calls = []
    def execute(*args, **kw):
        args = list(map(str, args))
        validate_command(args[2], args[3:])
        calls.append(args)
    ns["execute"] = execute
    exec(source["text-train"], ns)
    cmd = calls[0]
    start = cmd.index("alignment_transfer_s2ut.train_t2tt")
    validate_command(cmd[start], cmd[start + 1:])
    assert cmd[cmd.index("--tokenizer") + 1] == tokenizer
    assert ns["TEXT_ROOT"].name == ns["T2TT_RUN_ID"]
    assert str(ns["TEXT_WORK"]) != str(ns["RESTORED"])


@pytest.mark.parametrize("include_transfer", [None, True])
def test_evaluation_restores_models_and_scores_without_training(tmp_path, monkeypatch, include_transfer):
    source, ns = namespace(tmp_path, "evaluate")
    calls = []
    ns["RESEARCH_ROOT"] = tmp_path / "research"
    ns["BASELINE"] = tmp_path / "baseline.json"
    ns["MANIFEST"] = tmp_path / "manifest.jsonl"
    if include_transfer is not None:
        for name in ("DISTILLATION", "ANNEALED_DISTILLATION", "MASK_DISTILLATION"):
            ns["COMPARE_" + name] = include_transfer
    def execute_json(*args, **kw):
        validate_command(args[2], args[3:])
        calls.append(list(map(str, args)))
        folder = Path(args[args.index("--work") + 1])
        (folder / "checkpoints").mkdir(parents=True)
        (folder / "checkpoints/checkpoint_best.pt").touch()
        return {"status": "RESTORED"}
    def run(name, module, *args, **kw):
        validate_command(module, args)
        if module == "alignment_transfer_s2ut.benchmark" and args[0] == "compare":
            assert len(args[1:args.index("--output")]) >= 2
        calls.append(["python", "-m", module, *map(str, args)])
        if module == "alignment_transfer_s2ut.evaluate":
            write_json(Path(args[args.index("--output") + 1]), {"unit_error_rate": .2, "asr_bleu": None})
    display = ModuleType("IPython.display")
    display.Markdown = str
    display.display = lambda *a: None
    monkeypatch.setitem(__import__("sys").modules, "IPython.display", display)
    ns.update(execute_json=execute_json, run=run)
    exec(source["restore-models"], ns)
    exec(source["evaluate"], ns)
    assert len([c for c in calls if c[2:4] == ["alignment_transfer_s2ut.colab_runtime", "restore-training"]]) == (3 if include_transfer else 0)
    assert len([c for c in calls if c[2:4] == ["alignment_transfer_s2ut.benchmark", "infer"]]) == (4 if include_transfer else 1)
    assert not any(c[2:4] == ["alignment_transfer_s2ut.benchmark", "train"] for c in calls)
    assert not any("alignment_transfer_s2ut.train_t2tt" in c for c in calls)


def test_evaluation_preprocessing_is_restore_only(tmp_path):
    source, ns = namespace(tmp_path, "evaluate")
    calls = []
    ns.update(corpus_common=lambda **kw: calls.append(kw),
              corpus_stage_audio=lambda: None, corpus_prepare=lambda **kw: calls.append(kw),
              corpus_baseline=lambda **kw: calls.append(kw))
    exec(source["restore-inputs"], ns)
    assert calls == [{"restore_only": True}] * 3


def test_training_routes_teacher_and_all_transfer_conditions_to_the_same_inputs(tmp_path):
    source, ns = namespace(tmp_path, "training")
    ns.update(PREPARE_AUDIO_PACKS=False, NEED_TEACHER=True, NEED_TEXT=True,
              RUN_TEACHER_ARTIFACTS=True,
              RUN_DISTILLATION=True, RUN_ANNEALED_DISTILLATION=True, RUN_MASK_DISTILLATION=True,
              MASK_TOP_P=.8, BASELINE=tmp_path / "baseline.json", MANIFEST=tmp_path / "speech.jsonl",
              BACKBONE_CONFIG=tmp_path / "backbone.json", T2TT_SOURCE_MANIFEST_OVERRIDE="")
    exec(source["text-paths"], ns)
    ns["TEXT_BACKUP"] = tmp_path / "text-backup"
    ns["TEXT_DEV_METRICS"] = tmp_path / "dev-text.json"
    ns["status"] = {"checkpoint_hash": "fixture"}
    write_json(ns["TEXT_DEV_METRICS"], ns["status"])
    marker = ns["TEXT_BACKUP"] / "complete/snapshot.json"
    write_json(marker, {"complete": True})
    exec(source["teacher-paths"], ns)
    ns["CALIBRATION"] = tmp_path / "calibration.json"
    write_json(ns["CALIBRATION"], {"fixture": True})
    calls = []
    def execute_json(*args, **kw):
        validate_command(args[2], args[3:])
        calls.append(list(map(str, args)))
        return dict(training_budget=400000, initialization="fresh", selection_policy={},
                    transfer={"calibration_hash": "fixture"})
    def run(name, module, *args, **kw):
        validate_command(module, args)
        calls.append(["python", "-m", module, *map(str, args)])
    ns.update(execute_json=execute_json, run=run)
    exec(source["teacher-extraction"], ns)
    exec(source["teacher-projection"], ns)
    exec(source["transfer-training"], ns)
    training = [c for c in calls if c[2:4] == ["alignment_transfer_s2ut.benchmark", "train"]]
    assert len(training) == 3
    assert [c[c.index("--method") + 1] for c in training] == [
        "distillation", "annealed_distillation", "mask_distillation"]
    assert all(c[c.index("--baseline") + 1] == str(ns["BASELINE"]) for c in training)
    assert all(c[c.index("--teacher") + 1] == str(ns["TEACHER_ROOT"]) for c in training)
    attention = next(c for c in calls if c[2] == "alignment_transfer_s2ut.extract_t2tt_attention")
    assert "--teacher-training-manifest" in attention and "--all-layers" in attention


def test_external_text_evaluation_skips_speech_restoration(tmp_path):
    source, ns = namespace(tmp_path, "evaluate")
    ns.update(EVALUATE_S2UT=False, EVALUATE_T2TT_EXTERNAL=True, REFERENCE_BENCHMARK="saved-reference")
    output = tmp_path / "external-text" / ns["T2TT_RUN_ID"] / ns["T2TT_TOKENIZER"] / "report/text-bleu.json"
    calls = []
    def execute_json(*args, **kw):
        validate_command(args[2], args[3:])
        calls.append(args[2])
        return {}
    def run(name, module, *args, **kw):
        validate_command(module, args)
        calls.append(module)
        if args[0] == "evaluate":
            write_json(output, {"text_bleu": 1.})
    ns.update(execute_json=execute_json, run=run)
    exec(source["external-text"], ns)
    for identity in ("restore-inputs", "registration", "restore-speech-artifacts", "restore-models", "evaluate"):
        exec(source[identity], ns)
    assert calls == ["alignment_transfer_s2ut.evaluate_t2tt_external"] * 3


def test_text_only_training_skips_speech_preparation(tmp_path):
    source, ns = namespace(tmp_path, "training")
    ns.update(PREPARE_AUDIO_PACKS=False, PREPARE_S2UT=False)
    exec(source["audio-staging"], ns)
    exec(source["speech-preparation"], ns)


@pytest.mark.parametrize("name", ["training", "evaluate"])
def test_primary_defaults_and_corpus_settings_are_independent_of_dataset_names(tmp_path, name, monkeypatch):
    source, ns = namespace(tmp_path, name)
    assert ns["EXPERIMENT_ID"] == "experiment-001"
    assert ns["T2TT_RUN_ID"] == "t2tt-001"
    assert not any(brand in "".join(source.values()).lower() for brand in ("kokoro", "jparacrawl"))
    for key in ("CORPUS_ROOT", "S2ST_INPUT_MAP"):
        monkeypatch.setenv(key, ns["os"].environ.get(key, ""))
    ns.update(CORPUS_ROOT_OVERRIDE=str(tmp_path / "other-corpus"),
              SOURCE_MANIFEST_OVERRIDE=str(tmp_path / "manifests/pairs.jsonl"),
              RELEASE_SUMMARY_OVERRIDE="", config={})
    ns.update(execute_json=lambda *a, **kw: {"expected_pairs": None}, execute=lambda *a, **kw: None)
    exec(source["input-settings"], ns)
    assert ns["CORPUS_ROOT"] == tmp_path / "other-corpus"
    assert ns["SOURCE_MANIFEST"] == tmp_path / "manifests/pairs.jsonl"
    assert ns["RELEASE_SUMMARY"] is None
    ns["COMMON"] = tmp_path / "common"
    write_json(ns["COMMON"] / "dataset-lock.json", {
        "total_pairs": 7, "splits": {"train": {"pairs": 5}, "dev": {"pairs": 2}}, "wav_bytes_estimate": 1024})
    exec(source["helpers"], ns)
    calls = []
    ns["corpus_stage"] = lambda *a, **kw: calls.append(a)
    ns["corpus_common"]()
    assert calls[0][2] == "corpus" and "--summary" not in calls[0]
