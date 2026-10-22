"""Read-only checkpoint snapshot discovery and fail-closed CLI contracts."""
from pathlib import Path
import zipfile
import pytest
from alignment_transfer_s2ut.benchmark import restore_snapshot
from alignment_transfer_s2ut.io import digest, file_hash, write_json, read_json
from alignment_transfer_s2ut.cli import parser_for, run


def snapshot(root, updates):
    root.mkdir(parents=True)
    names = ["checkpoints/checkpoint_best.pt", "checkpoints/checkpoint_last.pt",
             "research-metadata.json", "s2ut-recipe-lock.json"]
    hashes = {}
    import hashlib
    with zipfile.ZipFile(root / "snapshot.zip", "w") as pack:
        for name in names:
            value = (name + str(updates)).encode()
            hashes[name] = hashlib.sha256(value).hexdigest()
            pack.writestr(name, value)
    write_json(root / "snapshot.json", {"storage": "zip-v1", "updates": updates,
        "zip_sha256": file_hash(root / "snapshot.zip"), "files": hashes})


def test_snapshot_restoration_is_verified_and_frozen_on_resume(tmp_path):
    backup = tmp_path / "backup"
    snapshot(backup / "first", 10)
    archive_hash = file_hash(backup / "first/snapshot.zip")
    value = restore_snapshot(backup, tmp_path / "restored")
    assert value["updates"] == 10 and (tmp_path / "restored/checkpoints/checkpoint_best.pt").is_file()
    snapshot(backup / "newer", 20)
    assert restore_snapshot(backup, tmp_path / "restored", resume=True) == value
    assert file_hash(backup / "first/snapshot.zip") == archive_hash
    with pytest.raises(ValueError, match="separate"):
        restore_snapshot(backup, backup / "derived")


def test_corrupt_or_unsafe_snapshots_are_rejected(tmp_path):
    snapshot(tmp_path / "source", 10)
    path = tmp_path / "source/snapshot.json"
    value = read_json(path)
    value["files"]["../escape"] = "0" * 64
    write_json(path, value, overwrite=True)
    with pytest.raises(ValueError, match="unsafe"):
        restore_snapshot(tmp_path / "source", tmp_path / "unsafe")
    value["files"].pop("../escape")
    write_json(path, value, overwrite=True)
    with (tmp_path / "source/snapshot.zip").open("ab") as stream:
        stream.write(b"corrupt archive")
    with pytest.raises(ValueError, match="checksum"):
        restore_snapshot(tmp_path / "source", tmp_path / "corrupt")


def test_speech_training_requires_registered_baseline_by_default():
    args = parser_for("train_s2ut").parse_args(["--manifest", "manifest", "--units", "units", "--output", "run"])
    assert args.trainer == "fairseq"
    with pytest.raises(ValueError, match="baseline-record"):
        run("train_s2ut", args)
    args = parser_for("train_s2ut").parse_args(["--manifest", "manifest", "--units", "units", "--output", "run",
        "--baseline-record", "baseline.json", "--warmup-updates", "400"])
    with pytest.raises(ValueError, match="unsupported overrides"):
        run("train_s2ut", args)


def test_native_notebook_keeps_baseline_and_binds_backbone():
    import ast
    import json
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           "notebooks/colab/benchmark_comparison.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    ast.parse(source)
    assert "RUN_TRANSFER = False" in source
    assert "RUN_MONOTONIC = False" in source
    assert "s2ut-training-experiment-001-paper-fisher-v1" in source
    assert "baseline['backbone']" in source
    assert "test_native_fp16_trainer_and_fp32_prior" in source
    assert "--finetune-from-model" not in source
    assert "--max-updates" not in source[source.index("TRANSFER_ARGS ="):]
    assert "'compare', *records," in source
    assert "'--output', COMPARISON_DIR / 'training-comparison.json', '--resume'" in source


def notebook_cell(prefix):
    import json
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           "notebooks/colab/benchmark_comparison.ipynb").read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
                and "".join(c["source"]).startswith(prefix))


@pytest.mark.parametrize("t2tt,monotonic", [(False, False), (True, False), (False, True), (True, True)])
def test_optional_monotonic_training_wiring(tmp_path, t2tt, monotonic):
    import json
    from types import SimpleNamespace
    calls, plans = [], []
    def plan(command, **kwargs):
        plans.append(command)
        return json.dumps(dict(training_budget=300000, initialization="fresh", selection_policy={}))
    (tmp_path / "text-evaluation").mkdir()
    write_json(tmp_path / "text-evaluation/dev-text.json", {"text_bleu": 1.})
    namespace = dict(Path=Path, json=json, ROOT=str(tmp_path), RUN_TRANSFER=t2tt, RUN_MONOTONIC=monotonic,
        BASELINE=tmp_path / "baseline.json", MANIFEST=tmp_path / "manifest.jsonl", PYTHON="python",
        BACKBONE_CONFIG=tmp_path / "backbone.json", T2TT_UPDATES=20000, ANNEAL_STEPS=200000,
        subprocess=SimpleNamespace(check_output=plan), run=lambda *a, **kw: calls.append(a))
    exec(notebook_cell("# Preserve primary preparation"), namespace)
    exec(notebook_cell("TRANSFER ="), namespace)
    assert any(a[1] == "alignment_transfer_s2ut.train_t2tt" for a in calls) == (t2tt or not monotonic)
    training = {a[0]: list(a[3:]) for a in calls if a[1] == "alignment_transfer_s2ut.benchmark"}
    assert ("native-transfer" in training) == t2tt
    assert ("native-monotonic" in training) == monotonic
    args = namespace["MONOTONIC_ARGS"]
    assert args[args.index("--prior-type") + 1] == "monotonic" and "--teacher" not in args
    for field in ("--baseline", "--manifest", "--units", "--method", "--anneal-steps"):
        original = namespace["TRANSFER_ARGS"]
        assert args[args.index(field) + 1] == original[original.index(field) + 1]
    assert "--teacher" in namespace["TRANSFER_ARGS"]
    assert len(plans) == int(monotonic) + int(t2tt or not monotonic)


@pytest.mark.parametrize("monotonic,t2tt,ready_t2tt,ready_monotonic,expected", [
    (False, False, True, False, ["baseline", "annealed-t2tt"]),
    (True, True, True, True, ["baseline", "annealed-monotonic", "annealed-t2tt"]),
    (True, False, False, True, ["baseline", "annealed-monotonic"]),
    (False, True, True, True, ["baseline", "annealed-t2tt"]),
    (True, True, False, True, []),
])
@pytest.mark.parametrize("asr", [False, True])
def test_optional_control_uses_common_evaluation_and_table(tmp_path, monkeypatch,
        monotonic, t2tt, ready_t2tt, ready_monotonic, expected, asr):
    import json
    import sys
    from types import ModuleType
    calls, tables = [], []
    ipython, display = ModuleType("IPython"), ModuleType("IPython.display")
    display.Markdown, display.display = str, tables.append
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    def run(*args, **kwargs):
        calls.append(args)
        if "--output" in args:
            path = Path(args[args.index("--output") + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            value = {"fair_comparison": True}
            if args[1] == "alignment_transfer_s2ut.evaluate":
                value = {"unit_error_rate": .25, **({"asr_bleu": 12.5} if asr else {})}
            write_json(path, value, overwrite=True)
    for name, ready in (("annealed-t2tt", ready_t2tt), ("annealed-monotonic", ready_monotonic)):
        if ready:
            folder = tmp_path / name / "checkpoints"
            folder.mkdir(parents=True)
            (folder / "checkpoint_best.pt").touch()
    namespace = dict(Path=Path, json=json, ROOT=str(tmp_path), RUN_TRANSFER=t2tt, RUN_MONOTONIC=monotonic,
        BASELINE=tmp_path / "baseline.json", MANIFEST=tmp_path / "manifest.jsonl",
        TRANSFER=tmp_path / "annealed-t2tt", MONOTONIC=tmp_path / "annealed-monotonic",
        GENERATE_WAVEFORMS=asr, ASR_BLEU=asr, run=run)
    exec(notebook_cell("from IPython.display"), namespace)
    inference = [a for a in calls if a[1:3] == ("alignment_transfer_s2ut.benchmark", "infer")]
    assert [a[0].removesuffix("-infer") for a in inference] == expected
    evaluations = [a for a in calls if a[1] == "alignment_transfer_s2ut.evaluate"]
    assert len(evaluations) == len(expected)
    for args in evaluations:
        assert args[args.index("--manifest") + 1] == namespace["MANIFEST"]
        assert args[args.index("--units") + 1] == tmp_path / "units"
        assert ("--asr-config" in args) == asr
    if expected:
        assert namespace["COMPARISON_DIR"].name == "-".join(expected)
        assert "| S2UT Baseline |" in tables[0] and "0.2500" in tables[0]
        assert ("Annealed Monotonic Guidance" in tables[0]) == ("annealed-monotonic" in expected)
        assert ("Annealed T2TT Guidance" in tables[0]) == ("annealed-t2tt" in expected)
        assert ("12.50" in tables[0]) == asr
    else:
        assert not tables and not calls
