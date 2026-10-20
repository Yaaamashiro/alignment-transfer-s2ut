"""Deterministic synthetic pipeline, explicitly separate from external-model validation."""
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf
import torch
import yaml
from .cli import parser_for, run
from .io import ArtifactStore, write_jsonl, write_json
from .data.manifest import load_manifest, manifest_hash
from .preprocessing import provenance
from .alignment.ctc import viterbi_spans
from .units.timing import reduce_units, frame_spans, run_spans
from .progress import progress

TINY_CONFIG = {"backbone": {"encoder_layers": 1, "decoder_layers": 1, "encoder_embed_dim": 16,
    "decoder_embed_dim": 16, "encoder_ffn_embed_dim": 32, "decoder_ffn_embed_dim": 32,
    "encoder_attention_heads": 2, "decoder_attention_heads": 2, "dropout": 0.0,
    "attention_dropout": 0.0, "activation_dropout": 0.0, "conv_channels": 32,
    "input_feat_per_channel": 8}}


def invoke(stage, *arguments):
    return run(stage, parser_for(stage).parse_args(list(map(str, arguments))))


def synthetic_frontends(manifest, ctc_root, units_root, overwrite=False):
    rows = load_manifest(manifest)
    identity = manifest_hash(rows)
    ctc, units = ArtifactStore(ctc_root), ArtifactStore(units_root)
    config = {"backend": "synthetic-smoke-fixture", "layer": 6, "clusters": 100}
    for row in rows:
        duration = sf.info(row["en_audio"]).duration
        for language in ("ja", "en"):
            text = row[language + "_text"]
            labels = [i + 1 for i in range(len(text))]
            emissions = np.full((3 * len(text) + 1, len(text) + 1), -20.0)
            emissions[:, 0] = 0
            for index, label in enumerate(labels):
                emissions[3 * index + 1:3 * index + 3, label] = 10
            emissions -= np.log(np.exp(emissions).sum(-1, keepdims=True))
            spans = viterbi_spans(emissions, labels, 0, duration)
            meta = provenance(row, "ctc_alignment", config, identity)
            meta.update(language=language, original_pair_id=row["pair_id"])
            ctc.put(row["pair_id"] + ":" + language, meta,
                    {"spans": spans, "duration": np.asarray(duration)}, overwrite)
        original = np.asarray([7, 7, 12, 12, 5, 5, 8, 8, 3, 3], dtype=np.int64)
        reduced, runs = reduce_units(original)
        frames = frame_spans(len(original), 16000, duration, [400], [320])
        meta = provenance(row, "units", config, identity)
        from .io import digest
        meta["unit_config_hash"] = digest(config)
        units.put(row["pair_id"], meta, {"original_units": original, "reduced_units": reduced,
                    "runs": runs, "frame_spans": frames, "unit_spans": run_spans(runs, frames),
                    "duration": np.asarray(duration)}, overwrite)


def smoke(output, overwrite=False):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("use a new empty smoke output directory")
    torch.set_num_threads(1)
    rows = []
    for index, (split, ja, en) in enumerate([("train", "あい", "ab"), ("train", "いあ", "ba"),
                                           ("dev", "あ あ", "a b"), ("test", "い い", "b a")]):
        row = {"pair_id": split + str(index), "split": split, "ja_text": ja, "en_text": en}
        for language in ("ja", "en"):
            path = root / (row["pair_id"] + "-" + language + ".wav")
            audio = 0.1 * np.sin(2 * np.pi * (220 + index * 20) * np.arange(4000) / 16000)
            sf.write(path, audio, 16000)
            row[language + "_audio"] = str(path)
        rows.append(row)
    write_jsonl(root / "input.jsonl", rows)
    config = root / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY_CONFIG), encoding="utf-8")
    manifest = root / "manifest.jsonl"
    invoke("prepare_manifest", "--input", root / "input.jsonl", "--output", manifest)
    common = ["--manifest", manifest, "--config", config, "--backend", "smoke", "--batch-size", 2,
              "--update-frequency", 1, "--max-updates", 2, "--warmup-updates", 0]
    invoke("train_t2tt", *common, "--output", root / "t2tt")
    invoke("extract_t2tt_attention", "--manifest", manifest, "--checkpoint", root / "t2tt/checkpoint_best.pt",
           "--output", root / "mt")
    synthetic_frontends(manifest, root / "ctc", root / "units")
    invoke("build_teacher_alignment", "--manifest", manifest, "--mt-attention", root / "mt",
           "--alignment", root / "ctc", "--units", root / "units", "--config", config,
           "--output", root / "teacher")
    methods = ["baseline", "distillation", "guidance", "annealed_guidance"]
    for method in methods:
        invoke("train_s2ut", *common, "--units", root / "units", "--teacher", root / "teacher",
               "--method", method, "--anneal-steps", 1, "--output", root / method)
    # Delete all teacher dependencies before inference, then generate units and test WAVs for every method.
    for name in ("t2tt", "mt", "ctc", "teacher"):
        shutil.rmtree(root / name)
    test_row = rows[-1]
    write_jsonl(root / "speech_only.jsonl", [{"pair_id": test_row["pair_id"], "ja_audio": test_row["ja_audio"]}])
    for method in methods:
        prediction_path = root / method / "predictions.jsonl"
        invoke("infer", "--input", root / "speech_only.jsonl", "--checkpoint", root / method / "checkpoint_best.pt",
               "--output", prediction_path, "--max-units", 8, "--beam-size", 1, "--smoke-vocoder")
        invoke("evaluate", "--manifest", manifest, "--predictions", prediction_path, "--units", root / "units",
               "--split", "test", "--output", root / method / "evaluation.json")
    result = {"status": "passed", "methods": methods, "teacher_dependencies_deleted": True,
              "backend": "synthetic-smoke-only", "external_models_validated": False,
              "output": str(root)}
    write_json(root / "smoke_result.json", result)
    progress("synthetic smoke complete")
    return result
