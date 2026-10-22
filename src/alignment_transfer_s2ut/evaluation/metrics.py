from pathlib import Path
import yaml
from ..data.manifest import load_manifest, manifest_hash
from ..data.audio import waveform
from ..io import read_json, read_jsonl, write_json, ArtifactStore, digest, file_hash
from ..preprocessing import pinned_model
from ..progress import progress


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def evaluate(args):
    rows = load_manifest(args.manifest, args.split)
    predictions = read_jsonl(args.predictions)
    by_id = {p["pair_id"]: p for p in predictions}
    if not rows or len(by_id) != len(predictions) or set(by_id) != {r["pair_id"] for r in rows}:
        raise ValueError("evaluation prediction IDs must match the selected split exactly")
    signatures = {(p["checkpoint_hash"], p["vocoder_hash"], digest(p["inference_config"])) for p in predictions}
    if len(signatures) != 1:
        raise ValueError("mixed checkpoints, vocoders or inference configurations")
    for row in rows:
        prediction = by_id[row["pair_id"]]
        if prediction["source_audio_hash"] != (row.get("ja_sha256") or file_hash(row["ja_audio"])):
            raise ValueError("evaluation source audio differs from the fixed manifest")
        if prediction.get("inference_beta") != 0:
            raise ValueError("evaluation requires teacher-free predictions")
    contract = {"predictions_hash": file_hash(args.predictions), "manifest_hash": manifest_hash(rows),
                "inference_config": predictions[0]["inference_config"], "vocoder_hash": predictions[0]["vocoder_hash"],
                "backend": predictions[0]["backend"],
                "asr_config_hash": file_hash(args.asr_config) if args.asr_config else None,
                "unit_artifacts_hash": digest([file_hash(ArtifactStore(args.units).path(row["pair_id"]))
                                               for row in rows]) if args.units else None}
    if args.resume and Path(args.output).is_file():
        saved = read_json(args.output)
        if saved.get("evaluation_contract") != contract:
            raise ValueError("evaluation resume contract changed")
        return saved
    results = {"split": args.split, "samples": len(rows), "checkpoint_hash": predictions[0]["checkpoint_hash"],
               "backend": predictions[0]["backend"], "vocoder_hash": predictions[0]["vocoder_hash"],
               "evaluation_contract": contract}
    if args.units:
        store, errors, count = ArtifactStore(args.units), 0, 0
        for row in rows:
            expected = {"split": args.split}
            if row.get("en_sha256"):
                expected["en_audio_hash"] = row["en_sha256"]
            _, artifact = store.get(row["pair_id"], expected)
            reference = artifact["reduced_units"].tolist()
            errors += edit_distance(reference, by_id[row["pair_id"]]["units"])
            count += len(reference)
        results["unit_error_rate"] = errors / count
    if args.asr_config:
        if any(p["backend"] == "smoke" for p in predictions):
            raise ValueError("synthetic smoke output cannot be used for ASR-BLEU research results")
        cfg = yaml.safe_load(Path(args.asr_config).read_text(encoding="utf-8"))
        pinned_model(cfg)
        from transformers import pipeline
        from sacrebleu.metrics import BLEU
        asr = pipeline("automatic-speech-recognition", model=cfg["model"], revision=cfg["revision"],
                       device=args.device)
        hypotheses = []
        for index, row in enumerate(rows):
            if file_hash(by_id[row["pair_id"]]["en_audio"]) != by_id[row["pair_id"]]["generated_audio_hash"]:
                raise ValueError("generated waveform changed before ASR evaluation")
            wave, rate = waveform(by_id[row["pair_id"]]["en_audio"])
            hypotheses.append(asr({"array": wave.numpy(), "sampling_rate": rate},
                                  generate_kwargs=cfg.get("generate_kwargs", {}))["text"])
            progress("evaluation ASR", index + 1, len(rows))
        bleu = BLEU(tokenize="13a")
        score = bleu.corpus_score(hypotheses, [[r["en_text"] for r in rows]])
        results.update(asr_bleu=score.score, bleu_signature=str(bleu.get_signature()),
                       asr_config_hash=digest(cfg), asr_hypotheses=dict(zip([r["pair_id"] for r in rows], hypotheses)))
    if not args.units and not args.asr_config:
        raise ValueError("supply --units and/or --asr-config")
    write_json(args.output, results, args.overwrite)
    return results
