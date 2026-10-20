from pathlib import Path
import math
import numpy as np
import soundfile as sf
import torch
import yaml
from ..data.audio import features
from ..io import atomic_write, read_json, read_jsonl, write_json, write_jsonl, file_hash, digest
from ..training import load_checkpoint, checkpoint_model
from ..tokenization.character import EOS, PAD, BOS, UNK
from ..progress import progress
from ..recipes import transform_features


@torch.no_grad()
def generate(model, source, lengths, beam_size=5, max_units=1000, length_penalty=1.0):
    if beam_size < 1 or max_units < 1:
        raise ValueError("beam size and maximum units must be positive")
    # Decode from source speech alone. Full-history decoding avoids mutable beam caches.
    model.eval()
    encoded = model.encode(source, lengths)
    beams = [([EOS], 0.0, False)]
    for _ in range(max_units + 1):
        candidates = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue
            previous = torch.tensor([tokens], device=source.device)
            logits, _, _ = model.decode(previous, encoded, prior=None, beta=0.0)
            probabilities = logits[0, -1].float().log_softmax(-1)
            probabilities[[PAD, BOS, UNK]] = -torch.inf
            if len(tokens) == 1:
                probabilities[EOS] = -torch.inf  # a vocoder needs at least one unit
            values, indices = probabilities.topk(min(beam_size, probabilities.numel() - 3))
            candidates.extend((tokens + [int(token)], score + float(value), int(token) == EOS)
                              for value, token in zip(values, indices) if torch.isfinite(value))
        beams = sorted(candidates, key=lambda item: item[1] / max(1, len(item[0]) - 1) ** length_penalty,
                       reverse=True)[:beam_size]
        if all(item[2] for item in beams):
            break
    chosen = max(beams, key=lambda item: item[1] / max(1, len(item[0]) - 1) ** length_penalty)
    tokens = [token - 4 for token in chosen[0][1:] if token != EOS]
    return tokens[:max_units], chosen[2]


def load_vocoder(path, device):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for name in ("checkpoint", "config"):
        if file_hash(config[name]) != config[name + "_sha256"]:
            raise ValueError("vocoder checksum mismatch: " + name)
    model_config = read_json(config["config"])
    if model_config.get("num_embeddings") != 100 or not model_config.get("dur_predictor_params"):
        raise ValueError("KM100 reduced units require a duration-predicting KM100 vocoder")
    if model_config.get("f0") or model_config.get("multispkr"):
        raise ValueError("v1 requires a single-speaker vocoder without external F0")
    from ..s2ut.fairseq_extension import verify_fairseq
    verify_fairseq()
    from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
    generator = CodeHiFiGANVocoder(config["checkpoint"], model_config).to(device).eval()
    sample_rate = int(model_config["sampling_rate"])
    def vocode(units):
        with torch.no_grad():
            return generator({"code": torch.tensor([units], device=device, dtype=torch.long)},
                             dur_prediction=True).cpu().numpy()
    return vocode, sample_rate, digest(config)


def infer(args):
    if args.units_only and (args.vocoder_config or args.smoke_vocoder):
        raise ValueError("--units-only cannot be combined with a vocoder")
    saved = load_checkpoint(args.checkpoint)
    if saved["modality"] != "speech":
        raise ValueError("S2UT checkpoint required")
    if args.smoke_vocoder and saved["backend"] != "smoke":
        raise ValueError("synthetic test vocoder cannot be used with research checkpoints")
    if not args.units_only and not args.vocoder_config and not args.smoke_vocoder:
        raise ValueError("waveform inference requires --vocoder-config (or explicit --units-only)")
    if args.audio:
        rows = [{"pair_id": Path(args.audio).stem, "ja_audio": str(Path(args.audio).resolve())}]
    else:
        rows = read_jsonl(args.input)
        for row in rows:
            if set(row) - {"pair_id", "ja_audio", "split"}:
                raise ValueError("inference input accepts only pair_id, ja_audio and optional split")
            row["ja_audio"] = str((Path(args.input).resolve().parent / row["ja_audio"]).resolve())
    if not rows or len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate inference inputs")
    model = checkpoint_model(saved, args.device).eval()
    vocoder_hash, vocode, rate = None, None, 16000
    if args.vocoder_config:
        vocode, rate, vocoder_hash = load_vocoder(args.vocoder_config, args.device)
    elif args.smoke_vocoder:
        vocoder_hash = "synthetic-test-tone-not-a-speech-vocoder"
        def vocode(units):
            return np.concatenate([0.05 * np.sin(2 * math.pi * (100 + int(u) * 5) *
                                   np.arange(320) / rate) for u in units]).astype(np.float32)
    output = Path(args.output)
    if output.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(output)
    previous_results = {r["pair_id"]: r for r in read_jsonl(output)} if args.resume and output.is_file() else {}
    checkpoint_hash = file_hash(args.checkpoint)
    predictions = []
    for index, row in enumerate(rows):
        inference_config = {"beam_size": args.beam_size, "max_units": args.max_units,
                            "length_penalty": args.length_penalty, "units_only": args.units_only}
        expected = {"checkpoint_hash": checkpoint_hash, "source_audio_hash": file_hash(row["ja_audio"]),
                    "backend": saved["backend"], "vocoder_hash": vocoder_hash, "inference_config": inference_config}
        cache_path = output.parent / (output.stem + "-cache") / (digest(row["pair_id"]) + ".json")
        cached = previous_results.get(row["pair_id"])
        if args.resume and cached is None and cache_path.is_file():
            cached = read_json(cache_path)
        if cached is not None:
            if any(cached.get(key) != value for key, value in expected.items()):
                raise ValueError("inference resume contract changed")
            if cached.get("en_audio") and file_hash(cached["en_audio"]) != cached["generated_audio_hash"]:
                raise ValueError("generated waveform hash mismatch")
            predictions.append(cached)
            progress("inference resume", index + 1, len(rows))
            continue
        feature_recipe = saved["recipe"].get("speech_recipe", {})
        source = features(row["ja_audio"], saved["backend"], saved["backbone"]["input_feat_per_channel"],
                          feature_recipe.get("waveform_scale", 1.0))
        source = transform_features(source, feature_recipe)
        units, completed = generate(model, source[None].to(args.device),
                                    torch.tensor([len(source)], device=args.device),
                                    args.beam_size, args.max_units, args.length_penalty)
        result = {**expected, "pair_id": row["pair_id"], "units": units, "eos_reached": completed, "inference_beta": 0}
        if vocode is not None:
            wave = np.asarray(vocode(units), dtype=np.float32).reshape(-1)
            if not wave.size or not np.isfinite(wave).all():
                raise ValueError("invalid vocoder output")
            audio_path = output.parent / "audio" / (digest(row["pair_id"]) + ".wav")
            atomic_write(audio_path, lambda stream: sf.write(stream, wave, rate, format="WAV"), args.overwrite or args.resume)
            result["en_audio"] = str(audio_path.resolve())
            result["generated_audio_hash"] = file_hash(audio_path)
        write_json(cache_path, result, args.overwrite)
        predictions.append(result)
        progress("inference", index + 1, len(rows))
    write_jsonl(output, predictions, args.overwrite or args.resume)
    return predictions
