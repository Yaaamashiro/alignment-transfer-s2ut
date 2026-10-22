"""One native fairseq source-only generator for external and transfer checkpoints."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from fairseq import tasks
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from ..benchmark import read_record, load_native, native_config
from ..io import read_json, read_jsonl, write_json, write_jsonl, file_hash, digest, atomic_write
from ..progress import progress
from .decode import load_vocoder


def infer_native(args):
    if args.units_only == bool(args.vocoder_config):
        raise ValueError("select --units-only or --vocoder-config")
    if args.beam_size < 1 or args.max_len_a < 0 or args.max_len_b < 1 or args.length_penalty < 0:
        raise ValueError("invalid native generation settings")
    record = read_record(args.record, verify_data=False)
    # Generation needs model configuration/dictionaries, never train/dev/test labels.
    for name, sha in record["prepared"]["files"].items():
        if not name.endswith(".tsv") and file_hash(Path(record["data"]) / name) != sha:
            raise ValueError("registered inference configuration/dictionary changed")
    rows = read_jsonl(args.input)
    if not rows or len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("empty/duplicate source-only inference IDs")
    for row in rows:
        if set(row) - {"pair_id", "ja_audio", "split"}:
            raise ValueError("native inference accepts only pair_id, ja_audio and optional split")
        row["ja_audio"] = str((Path(args.input).resolve().parent / row["ja_audio"]).resolve())
    cfg = deepcopy(record["config"])
    # Identical stock architecture/state keys at inference; no teacher or transfer
    # task construction, even if the training checkpoint came from our extension.
    cfg["task"]["_name"] = "speech_to_speech"
    cfg["task"].pop("alignment_config", None)
    cfg["task"]["eval_inference"] = False
    cfg["task"]["data"] = record["data"]
    cfg["model"]["_name"] = "s2ut_transformer_fisher"
    if "arch" in cfg["model"]:
        cfg["model"]["arch"] = "s2ut_transformer_fisher"
    cfg = native_config(cfg)
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    state = load_native(record["checkpoint"])
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.float().to(args.device).eval()
    generation = Namespace(beam=args.beam_size, max_len_a=args.max_len_a, max_len_b=args.max_len_b,
                           lenpen=args.length_penalty, min_len=1)
    generator = task.build_generator([model], generation)
    source = SpeechToTextDataset("inference", False, task.data_cfg,
        [r["ja_audio"] for r in rows], [1] * len(rows), ids=[r["pair_id"] for r in rows])
    vocoder_hash, vocode, rate = None, None, None
    if args.vocoder_config:
        vocode, rate, vocoder_hash = load_vocoder(args.vocoder_config, args.device)
    inference_config = {"engine": "native_fairseq", "revision": record["fairseq_revision"],
                        "beam_size": args.beam_size, "max_len_a": args.max_len_a, "max_len_b": args.max_len_b,
                        "length_penalty": args.length_penalty, "units_only": args.units_only, "precision": "fp32",
                        "source_config_hash": record["prepared"]["files"]["config.yaml"]}
    output, predictions = Path(args.output), []
    if output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(output)
    previous = {r["pair_id"]: r for r in read_jsonl(output)} if args.resume and output.is_file() else {}
    for index, row in enumerate(rows):
        expected = {"checkpoint_hash": record["checkpoint_hash"], "backend": "fairseq", "vocoder_hash": vocoder_hash,
                    "source_audio_hash": file_hash(row["ja_audio"]), "inference_config": inference_config}
        cache = output.parent / (output.stem + "-cache") / (digest(row["pair_id"]) + ".json")
        cached = previous.get(row["pair_id"])
        if cached is None and args.resume and cache.is_file():
            cached = read_json(cache)
        if cached is not None:
            if any(cached.get(k) != v for k, v in expected.items()):
                raise ValueError("native inference resume contract changed")
            if cached.get("en_audio") and file_hash(cached["en_audio"]) != cached["generated_audio_hash"]:
                raise ValueError("generated waveform hash mismatch")
            prediction = cached
        else:
            feature = source._get_source_audio(index).unsqueeze(0).to(args.device)
            sample = {"id": torch.tensor([index], device=args.device), "net_input": {
                "src_tokens": feature, "src_lengths": torch.tensor([feature.size(1)], device=args.device)}}
            with torch.no_grad():
                tokens = task.inference_step(generator, [model], sample)[0][0]["tokens"].tolist()
            symbols = [task.target_dictionary[token] for token in tokens if token != task.target_dictionary.eos()]
            if not symbols or any(not token.isdecimal() or not 0 <= int(token) < 100 for token in symbols):
                raise ValueError("native generator returned empty/non-KM100 output")
            units = list(map(int, symbols))
            prediction = {**expected, "pair_id": row["pair_id"], "units": units, "inference_beta": 0,
                          "eos_reached": tokens[-1] == task.target_dictionary.eos()}
            if vocode:
                wave = np.asarray(vocode(units), dtype=np.float32).reshape(-1)
                if not wave.size or not np.isfinite(wave).all():
                    raise ValueError("invalid vocoder output")
                path = output.parent / (output.stem + "-audio") / (digest(row["pair_id"]) + ".wav")
                atomic_write(path, lambda f: sf.write(f, wave, rate, format="WAV"), args.overwrite or args.resume)
                prediction.update(en_audio=str(path.resolve()), generated_audio_hash=file_hash(path))
            write_json(cache, prediction, args.overwrite)
        predictions.append(prediction)
        progress("native teacher-free inference", index + 1, len(rows))
    write_jsonl(output, predictions, args.resume or args.overwrite)
    return predictions
