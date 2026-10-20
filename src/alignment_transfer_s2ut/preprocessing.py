from pathlib import Path
import re
import numpy as np
import torch
import yaml
from .io import ArtifactStore, digest, file_hash
from .data.audio import waveform
from .data.manifest import load_manifest, manifest_hash
from .alignment.ctc import viterbi_spans
from .units.timing import reduce_units, frame_spans, run_spans
from .progress import progress


def pinned_model(config):
    if not config.get("model") or not re.fullmatch(r"[0-9a-f]{40}", str(config.get("revision", ""))):
        raise ValueError("model ID and immutable 40-character revision are required")


def provenance(row, stage, config, manifest):
    return {"stage": stage, "split": row["split"], "manifest_hash": manifest,
            "config_hash": digest(config), "ja_audio_hash": file_hash(row["ja_audio"]),
            "en_audio_hash": file_hash(row["en_audio"]),
            "text_hash": digest([row["ja_text"], row["en_text"]])}


def existing(store, row, expected, resume):
    if resume and store.path(row["pair_id"]).exists():
        store.get(row["pair_id"], expected)
        return True
    return False


def extract_units(manifest, output, config_path, device="cpu", resume=False, overwrite=False):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pinned_model(config["hubert"])
    if config.get("layer") != 6 or config.get("clusters") != 100 or not config.get("reduce_consecutive", True):
        raise ValueError("v1 requires HuBERT layer 6 / KM100")
    kmeans_path = config["kmeans"]["path"]
    if file_hash(kmeans_path) != config["kmeans"]["sha256"]:
        raise ValueError("k-means checksum mismatch")
    import joblib
    from transformers import HubertModel, AutoFeatureExtractor
    progress("units: loading pinned HuBERT and k-means")
    centers = np.asarray(joblib.load(kmeans_path).cluster_centers_, dtype=np.float32)
    if centers.shape != (100, 768):
        raise ValueError("expected 100 HuBERT Base centers of dimension 768")
    model = HubertModel.from_pretrained(config["hubert"]["model"],
                                       revision=config["hubert"]["revision"]).to(device).eval()
    processor = AutoFeatureExtractor.from_pretrained(config["hubert"]["model"],
                                                     revision=config["hubert"]["revision"])
    rows, store = load_manifest(manifest), ArtifactStore(output)
    identity = manifest_hash(rows)
    for index, row in enumerate(rows):
        meta = provenance(row, "units", config, identity)
        meta["unit_config_hash"] = digest(config)
        if not existing(store, row, meta, resume):
            wave, rate = waveform(row["en_audio"])
            inputs = processor(wave.numpy(), sampling_rate=rate, return_tensors="pt")
            with torch.no_grad():
                hidden = model(**{k: v.to(device) for k, v in inputs.items()},
                               output_hidden_states=True).hidden_states[6][0]
                center_tensor = torch.as_tensor(centers, device=device)
                # Work in chunks over already extracted features to bound distance-matrix memory.
                units = torch.cat([((h.square().sum(-1, keepdim=True) +
                                     center_tensor.square().sum(-1)[None]) - 2 * h @ center_tensor.T).argmin(-1)
                                   for h in hidden.split(4096)]).cpu().numpy()
            reduced, runs = reduce_units(units)
            frames = frame_spans(len(units), rate, len(wave) / rate,
                                 model.config.conv_kernel, model.config.conv_stride)
            store.put(row["pair_id"], meta, {"original_units": units, "reduced_units": reduced,
                      "runs": runs, "frame_spans": frames, "unit_spans": run_spans(runs, frames),
                      "duration": np.asarray(len(wave) / rate)}, overwrite)
        progress("units", index + 1, len(rows))


def align_speech_text(manifest, output, config_path, device="cpu", resume=False, overwrite=False):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rows, store = load_manifest(manifest), ArtifactStore(output)
    identity = manifest_hash(rows)
    from transformers import AutoProcessor, AutoModelForCTC
    for language in ("ja", "en"):
        cfg = config[language]
        pinned_model(cfg)
        progress("CTC: loading " + language)
        processor = AutoProcessor.from_pretrained(cfg["model"], revision=cfg["revision"])
        model = AutoModelForCTC.from_pretrained(cfg["model"], revision=cfg["revision"]).to(device).eval()
        vocabulary = processor.tokenizer.get_vocab()
        blank = model.config.pad_token_id
        if blank is None:
            raise ValueError("CTC model must declare its blank ID")
        for index, row in enumerate(rows):
            key = row["pair_id"] + ":" + language
            meta = provenance(row, "ctc_alignment", config, identity)
            meta.update(language=language, original_pair_id=row["pair_id"])
            if resume and store.path(key).exists():
                store.get(key, meta)
                continue
            text, labels, indices = row[language + "_text"], [], []
            masked = []
            transform = cfg.get("text_transform", "identity")
            if transform not in {"identity", "upper", "lower"}:
                raise ValueError("CTC text_transform must be explicit")
            if cfg.get("unknown_policy", "error") not in {"mask", "error"}:
                raise ValueError("unknown_policy must be mask or error")
            for position, char in enumerate(text):
                if char in cfg.get("ignore_characters", ""):
                    masked.append(position)
                    continue
                token = char if transform == "identity" else getattr(char, transform)()
                if token == " ":
                    token = processor.tokenizer.word_delimiter_token
                if token not in vocabulary or vocabulary[token] in {blank, processor.tokenizer.unk_token_id}:
                    if cfg.get("unknown_policy") == "mask":
                        masked.append(position)
                        continue
                    raise ValueError(f"CTC vocabulary cannot align codepoint {char!r}; "
                                     "configure a character model or explicitly ignore that character")
                labels.append(vocabulary[token])
                indices.append(position)
            if not labels:
                raise ValueError("CTC transcript has no alignable codepoints: " + key)
            wave, rate = waveform(row[language + "_audio"])
            inputs = processor(wave.numpy(), sampling_rate=rate, return_tensors="pt")
            with torch.no_grad():
                log_probs = model(**{k: v.to(device) for k, v in inputs.items()}).logits[0].log_softmax(-1).cpu().numpy()
            aligned = viterbi_spans(log_probs, labels, blank, len(wave) / rate)
            spans = np.full((len(text), 2), -1.0)
            spans[indices] = aligned
            meta["masked_codepoint_indices"] = masked
            store.put(key, meta, {"spans": spans, "duration": np.asarray(len(wave) / rate)}, overwrite)
            progress("CTC " + language, index + 1, len(rows))


def build_teacher(manifest, mt_root, alignment_root, unit_root, output, s2ut_config,
                  resume=False, overwrite=False):
    from .config import BackboneConfig
    rows = load_manifest(manifest)
    identity = manifest_hash(rows)
    mt_store, ctc_store, units_store, output_store = map(ArtifactStore,
                                                       (mt_root, alignment_root, unit_root, output))
    architecture_hash = digest(BackboneConfig.load(s2ut_config).asdict())
    teacher_identity = None
    speech_identity = None
    for index, row in enumerate(rows):
        pair = row["pair_id"]
        expected = {"split": row["split"], "manifest_hash": identity,
                    "text_hash": digest([row["ja_text"], row["en_text"]])}
        mt_meta, mt = mt_store.get(pair, expected)
        source_meta, source = ctc_store.get(pair + ":ja", {**expected, "original_pair_id": pair})
        target_meta, target = ctc_store.get(pair + ":en", {**expected, "original_pair_id": pair})
        units_meta, units = units_store.get(pair, expected)
        mt_identity = tuple(mt_meta[key] for key in ("mt_checkpoint_hash", "mt_architecture_hash", "tokenizer_hash", "backend"))
        acoustic_identity = (source_meta["config_hash"], target_meta["config_hash"], units_meta["unit_config_hash"])
        if teacher_identity is not None and teacher_identity != mt_identity:
            raise ValueError("mixed MT checkpoint, architecture, tokenizer or backend")
        if speech_identity is not None and speech_identity != acoustic_identity:
            raise ValueError("mixed CTC or unit configuration")
        teacher_identity, speech_identity = mt_identity, acoustic_identity
        for meta in (source_meta, target_meta, units_meta):
            if meta["ja_audio_hash"] != file_hash(row["ja_audio"]) or meta["en_audio_hash"] != file_hash(row["en_audio"]):
                raise ValueError("stale speech artifact")
        if mt["attention"].shape != (len(row["en_text"]), len(row["ja_text"])):
            raise ValueError("attention dimensions differ from original codepoints")
        if mt_meta["mt_architecture_hash"] != architecture_hash:
            raise ValueError("T2TT / S2UT architecture mismatch")
        if source["spans"].shape != (len(row["ja_text"]), 2) or target["spans"].shape != (len(row["en_text"]), 2):
            raise ValueError("CTC spans do not preserve original codepoint indices")
        meta = {**expected, "stage": "teacher_alignment", "mt_checkpoint_hash": mt_meta["mt_checkpoint_hash"],
                "ja_audio_hash": source_meta["ja_audio_hash"], "en_audio_hash": target_meta["en_audio_hash"],
                "mt_architecture_hash": mt_meta["mt_architecture_hash"],
                "s2ut_architecture_hash": architecture_hash, "tokenizer_hash": mt_meta["tokenizer_hash"],
                "ctc_alignment_hash": digest([file_hash(ctc_store.path(pair + ":ja")),
                                               file_hash(ctc_store.path(pair + ":en"))]),
                "ctc_config_hash": digest(acoustic_identity[:2]),
                "unit_config_hash": units_meta["unit_config_hash"],
                "units_hash": file_hash(units_store.path(pair)), "backend": mt_meta["backend"]}
        if not existing(output_store, row, meta, resume):
            output_store.put(pair, meta, {"mt_attention": mt["attention"], "source_spans": source["spans"],
                             "target_spans": target["spans"], "source_duration": source["duration"],
                             "unit_spans": units["unit_spans"], "reduced_units": units["reduced_units"]}, overwrite)
        progress("teacher packet", index + 1, len(rows))
