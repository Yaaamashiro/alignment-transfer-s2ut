"""Bridge external speech corpora and benchmark artifacts to native S2UT."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from .benchmark import tsv, benchmark_digest
from .data.manifest import corpus_audio_path, kokoro_audio_path, prepare_manifest, load_manifest, manifest_hash, speech_audio_path, validate
from .io import ArtifactStore, digest, file_hash, read_json, read_jsonl, write_jsonl, local_path
from .units.timing import reduce_units, frame_spans, run_spans
from .progress import progress


def derived_output(output, *inputs):
    output = Path(output).resolve()
    roots = [Path(p).resolve() for p in inputs]
    if os.environ.get("CORPUS_ROOT"):
        roots.append(Path(os.environ["CORPUS_ROOT"]).resolve())
    if any(output.is_relative_to(root) for root in roots):
        raise ValueError("derived output must be outside benchmark/corpus inputs")
    return output


def import_kokoro(source, summary, root, output, resume=False):
    return import_corpus(source, root, output, summary=summary, resume=resume, legacy=True)


def import_corpus(source, root, output, summary=None, resume=False, *, legacy=False):
    """Convert the QC release to compact, portable benchmark common manifests."""
    root = Path(root).resolve()
    inputs = [root, Path(source).parent if legacy else Path(source)]
    if summary:
        inputs.append(Path(summary).parent if legacy else Path(summary))
    output = derived_output(output, *inputs)
    release = read_json(summary) if summary else None
    if release is not None and release.get("complete") is not True:
        raise ValueError("corpus release is incomplete")
    counts, rows, ids = Counter(), [], set()
    checksum = hashlib.sha256()
    splits = ("train", "dev", "test")
    with Path(source).open("rb") as stream:
        for line in stream:
            checksum.update(line)
            if not line.strip():
                continue
            raw = json.loads(line)
            pair, split = raw["pair_id"], raw["split"]
            if (not isinstance(pair, str) or not pair.strip() or any(c in pair for c in "\t\r\n/\\")
                    or pair in ids or split not in splits or
                    raw.get("qc_status", None if legacy else "accepted") != "accepted"):
                raise ValueError(f"invalid or duplicate corpus record: {pair}")
            ids.add(pair)
            counts[split] += 1
            if len(ids) % 2048 == 0:
                progress("corpus: read manifest", len(ids), release["accepted_pairs"] if release else None)
            if split == "test":
                continue
            row = {key: raw[key] for key in ("pair_id", "split")}
            row["corpus"] = raw.get("corpus", "external")
            row["source_id"] = raw.get("source_id", pair)
            for language in ("ja", "en"):
                key = language + "_audio"
                raw_path = raw.get(language + "_wav_16k") if legacy else (
                    raw.get(key + "_corpus_relative") or raw.get(key) or raw.get(language + "_wav_16k"))
                resolve = kokoro_audio_path if legacy else corpus_audio_path
                row[key], row[key + "_corpus_relative"] = resolve(raw_path, root, language)
                checksum_key = language + "_sha256"
                duration_key = language + "_duration"
                duration = raw[duration_key]
                if (not re.fullmatch(r"[0-9a-f]{64}", raw[checksum_key]) or isinstance(duration, bool)
                        or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0):
                    raise ValueError(f"invalid {language} WAV metadata: {pair}")
                row[checksum_key] = raw[checksum_key]
                row[duration_key] = duration
                for suffix in ("_text", "_text_raw", "_tts_text"):
                    row[language + suffix] = raw.get(language + suffix, raw[language + "_text"])
            rows.append(row)
    if not counts["train"] or not counts["dev"]:
        raise ValueError("corpus manifest requires nonempty train and dev splits")
    if release is not None and (len(ids) != release["accepted_pairs"] or
            any(counts[split] != release[split]["accepted"] for split in splits)):
        raise ValueError("corpus manifest counts differ from the completed release")
    progress("corpus: read manifest", len(ids), release["accepted_pairs"] if release else len(ids))
    validate(rows, check_audio=False)
    lock = dict(schema_version=1, source_manifest=str(Path(source).resolve()),
                source_manifest_sha256=checksum.hexdigest(),
                audio_path_policy="kokoro-audio-16k-v1" if legacy else "corpus-relative-v1",
                total_pairs=len(rows), release_pairs=len(ids),
                splits={split: {"pairs": counts[split] if split != "test" else 0} for split in splits},
                wav_bytes_estimate=sum(round(row[lang + "_duration"] * 16000) * 2 + 44
                                       for row in rows for lang in ("ja", "en")))
    marker = output / "dataset-lock.json"
    if marker.is_file() and (not resume or read_json(marker) != lock):
        raise ValueError("corpus common data identity changed")
    from direct_s2st.io import atomic_write_jsonl, atomic_write_json
    for split in splits:
        atomic_write_jsonl(output / (split + ".jsonl"), (r for r in rows if r["split"] == split), resume=resume)
    atomic_write_json(marker, lock, resume=resume)
    return lock


def export_manifest(common, prepared, output, include_test=False, resume=False):
    common, prepared, output = Path(common).resolve(), Path(prepared).resolve(), Path(output).resolve()
    derived_output(output, common, prepared)
    result = []
    data_lock = read_json(prepared / "data-lock.json")
    if data_lock.get("common_dataset_lock_sha256") != file_hash(common / "dataset-lock.json"):
        raise ValueError("common manifest lock differs from the prepared benchmark corpus")
    for split in ("train", "dev", "test") if include_test else ("train", "dev"):
        rows = read_jsonl(common / (split + ".jsonl"))
        expected = {row["id"]: row for row in tsv(prepared / (split + ".tsv"))}
        labels = {language: {r["id"]: r["tgt_text"].split() for r in
                  tsv(prepared / (task + "/" + split + ".tsv"))}
                  for language, task in (("ja", "source_letter"), ("en", "target_letter"))}
        if len(rows) != len(expected) or {r["pair_id"] for r in rows} != set(expected):
            raise ValueError("common/prepared sample mismatch")
        for row in rows:
            if row["split"] != split or speech_audio_path(row["ja_audio"]) != speech_audio_path(expected[row["pair_id"]]["src_audio"]):
                raise ValueError("common/prepared split or source mismatch")
            for language in ("ja", "en"):
                # Same explicit ASCII-whitespace policy as the frozen benchmark
                # labels. No Unicode normalization or new dictionary fitting.
                normalized = re.sub(r"[ \t\r\n]+", " ", row[language + "_text"].strip(" \t\r\n"))
                tokens = ["<space>" if c == " " else c for c in normalized]
                actual = labels[language].get(row["pair_id"])
                if actual != tokens:
                    raise ValueError("common text differs from the actual benchmark auxiliary labels: "
                                     f"split={split} pair_id={row['pair_id']} language={language} "
                                     f"common_tokens={len(tokens)} label_tokens={None if actual is None else len(actual)}")
            result.append({key: row[key] for key in ("pair_id", "split", "ja_text", "en_text", "ja_audio", "en_audio",
                                                     "ja_sha256", "en_sha256") if key in row})
    staging = output.with_name(output.stem + "-input.jsonl")
    write_jsonl(staging, result, overwrite=resume)
    return prepare_manifest(staging, output, resume=resume)


def import_units(manifest, benchmark_units, prepared, output, resume=False):
    rows = load_manifest(manifest, check_audio=False)
    root, prepared = Path(benchmark_units), Path(prepared)
    derived_output(output, root, prepared)
    lock = read_json(root / "unit-lock.json")
    data_lock = read_json(prepared / "data-lock.json")
    if lock != data_lock["unit_configuration"] or lock["hubert_layer"] != 6 or lock["kmeans_clusters"] != 100:
        raise ValueError("benchmark unit configuration mismatch")
    paths = {p.name: p for p in root.glob("manifest.shard-*-of-*.jsonl")}
    # Consolidated inline files are a content-preserving storage migration.
    for path in root.glob("manifest.inline.shard-*-of-*.jsonl"):
        paths[path.name.replace("manifest.inline.", "manifest.")] = path
    selected = {r["pair_id"]: r for r in rows}
    expected = {r["id"]: r for split in {r["split"] for r in rows} for r in tsv(prepared / (split + ".tsv"))}
    store, processed = ArtifactStore(output), set()
    identity = manifest_hash(rows)
    for name in sorted(paths):
        path = paths[name]
        original = root / name
        locked = data_lock["unit_manifest_sha256"]
        if str(path.resolve()) in locked:
            if locked[str(path.resolve())] != file_hash(path):
                raise ValueError("benchmark unit manifest differs from prepared data lock")
        elif original.is_file() and str(original.resolve()) in locked:
            if locked[str(original.resolve())] != file_hash(original):
                raise ValueError("original benchmark unit manifest differs from prepared data lock")
            migration = prepared / "unit-storage-migration.json"
            if migration.is_file() and read_json(migration)["inline"].get(str(path.resolve())) != file_hash(path):
                raise ValueError("inline units differ from recorded storage migration")
        else:
            raise ValueError("unit manifest has no prepared data lock provenance")
        for record in read_jsonl(path):
            pair = record["pair_id"]
            if pair not in selected:
                continue
            if pair in processed:
                raise ValueError("duplicate benchmark unit pair")
            row = selected[pair]
            if record["split"] != row["split"] or any(record.get(k) != lock[k] for k in
                    ("hubert_revision", "hubert_layer", "kmeans_clusters", "kmeans_sha256")):
                raise ValueError("benchmark unit identity/split mismatch")
            if record.get("units_storage") == "inline-v1":
                original_units = record["units_original"]
                reduced_units = record["units_reduced"]
                if record["units_sha256"] != benchmark_digest(dict(original=original_units, reduced=reduced_units)):
                    raise ValueError("corrupt inline benchmark units")
            else:
                if os.environ.get("S2ST_INPUT_MAP"):
                    raise ValueError("Drive-safe import requires benchmark inline unit migration before GPU work")
                original_units = list(map(int, Path(record["units_original_path"]).read_text().split()))
                reduced_units = list(map(int, Path(record["units_reduced_path"]).read_text().split()))
            reduced, runs = reduce_units(original_units)
            if reduced.tolist() != reduced_units or reduced_units != list(map(int, expected[pair]["tgt_audio"].split())):
                raise ValueError("imported units differ from benchmark training targets")
            if len(original_units) != record["unit_count_original"] or len(reduced) != record["unit_count_reduced"]:
                raise ValueError("benchmark unit count mismatch")
            metadata = {"stage": "units", "split": row["split"], "manifest_hash": identity,
                        "config_hash": digest(lock), "unit_config_hash": digest(lock),
                        "ja_audio_hash": row["ja_sha256"], "en_audio_hash": row["en_sha256"],
                        "text_hash": digest([row["ja_text"], row["en_text"]]), "benchmark_import": True}
            if resume and store.path(pair).is_file():
                store.verify_for_resume(pair, metadata)
            else:
                info = sf.info(local_path(row["en_audio"]))
                if info.samplerate != 16000 or info.channels != 1 or (info.frames - 400) // 320 + 1 != len(original_units):
                    raise ValueError("HuBERT frame count and target waveform disagree")
                frames = frame_spans(len(original_units), 16000, info.duration,
                                     [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2])
                store.put(pair, metadata, {"original_units": np.asarray(original_units), "reduced_units": reduced,
                    "runs": runs, "frame_spans": frames, "unit_spans": run_spans(runs, frames),
                    "duration": np.asarray(info.duration)})
            processed.add(pair)
            progress("import benchmark units", len(processed), len(rows))
    if processed != set(selected):
        raise ValueError("missing benchmark units for selected pairs")
    return {"pairs": len(processed), "unit_config_hash": digest(lock)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    child = sub.add_parser("manifest")
    child.add_argument("--common", required=True)
    child.add_argument("--include-test", action="store_true", help="explicit final evaluation export; preprocessing uses train/dev only")
    child = sub.add_parser("units")
    child.add_argument("--manifest", required=True)
    child.add_argument("--benchmark-units", required=True)
    child = sub.add_parser("kokoro")
    child.add_argument("--source", required=True)
    child.add_argument("--summary", required=True)
    child.add_argument("--corpus-root", required=True)
    child = sub.add_parser("corpus")
    child.add_argument("--source", required=True)
    child.add_argument("--summary")
    child.add_argument("--corpus-root", required=True)
    for child in sub.choices.values():
        child.add_argument("--output", required=True)
        child.add_argument("--resume", action="store_true")
    for action in ("manifest", "units"):
        sub.choices[action].add_argument("--prepared", required=True)
    args = parser.parse_args()
    if args.action == "manifest":
        result = {"pairs": len(export_manifest(args.common, args.prepared, args.output, args.include_test, args.resume))}
    elif args.action == "units":
        result = import_units(args.manifest, args.benchmark_units, args.prepared, args.output, args.resume)
    elif args.action == "kokoro":
        result = import_kokoro(args.source, args.summary, args.corpus_root, args.output, args.resume)
    else:
        result = import_corpus(args.source, args.corpus_root, args.output, args.summary, args.resume)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
