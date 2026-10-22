"""Bridge immutable benchmark pairs/units to alignment preprocessing artifacts."""
import argparse
import os
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from .benchmark import tsv, benchmark_digest
from .data.manifest import prepare_manifest, load_manifest, manifest_hash
from .io import ArtifactStore, digest, file_hash, read_json, read_jsonl, write_jsonl
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
            if row["split"] != split or Path(row["ja_audio"]).resolve() != Path(expected[row["pair_id"]]["src_audio"]).resolve():
                raise ValueError("common/prepared split or source mismatch")
            for language in ("ja", "en"):
                # Same explicit ASCII-whitespace policy as the frozen benchmark
                # labels. No Unicode normalization or new dictionary fitting.
                normalized = re.sub(r"[ \t\r\n]+", " ", row[language + "_text"].strip(" \t\r\n"))
                tokens = ["<space>" if c == " " else c for c in normalized]
                if labels[language].get(row["pair_id"]) != tokens:
                    raise ValueError("common text differs from the actual benchmark auxiliary labels")
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
                store.get(pair, metadata)
            else:
                info = sf.info(row["en_audio"])
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
    for child in sub.choices.values():
        child.add_argument("--prepared", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "manifest":
        result = {"pairs": len(export_manifest(args.common, args.prepared, args.output, args.include_test, args.resume))}
    else:
        result = import_units(args.manifest, args.benchmark_units, args.prepared, args.output, args.resume)
    import json
    print(json.dumps(result))


if __name__ == "__main__":
    main()
