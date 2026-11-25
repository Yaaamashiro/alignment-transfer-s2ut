"""Reconstruct and attest the registered Fairseq Dev-loss population."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


TAIL_REASON = "fairseq_final_batch_not_divisible"


def ids_hash(ids):
    return sha256(json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def checked_ids(expected, observed, *, stage):
    counts = Counter(observed)
    expected = set(expected)
    missing = sorted(expected - counts.keys())
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    extra = sorted(counts.keys() - expected)
    if missing or duplicates or extra:
        raise ValueError(f"Dev {stage} missing IDs: {missing[:20]} ({len(missing)} total); "
                         f"duplicate IDs: {duplicates[:20]}; unexpected IDs: {extra[:20]}")


def native_selection(dataset, settings, positions, seed):
    """Expose the raw final batch before the pinned wrapper applies its tail rule."""
    from fairseq.data import data_utils

    ids = list(dataset.ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Dev dataset has empty or duplicated registered IDs")
    with data_utils.numpy_seed(seed + 1):
        ordered = dataset.ordered_indices()
    checked_ids(range(len(ids)), map(int, ordered), stage="ordering")
    indices, ignored = dataset.filter_indices_by_size(ordered, positions)
    if ignored:
        reasons = [(ids[i], dataset.size(i)) for i in ignored]
        raise ValueError(f"Dev size filtering rejected IDs and sizes {reasons[:20]} under limits {positions}")
    multiple = settings.required_batch_size_multiple
    if not len(indices) or multiple < 1 or dataset.get_batch_shapes() is not None:
        raise ValueError("Unsupported registered Dev batching policy")
    if multiple == 1:
        return list(map(int, indices)), []

    from fairseq.data.data_utils_fast import batch_by_size_fn

    raw_batches = batch_by_size_fn(indices, dataset.num_tokens, settings.max_tokens_valid or -1,
                                   settings.batch_size_valid or -1, multiple)
    checked_ids(range(len(ids)), (int(i) for batch in raw_batches for i in batch), stage="raw batching")
    tail = raw_batches[-1] if len(raw_batches[-1]) % multiple else []
    omitted = [{"id": ids[i], "source_frames": int(dataset.size(i)[0]), "target_units": int(dataset.size(i)[1]),
                "reason": TAIL_REASON} for i in tail]
    retained = [int(i) for batch in (raw_batches[:-1] if omitted else raw_batches) for i in batch]
    if not retained:
        raise ValueError(f"Fairseq Dev tail removal would leave no samples: {omitted}")
    return retained, omitted


def observed_ids(dataset, indices):
    ids = dataset.ids
    if any(not 0 <= index < len(ids) for index in indices):
        raise ValueError("Dev iterator contains an out-of-range Fairseq sample index")
    return [ids[index] for index in indices]


def attest_selection(dataset, selection, excluded, batch_multiple, prepared_sha, observed, batches):
    expected = observed_ids(dataset, selection)
    delivered = observed_ids(dataset, observed)
    checked_ids(expected, delivered, stage="Criterion batches")
    return {"prepared_sha256": prepared_sha, "prepared_count": len(dataset), "prepared_ids_sha256": ids_hash(dataset.ids),
            "expected_ids_sha256": ids_hash(expected),
            "evaluated_ids": sorted(delivered), "excluded": excluded, "batch_size_multiple": batch_multiple,
            "batch_count": batches}


def audit(baseline, prepared=None, manifest=None):
    from types import SimpleNamespace
    from fairseq import utils
    from fairseq.data import Dictionary
    from fairseq.data.audio.data_cfg import S2SDataConfig
    from fairseq.data.audio.speech_to_speech_dataset import SpeechToSpeechDatasetCreator
    from fairseq.tasks.fairseq_task import FairseqTask
    from .io import digest, file_hash, local_path, read_json
    from .s2ut.fairseq_extension import FAIRSEQ_REVISION, verify_fairseq

    verify_fairseq()
    record = read_json(baseline)
    if (record.get("format") != "native-fairseq-s2ut-comparison-v1" or record.get("record_hash") !=
            digest({key: value for key, value in record.items() if key != "record_hash"})):
        raise ValueError("Corrupt Baseline registration")
    root = Path(prepared or record["data"])
    paths = {name: local_path(root / name) for name in ("dev.tsv", "config.yaml", "dict.txt")}
    for name, path in paths.items():
        if file_hash(path) != record["prepared"]["files"][name]:
            raise ValueError(f"Audit input differs from the registered Baseline: {name}")
    config = record["config"]
    settings = SimpleNamespace(**config["dataset"])
    task, model = config["task"], config["model"]
    if settings.max_valid_steps is not None or settings.skip_invalid_size_inputs_valid_test:
        raise ValueError("Audit requires the registered uncapped Dev size-filter policy")
    positions = utils.resolve_max_positions((task["max_source_positions"], task["max_target_positions"]),
                                            (model.get("max_source_positions"), model.get("max_target_positions")))
    seed = config["common"]["seed"]
    dataset = SpeechToSpeechDatasetCreator.from_tsv(str(paths["dev.tsv"].parent), S2SDataConfig(paths["config.yaml"]),
        "dev", False, 1, seed, task["target_is_code"], Dictionary.load(str(paths["dict.txt"])), task["n_frames_per_step"])
    if manifest is not None:
        from .data.manifest import load_manifest
        registered = [row["pair_id"] for row in load_manifest(manifest, split="dev", check_audio=False, require_audio=False)]
        checked_ids(registered, dataset.ids, stage="manifest-to-prepared")
    expected, excluded = native_selection(dataset, settings, positions, seed)
    iterator = FairseqTask(SimpleNamespace()).get_batch_iterator(dataset, max_tokens=settings.max_tokens_valid,
        max_sentences=settings.batch_size_valid, max_positions=positions, ignore_invalid_inputs=False,
        required_batch_size_multiple=settings.required_batch_size_multiple, seed=seed, epoch=1, skip_remainder_batch=False)
    seen = [int(i) for batch in iterator.frozen_batches for i in batch]
    result = attest_selection(dataset, expected, excluded, settings.required_batch_size_multiple,
                              record["prepared"]["files"]["dev.tsv"], seen, len(iterator.frozen_batches))
    return {**result, "status": "PASS", "evaluated_count": len(seen), "max_positions": positions,
            "max_tokens_valid": settings.max_tokens_valid, "fairseq_revision": FAIRSEQ_REVISION}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, help="Local copy of the registered Dev metadata")
    parser.add_argument("--manifest", type=Path, help="Optional full transfer manifest for Dev ID comparison")
    args = parser.parse_args()
    print(json.dumps(audit(args.baseline, args.prepared, args.manifest), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
