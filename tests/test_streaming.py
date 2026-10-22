from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import threading
import numpy as np
import pytest
import torch
import yaml
from alignment_transfer_s2ut.data.streaming import LocalCache, CachedArtifactStore, OrderedLoader, sample_requests
from alignment_transfer_s2ut.io import ArtifactStore, read_jsonl
from alignment_transfer_s2ut.smoke_pipeline import invoke
from alignment_transfer_s2ut.training import load_checkpoint, collate
from alignment_transfer_s2ut.data.manifest import load_manifest
from test_review_improvements import review_data


def test_loading_overlaps_consumer_without_waiting_for_all_inputs():
    gate, next_started = threading.Event(), threading.Event()
    def load(index):
        if index:
            next_started.set()
            assert gate.wait(5), "test consumer failed to release background work"
        return index
    with OrderedLoader(((i, (0, i + 1)) for i in range(20)), load, workers=2, capacity=2) as loader:
        try:
            assert loader.take(1) == [0]  # First item is usable while later input is blocked.
            assert next_started.wait(2)
            assert loader.position == (0, 1)
            assert len(loader.pending) + len(loader.ready) <= 2
        finally:
            gate.set()
        assert loader.take(2) == [1, 2]


def test_loader_order_partial_batches_and_resume_ignore_lookahead():
    rows = list(range(7))
    reference = sample_requests(rows, 23)
    expected = [next(reference)[0] for _ in range(30)]
    with OrderedLoader(sample_requests(rows, 23), lambda row: row, workers=3, capacity=8) as loader:
        assert loader.peek(6) == expected[:6]
        first = loader.take(2)  # Simulate a max_tokens batch shrink, keeping other prepared items.
        position = loader.position
        assert loader.peek(6) == expected[2:8]
    with OrderedLoader(sample_requests(rows, 23, *position), lambda row: row, workers=1,
                       capacity=4, position=position) as resumed:
        rest = [resumed.take(1)[0] for _ in range(28)]
    assert first + rest == expected


def test_loader_failure_propagates_without_advancing_or_skipping():
    def load(row):
        if row == 1:
            raise FileNotFoundError("required input missing")
        return row
    with OrderedLoader(((i, (0, i + 1)) for i in range(4)), load, workers=2, capacity=3) as loader:
        assert loader.take(1) == [0]
        with pytest.raises(FileNotFoundError, match="required input"):
            loader.take(1)
        assert loader.position == (0, 1)


def test_cache_reuses_concurrent_loads_and_rejects_corruption(tmp_path):
    cache = LocalCache(tmp_path / "cache", 100000, reserve_bytes=0)
    calls, lock = [], threading.Lock()
    def create():
        with lock:
            calls.append(1)
        return {"source": "verified"}, {"array": np.arange(100, dtype=np.float32)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cache.get_or_create("sample", {"hash": "fixed"}, create), range(8)))
    assert len(calls) == 1
    assert all(np.array_equal(item[1]["array"], np.arange(100)) for item in results)
    entry = next((tmp_path / "cache/items").glob("*.npz"))
    with entry.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        cache.get_or_create("sample", {"hash": "fixed"}, create)
    assert len(calls) == 1  # Never hide a corrupt cache entry by silently replacing it.


def test_cache_budget_eviction_bypass_and_owned_directory_guard(tmp_path):
    cache = LocalCache(tmp_path / "cache", 24000, reserve_bytes=0)
    def create(i):
        return {"index": i}, {"array": np.random.default_rng(i).random(2048, dtype=np.float32)}
    for i in range(8):
        cache.get_or_create("item", i, lambda i=i: create(i))
        assert sum(path.stat().st_size for path in cache.items.iterdir()) <= 24000
    assert len(list(cache.items.glob("*.npz"))) < 8
    small = LocalCache(tmp_path / "small", 100, reserve_bytes=0)
    small.get_or_create("large", 0, lambda: create(0))
    assert not list(small.items.glob("*.npz"))
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "keep.txt").write_text("user data")
    with pytest.raises(ValueError, match="empty directory"):
        LocalCache(unknown, 100)
    assert (unknown / "keep.txt").read_text() == "user data"


def test_artifact_cache_checks_identity_and_preserves_original_file_hash(tmp_path):
    original = ArtifactStore(tmp_path / "units")
    original.put("pair", {"split": "train", "unit_config_hash": "a"}, {"units": np.arange(3)})
    cache = LocalCache(tmp_path / "cache", 100000, reserve_bytes=0)
    cached = CachedArtifactStore(tmp_path / "units", cache)
    assert cached.hash("pair") == original.hash("pair")
    cached.get("pair", {"unit_config_hash": "a"})
    original.put("pair", {"split": "train", "unit_config_hash": "b"}, {"units": np.arange(4)}, overwrite=True)
    with pytest.raises(ValueError, match="provenance"):
        cached.get("pair", {"unit_config_hash": "a"})


def test_feature_cache_reuses_features_and_detects_changed_source(review_data, tmp_path, monkeypatch):
    import alignment_transfer_s2ut.training as training
    row = dict(load_manifest(review_data / "manifest.jsonl")[0])
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(row["ja_audio"], source / "ja.wav")
    row["ja_audio"] = str(source / "ja.wav")
    cache = LocalCache(tmp_path / "cache", 1000000, reserve_bytes=0)
    count, original = [], training.features
    def counted(*args, **kwargs):
        count.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "features", counted)
    def load():
        return collate([row], "speech", None, None, "smoke", ArtifactStore(review_data / "units"), 8,
                       "cpu", cache=cache, manifest_hashes=True)
    first, second = load(), load()
    assert len(count) == 1
    assert torch.equal(first[0], second[0])
    with (source / "ja.wav").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source audio changed"):
        load()


def test_training_starts_before_loading_dev_payloads(review_data, tmp_path, monkeypatch):
    from alignment_transfer_s2ut.s2ut.smoke import SmokeModel
    seen, first_forward = [], []
    original_get, original_forward = ArtifactStore.get, SmokeModel.forward
    def tracked_get(self, pair, expected=None):
        seen.append(pair)
        return original_get(self, pair, expected)
    def tracked_forward(self, *args, **kwargs):
        if not first_forward:
            first_forward.append(list(seen))
        return original_forward(self, *args, **kwargs)
    monkeypatch.setattr(ArtifactStore, "get", tracked_get)
    monkeypatch.setattr(SmokeModel, "forward", tracked_forward)
    invoke("train_s2ut", "--trainer", "legacy", "--manifest", review_data / "manifest.jsonl", "--config", review_data / "tiny.yaml",
           "--backend", "smoke", "--units", review_data / "units", "--loader-workers", 2,
           "--batch-size", 1, "--update-frequency", 1, "--max-updates", 1, "--output", tmp_path / "train")
    assert first_forward and "2" not in first_forward[0] and "3" not in first_forward[0]
    assert "2" in seen  # Dev is checked when validation actually consumes it.


def test_lazy_mixed_units_fail_when_consumed_without_skipping_updates(review_data, tmp_path):
    units = tmp_path / "units"
    shutil.copytree(review_data / "units", units)
    store = ArtifactStore(units)
    metadata, arrays = store.get("1")
    metadata["unit_config_hash"] = "different-config"
    store.put("1", metadata, arrays, overwrite=True)
    with pytest.raises(ValueError, match="provenance"):
        invoke("train_s2ut", "--trainer", "legacy", "--manifest", review_data / "manifest.jsonl", "--config", review_data / "tiny.yaml",
               "--backend", "smoke", "--units", units, "--loader-workers", 2, "--batch-size", 1,
               "--update-frequency", 1, "--max-updates", 2, "--output", tmp_path / "train")
    assert load_checkpoint(tmp_path / "train/checkpoint_last.pt")["step"] == 1


@pytest.mark.parametrize("method,prior", [("baseline", "t2tt"), ("distillation", "t2tt"),
    ("guidance", "t2tt"), ("annealed_guidance", "t2tt"), ("annealed_guidance", "monotonic")])
def test_streaming_training_matches_serial_and_resumes_with_other_workers(review_data, tmp_path, method, prior):
    torch.set_num_threads(1)
    common = ["--manifest", review_data / "manifest.jsonl", "--config", review_data / "tiny.yaml",
        "--backend", "smoke", "--units", review_data / "units", "--teacher", review_data / "teacher",
        "--method", method, "--prior-type", prior, "--batch-size", 2, "--max-tokens", 65,
        "--update-frequency", 2, "--warmup-updates", 0, "--anneal-steps", 1]
    invoke("train_s2ut", "--trainer", "legacy", *common, "--max-updates", 3, "--output", tmp_path / "serial")
    optimized = [*common, "--loader-workers", 2, "--prefetch-batches", 3, "--cache-dir", tmp_path / "cache"]
    invoke("train_s2ut", "--trainer", "legacy", *optimized, "--max-updates", 3, "--output", tmp_path / "full")
    invoke("train_s2ut", "--trainer", "legacy", *optimized, "--max-updates", 1, "--output", tmp_path / "resume")
    invoke("train_s2ut", "--trainer", "legacy", *optimized, "--loader-workers", 1, "--prefetch-batches", 1,
           "--max-updates", 3, "--output", tmp_path / "resume", "--resume")
    states = [load_checkpoint(tmp_path / name / "checkpoint_last.pt") for name in ("serial", "full", "resume")]
    for key, value in states[0]["model"].items():
        assert all(torch.equal(value, state["model"][key]) for state in states[1:]), key
    assert len({(s["epoch"], s["cursor"]) for s in states}) == 1
    assert all(torch.equal(states[0]["torch_rng"], state["torch_rng"]) for state in states[1:])
    logs = read_jsonl(tmp_path / "full/metrics.jsonl")
    assert all(row["input/data_wait_seconds"] >= 0 and row["input/optimization_seconds"] > 0 for row in logs)


@pytest.mark.fairseq
def test_native_strong_augmentation_stays_identical_with_prefetch_and_resume(review_data, tmp_path):
    pytest.importorskip("fairseq")
    from alignment_transfer_s2ut.recipes import load_recipe
    recipe = load_recipe("strong")
    recipe.pop("name")
    recipe["specaugment"].update(freq_mask_F=3, time_mask_T=5)
    recipe["auxiliary"].update(source_layer=1, target_layer=1, decoder_ctc_layer=1, ctc_vocab_size=16)
    recipe["auxiliary"]["decoder_args"].update(decoder_layers=1, decoder_embed_dim=16,
        decoder_ffn_embed_dim=32, decoder_attention_heads=2, dropout=.1)
    path = tmp_path / "strong.yaml"
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    common = ["--manifest", review_data / "manifest.jsonl", "--config", review_data / "tiny.yaml",
        "--units", review_data / "units", "--recipe", "strong", "--recipe-config", path,
        "--method", "annealed_guidance", "--prior-type", "monotonic", "--batch-size", 2,
        "--max-tokens", 65, "--update-frequency", 2, "--warmup-updates", 0]
    invoke("train_s2ut", "--trainer", "legacy", *common, "--max-updates", 3, "--output", tmp_path / "serial")
    optimized = [*common, "--loader-workers", 2, "--cache-dir", tmp_path / "cache"]
    invoke("train_s2ut", "--trainer", "legacy", *optimized, "--max-updates", 1, "--output", tmp_path / "resume")
    invoke("train_s2ut", "--trainer", "legacy", *optimized, "--max-updates", 3, "--output", tmp_path / "resume", "--resume")
    first, second = [load_checkpoint(tmp_path / name / "checkpoint_last.pt") for name in ("serial", "resume")]
    for key in first["model"]:
        assert torch.equal(first["model"][key], second["model"][key]), key
    assert np.array_equal(first["numpy_rng"][1], second["numpy_rng"][1])
