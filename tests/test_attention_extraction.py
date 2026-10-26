"""Batched teacher-forced attention, legacy provenance and partial resume."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

from alignment_transfer_s2ut import training as t
from alignment_transfer_s2ut.cli import parser_for
from alignment_transfer_s2ut.config import BackboneConfig
from alignment_transfer_s2ut.data.manifest import manifest_hash
from alignment_transfer_s2ut.io import ArtifactStore, digest, file_hash, write_jsonl
from alignment_transfer_s2ut.smoke_pipeline import TINY_CONFIG
from alignment_transfer_s2ut.tokenization.character import CharacterTokenizer
from test_benchmark_colab import benchmark_runtime


@pytest.fixture
def attention_data(tmp_path):
    torch.set_num_threads(1)
    rows = []
    for i, (split, ja, en) in enumerate([
            ("train", "あい", "ab"), ("train", "い あいあ", "a bba"),
            ("train", "あいあ", "baababa"), ("train", "あ あい", "b a"),
            ("dev", "いあぬ", "abz"), ("dev", "ぬ", "z"), ("test", "ああ", "ba")]):
        rows.append(dict(pair_id=str(i), split=split, ja_text=ja, en_text=en,
                         ja_audio=str(tmp_path / f"ja-{i}.wav"), en_audio=str(tmp_path / f"en-{i}.wav")))
    manifest = tmp_path / "manifest.jsonl"
    # Extraction never needs audio payloads, even to resume old packets.
    write_jsonl(manifest, rows)
    return rows, manifest


def save_checkpoint(root, rows, backend, config=None):
    config = config or BackboneConfig(**TINY_CONFIG["backbone"])
    source, target = [CharacterTokenizer.fit(rows, field) for field in ("ja_text", "en_text")]
    t.seed_all(7)
    model = t.build_model(config, "text", source.vocabulary, target.vocabulary, backend)
    saved = dict(modality="text", backbone=config.asdict(), source_vocabulary=source.vocabulary,
                 target_vocabulary=target.vocabulary, backend=backend, model=deepcopy(model.state_dict()),
                 recipe={"seed": 7}, train_hash=manifest_hash([r for r in rows if r["split"] == "train"]))
    checkpoint = root / "checkpoint.pt"
    torch.save(saved, checkpoint)
    return checkpoint, saved


def arguments(manifest, checkpoint, output, *extra):
    return parser_for("extract_t2tt_attention").parse_args([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint), "--output", str(output), *map(str, extra)])


def legacy_packets(rows, saved, checkpoint, heads=(0,), device="cpu", layer=-1):
    source, target = [CharacterTokenizer(saved[name + "_vocabulary"]) for name in ("source", "target")]
    model = t.checkpoint_model(saved, device).eval()
    heads = None if heads is None else list(heads)
    packets = {}
    for row in rows:
        values = t.collate([row], "text", source, target, saved["backend"], None, 80, device)
        with torch.no_grad():
            _, raw, _, _ = model(*values[:3], layer=layer, heads=heads)
        matrix = raw[0, :len(row["en_text"]), :len(row["ja_text"])].cpu().numpy()
        matrix = matrix * (np.asarray(target.encode(row["en_text"])) != 3)[:, None] * (
            np.asarray(source.encode(row["ja_text"])) != 3)[None, :]
        meta = dict(stage="t2tt_attention", split=row["split"], manifest_hash=manifest_hash(rows),
            text_hash=digest([row["ja_text"], row["en_text"]]), backend=saved["backend"],
            mt_checkpoint_hash=file_hash(checkpoint), mt_architecture_hash=digest(saved["backbone"]),
            tokenizer_hash=digest([source.fingerprint, target.fingerprint]), layer=layer, heads=heads,
            teacher_forcing=True, training_split="train")
        packets[row["pair_id"]] = meta, matrix
    return packets


@pytest.mark.parametrize("backend", ["smoke", "fairseq"])
@pytest.mark.parametrize("layer", [0, -1])
def test_random_lengths_and_nonfinal_decoder_layer_match_unbatched(tmp_path, backend, layer):
    if backend == "fairseq" and importlib.util.find_spec("fairseq") is None:
        pytest.skip("pinned fairseq not installed")
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261027)
    rows = []
    for index in range(40):
        ja = "".join(rng.choice(list("あいう漢字😀 "), size=int(rng.integers(1, 97)))) + str(index)
        en = "".join(rng.choice(list('abcXYZ "'), size=int(rng.integers(1, 81)))) + str(index)
        if index == 0:
            ja += "0123456789"
            en += "0123456789"
        if index >= 32:
            ja += "ぬ"
            en += "Ω"
        rows.append(dict(pair_id=str(index), split="train" if index < 32 else "dev", ja_text=ja, en_text=en,
                         ja_audio=str(tmp_path / f"ja-{index}.wav"), en_audio=str(tmp_path / f"en-{index}.wav")))
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    values = dict(TINY_CONFIG["backbone"], encoder_layers=2, decoder_layers=2)
    checkpoint, saved = save_checkpoint(tmp_path, rows, backend, BackboneConfig(**values))
    heads = [1] if layer == 0 else None
    reference = legacy_packets(rows, saved, checkpoint, heads=heads, layer=layer)
    args = arguments(manifest, checkpoint, tmp_path / "attention", "--layer", layer,
                     "--max-tokens", 512, "--max-attention-cells", 24000,
                     *([] if heads is None else ["--heads", *heads]))
    result = t.extract_attention(args)
    assert result["pairs"] == 40 and result["forwards"] < 40
    store = ArtifactStore(args.output)
    for pair, (expected_meta, matrix) in reference.items():
        _, packet = store.get(pair, expected_meta)
        assert packet["attention"].shape == matrix.shape
        np.testing.assert_allclose(packet["attention"], matrix, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("backend", ["smoke", "fairseq"])
@pytest.mark.parametrize("heads", [None, [0], [0, 1]])
def test_batched_attention_matches_single_pair_and_reuses_legacy_packets(tmp_path, attention_data, backend, heads, monkeypatch):
    if backend == "fairseq" and importlib.util.find_spec("fairseq") is None:
        pytest.skip("pinned fairseq not installed")
    rows, manifest = attention_data
    checkpoint, saved = save_checkpoint(tmp_path, rows, backend)
    reference = legacy_packets(rows, saved, checkpoint, heads=heads)
    store = ArtifactStore(tmp_path / "attention")
    for pair in ("0", "4"):
        meta, matrix = reference[pair]
        store.put(pair, meta, {"attention": matrix})
    old_hashes = {pair: store.hash(pair) for pair in ("0", "4")}
    # Interrupted between atomic NPZ publication and external checksum write.
    store.path("0").with_suffix(".sha256.json").unlink()
    hashes = []
    original_hash = t.file_hash
    def track_hash(path):
        hashes.append(Path(path))
        return original_hash(path)
    monkeypatch.setattr(t, "file_hash", track_hash)
    args = arguments(manifest, checkpoint, store.root, "--batch-size", 3, "--resume",
                     *([] if heads is None else ["--heads", *heads]))
    result = t.extract_attention(args)
    assert result["pairs"] == 7 and result["reused"] == 2 and result["extracted"] == 5
    assert result["forwards"] == 2
    assert hashes == [checkpoint]  # One full checkpoint read for SHA, not one per pair.
    assert {pair: store.hash(pair) for pair in old_hashes} == old_hashes
    assert store.path("0").with_suffix(".sha256.json").is_file()
    for pair, (expected_meta, expected_matrix) in reference.items():
        meta, packet = store.get(pair, expected_meta)
        assert packet["attention"].shape == expected_matrix.shape
        np.testing.assert_allclose(packet["attention"], expected_matrix, atol=2e-6, rtol=2e-5)
        assert (packet["attention"] >= 0).all()
    assert not store.get("5")[1]["attention"].any()  # All-OOV held-out pair stays invalid.
    hashes.clear()
    assert t.extract_attention(args)["forwards"] == 0
    assert hashes == [checkpoint]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="batched native attention parity requires CUDA")
def test_batched_native_attention_matches_single_pair_on_cuda(tmp_path, attention_data):
    pytest.importorskip("fairseq")
    rows, manifest = attention_data
    checkpoint, saved = save_checkpoint(tmp_path, rows, "fairseq")
    reference = legacy_packets(rows, saved, checkpoint, heads=None, device="cuda")
    output = tmp_path / "attention"
    result = t.extract_attention(arguments(manifest, checkpoint, output, "--device", "cuda"))
    assert result["forwards"] == 1
    for pair, (expected_meta, matrix) in reference.items():
        _, packet = ArtifactStore(output).get(pair, expected_meta)
        np.testing.assert_allclose(packet["attention"], matrix, atol=2e-6, rtol=2e-5)


def test_batch_size_one_is_exact_and_interrupted_writes_resume(tmp_path, attention_data, monkeypatch):
    rows, manifest = attention_data
    checkpoint, saved = save_checkpoint(tmp_path, rows, "smoke")
    reference = legacy_packets(rows, saved, checkpoint, heads=None)
    args = arguments(manifest, checkpoint, tmp_path / "attention", "--batch-size", 1)
    store = ArtifactStore(args.output)
    original = ArtifactStore.put
    calls = []
    def interrupted(self, pair, *args, **kwargs):
        calls.append(pair)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return original(self, pair, *args, **kwargs)
    monkeypatch.setattr(ArtifactStore, "put", interrupted)
    with pytest.raises(KeyboardInterrupt):
        t.extract_attention(args)
    assert len(list(store.root.glob("*.npz"))) == 2
    before = {r["pair_id"]: store.hash(r["pair_id"]) for r in rows[:2]}
    monkeypatch.setattr(ArtifactStore, "put", original)
    args.resume = True
    result = t.extract_attention(args)
    assert result["reused"] == 2 and result["forwards"] == 5
    assert {pair: store.hash(pair) for pair in before} == before
    for pair, (expected_meta, matrix) in reference.items():
        _, packet = store.get(pair, expected_meta)
        np.testing.assert_array_equal(packet["attention"], matrix)


@pytest.mark.parametrize("option,budget", [("--max-tokens", 4), ("--max-attention-cells", 5)])
def test_oversized_pair_is_not_silently_skipped(tmp_path, attention_data, option, budget):
    rows, manifest = attention_data
    checkpoint, _ = save_checkpoint(tmp_path, rows, "smoke")
    args = arguments(manifest, checkpoint, tmp_path / "attention", option, budget)
    with pytest.raises(ValueError, match="pair exceeds attention batch budget: 0"):
        t.extract_attention(args)
    assert not list((tmp_path / "attention").glob("*.npz"))


@pytest.mark.parametrize("budget_option,budget", [("--max-tokens", 22), ("--max-attention-cells", 80)])
def test_attention_batches_respect_padded_budgets(tmp_path, attention_data, monkeypatch, budget_option, budget):
    rows, manifest = attention_data
    checkpoint, _ = save_checkpoint(tmp_path, rows, "smoke")
    original = t.checkpoint_model
    batches = []
    def model(saved, device):
        network = original(saved, device)
        def record(module, inputs):
            source, lengths, previous = inputs
            b, s, u = len(source), source.size(1), previous.size(1)
            assert b <= 4
            assert (b * (s + u) if budget_option == "--max-tokens" else b * s * u) <= budget
            batches.append(b)
        network.register_forward_pre_hook(record)
        return network
    monkeypatch.setattr(t, "checkpoint_model", model)
    result = t.extract_attention(arguments(manifest, checkpoint, tmp_path / "attention", "--batch-size", 4, budget_option, budget))
    assert result["pairs"] == 7 and sum(batches) == 7 and len(batches) > 2


def test_checkpoint_mutation_and_mixed_provenance_are_rejected(tmp_path, attention_data, monkeypatch):
    rows, manifest = attention_data
    checkpoint, saved = save_checkpoint(tmp_path, rows, "smoke")
    original = t.checkpoint_model
    def model(saved, device):
        network = original(saved, device)
        def mutate(module, inputs):
            with checkpoint.open("ab") as stream:
                stream.write(b"changed")
        network.register_forward_pre_hook(mutate)
        return network
    monkeypatch.setattr(t, "checkpoint_model", model)
    args = arguments(manifest, checkpoint, tmp_path / "attention")
    with pytest.raises(ValueError, match="checkpoint changed"):
        t.extract_attention(args)
    assert not list((tmp_path / "attention").glob("*.npz"))
    monkeypatch.setattr(t, "checkpoint_model", original)
    checkpoint, _ = save_checkpoint(tmp_path, rows, "smoke")
    store = ArtifactStore(args.output)
    meta, matrix = legacy_packets(rows, saved, checkpoint)["0"]
    store.put("0", {**meta, "mt_checkpoint_hash": "wrong"}, {"attention": matrix})
    args.resume = True
    with pytest.raises(ValueError, match="provenance mismatch: mt_checkpoint_hash"):
        t.extract_attention(args)


@pytest.mark.parametrize("option,value", [("--batch-size", 0), ("--max-tokens", 0), ("--max-attention-cells", -1)])
def test_invalid_attention_budgets_fail_before_loading(tmp_path, option, value):
    with pytest.raises(ValueError, match="must be positive"):
        t.extract_attention(arguments("unused", "unused", tmp_path, option, value))


def test_attention_progress_shows_actual_counts_rate_and_resume(tmp_path, attention_data, capsys):
    rows, manifest = attention_data
    checkpoint, _ = save_checkpoint(tmp_path, rows, "smoke")
    args = arguments(manifest, checkpoint, tmp_path / "attention", "--batch-size", 3)
    result = t.extract_attention(args)
    assert result["forwards"] == 3 and result["extracted"] == 7
    output = capsys.readouterr().err
    assert "[T2TT attention] 7/7 new=7 reused=0" in output
    assert "pairs_per_second=" in output and "eta_seconds=0" in output
    args.resume = True
    result = t.extract_attention(args)
    assert result["forwards"] == 0
    assert "[T2TT attention] 7/7 new=0 reused=7" in capsys.readouterr().err


def test_legacy_completed_snapshot_missing_a_pair_is_revalidated(tmp_path, attention_data, benchmark_runtime):
    from alignment_transfer_s2ut import colab_runtime as c
    from alignment_transfer_s2ut.io import read_json, write_json
    rows, manifest = attention_data
    checkpoint, saved = save_checkpoint(tmp_path, rows, "smoke")
    work, backup = tmp_path / "attention", tmp_path / "drive"
    store = ArtifactStore(work)
    for pair, (meta, matrix) in legacy_packets(rows, saved, checkpoint, heads=None).items():
        store.put(pair, meta, {"attention": matrix})
    command = [sys.executable, "-m", "alignment_transfer_s2ut.extract_t2tt_attention",
               "--manifest", str(manifest), "--checkpoint", str(checkpoint), "--output", str(work), "--resume"]
    identity = c.identity_for(command, work)
    # Reproduce an old publisher excluding one unreceipted NPZ but including
    # the old completion marker. A new VM has no local copy of that missing pair.
    frozen = tmp_path / "legacy-snapshot"
    frozen.mkdir()
    for path in work.iterdir():
        if path.stem.startswith(store.path("0").stem):
            continue
        shutil.copyfile(path, frozen / path.name)
    write_json(frozen / "stage-complete.json", {"identity": digest(identity)})
    benchmark_runtime.publish_packed(frozen, backup, identity, 1)
    work.rename(tmp_path / "lost-vm")
    assert c.stage_session(command, work, backup)["status"] == "COMPLETE"
    assert store.path("0").is_file() and store.path("0").with_suffix(".sha256.json").is_file()
    assert read_json(work / "stage-complete.json")["artifact_receipts_verified"] is True
    assert all(store.get(row["pair_id"])[0]["pair_id"] == row["pair_id"] for row in rows)
    work.rename(tmp_path / "second-lost-vm")
    assert c.stage_session(command, work, backup)["reused"]
    assert store.path("0").is_file()
