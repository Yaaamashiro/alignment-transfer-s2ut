"""Batched teacher-forced attention, legacy provenance and partial resume."""
from copy import deepcopy
import importlib.util
from pathlib import Path

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


def save_checkpoint(root, rows, backend):
    config = BackboneConfig(**TINY_CONFIG["backbone"])
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


def legacy_packets(rows, saved, checkpoint):
    source, target = [CharacterTokenizer(saved[name + "_vocabulary"]) for name in ("source", "target")]
    model = t.checkpoint_model(saved).eval()
    packets = {}
    for row in rows:
        values = t.collate([row], "text", source, target, saved["backend"], None, 80, "cpu")
        with torch.no_grad():
            _, raw, _, _ = model(*values[:3], layer=-1, heads=[0])
        matrix = raw[0, :len(row["en_text"]), :len(row["ja_text"])].numpy()
        matrix = matrix * (np.asarray(target.encode(row["en_text"])) != 3)[:, None] * (
            np.asarray(source.encode(row["ja_text"])) != 3)[None, :]
        meta = dict(stage="t2tt_attention", split=row["split"], manifest_hash=manifest_hash(rows),
            text_hash=digest([row["ja_text"], row["en_text"]]), backend=saved["backend"],
            mt_checkpoint_hash=file_hash(checkpoint), mt_architecture_hash=digest(saved["backbone"]),
            tokenizer_hash=digest([source.fingerprint, target.fingerprint]), layer=-1, heads=[0],
            teacher_forcing=True, training_split="train")
        packets[row["pair_id"]] = meta, matrix
    return packets


@pytest.mark.parametrize("backend", ["smoke", "fairseq"])
def test_batched_attention_matches_single_pair_and_reuses_legacy_packets(tmp_path, attention_data, backend, monkeypatch):
    if backend == "fairseq" and importlib.util.find_spec("fairseq") is None:
        pytest.skip("pinned fairseq not installed")
    rows, manifest = attention_data
    checkpoint, saved = save_checkpoint(tmp_path, rows, backend)
    reference = legacy_packets(rows, saved, checkpoint)
    store = ArtifactStore(tmp_path / "attention")
    for pair in ("0", "4"):
        meta, matrix = reference[pair]
        store.put(pair, meta, {"attention": matrix})
    old_hashes = {pair: store.hash(pair) for pair in ("0", "4")}
    hashes = []
    original_hash = t.file_hash
    def track_hash(path):
        hashes.append(Path(path))
        return original_hash(path)
    monkeypatch.setattr(t, "file_hash", track_hash)
    args = arguments(manifest, checkpoint, store.root, "--batch-size", 3, "--heads", 0, "--resume")
    result = t.extract_attention(args)
    assert result["pairs"] == 7 and result["reused"] == 2 and result["extracted"] == 5
    assert result["forwards"] == 2
    assert hashes == [checkpoint]  # One full checkpoint read for SHA, not one per pair.
    assert {pair: store.hash(pair) for pair in old_hashes} == old_hashes
    for pair, (expected_meta, expected_matrix) in reference.items():
        meta, packet = store.get(pair, expected_meta)
        assert packet["attention"].shape == expected_matrix.shape
        np.testing.assert_allclose(packet["attention"], expected_matrix, atol=2e-6, rtol=2e-5)
        assert (packet["attention"] >= 0).all()
    assert not store.get("5")[1]["attention"].any()  # All-OOV held-out pair stays invalid.
    hashes.clear()
    assert t.extract_attention(args)["forwards"] == 0
    assert hashes == [checkpoint]


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
