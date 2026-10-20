from pathlib import Path
import numpy as np
import pytest
from alignment_transfer_s2ut.config import BackboneConfig, compare
from alignment_transfer_s2ut.data.manifest import validate
from alignment_transfer_s2ut.io import ArtifactStore
from alignment_transfer_s2ut.tokenization.character import CharacterTokenizer, UNK


def row(pair, split="train", ja="あ", en="a"):
    return dict(pair_id=pair, split=split, ja_text=ja, en_text=en,
                ja_audio=pair + "-ja.wav", en_audio=pair + "-en.wav")


def test_architecture_consistency_and_deliberate_difference():
    root = Path(__file__).resolve().parents[1]
    left = BackboneConfig.load(root / "configs/t2tt/fisher.yaml")
    right = BackboneConfig.load(root / "configs/s2ut/fisher.yaml")
    assert all(r["match"] for r in compare(left, right).values())
    assert left.encoder_attention_heads == 4 and left.decoder_attention_heads == 8
    assert not compare(left, BackboneConfig(dropout=0.2))["dropout"]["match"]


def test_codepoints_no_normalization_literal_specials_and_train_only_vocabulary():
    text = "a \t\nＡe\u0301😀<space>"
    rows = [row("1", en=text), row("2", "dev", en="z")]
    tokenizer = CharacterTokenizer.fit(rows, "en_text")
    assert len(tokenizer.tokenize(text)) == len(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode("z") == [UNK]
    assert tokenizer.fingerprint == CharacterTokenizer(tokenizer.vocabulary).fingerprint


def test_leakage_guards():
    with pytest.raises(ValueError, match="pair_id"):
        validate([row("1"), row("1", "test")], False)
    with pytest.raises(ValueError, match="leakage"):
        validate([row("1"), row("2", "dev")], False)
    with pytest.raises(ValueError, match="split"):
        validate([row("1", "validation")], False)


def test_artifact_hash_provenance_and_nonfinite_guard(tmp_path):
    store = ArtifactStore(tmp_path)
    store.put("../arbitrary/id", {"stage": "test", "split": "train"}, {"matrix": np.eye(2)})
    assert store.path("../arbitrary/id").parent == tmp_path
    store.get("../arbitrary/id", {"split": "train"})
    with pytest.raises(ValueError, match="provenance"):
        store.get("../arbitrary/id", {"split": "test"})
    with store.path("../arbitrary/id").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        store.get("../arbitrary/id")
    with pytest.raises(ValueError, match="finite"):
        store.put("nan", {}, {"matrix": np.array([[np.nan]])})


def test_artifact_resume_after_external_checksum_write_interrupted(tmp_path):
    store = ArtifactStore(tmp_path)
    store.put("pair", {"split": "train"}, {"array": np.eye(2)})
    store.path("pair").with_suffix(".sha256.json").unlink()
    metadata, arrays = store.get("pair", {"split": "train"})
    assert metadata["payload_hash"]
    np.testing.assert_array_equal(arrays["array"], np.eye(2))
