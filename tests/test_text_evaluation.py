"""Held-out text generation, BLEU and resumable provenance without audio assets."""
import pytest
import torch
import yaml
from alignment_transfer_s2ut.cli import parser_for, run
from alignment_transfer_s2ut.evaluation import text
from alignment_transfer_s2ut.io import read_json, read_jsonl, write_json, write_jsonl
from alignment_transfer_s2ut.smoke_pipeline import TINY_CONFIG, invoke
from alignment_transfer_s2ut.tokenization.character import CharacterTokenizer, EOS, PAD, BOS
from alignment_transfer_s2ut.training import load_checkpoint


class Translator:
    def __init__(self, translations, vocabulary_size):
        self.translations, self.size = translations, vocabulary_size

    def eval(self):
        return self

    def encode(self, source, lengths):
        assert source.size(0) == 1 and lengths.tolist() == [source.size(1)]
        return self.translations[tuple(source[0].tolist())] + [EOS]

    def decode(self, previous, encoded, prior=None, beta=0.):
        prefix = previous[0].tolist()
        assert prefix[0] == EOS and prefix[1:] == encoded[:len(prefix) - 1]
        assert prior is None and beta == 0.
        logits = torch.full((1, len(prefix), self.size), -torch.inf)
        # Special symbols must never become generated text even with large logits.
        logits[0, -1, [BOS, PAD]] = 10.
        logits[0, -1, encoded[len(prefix) - 1]] = 9.
        return logits, None, None


@pytest.fixture
def text_data(tmp_path):
    torch.set_num_threads(1)
    rows = [dict(pair_id=str(i), split=split, ja_text=ja, en_text=en,
                 ja_audio=f"missing/{i}-ja.wav", en_audio=f"missing/{i}-en.wav")
            for i, (split, ja, en) in enumerate([
                ("train", "あいうえ", "a small bird can fly. a little cat can sit."),
                ("dev", "あい", "a small bird can fly."),
                ("dev", "うえ", "a little cat can sit."),
                ("test", "えあ", "excluded test reference")])]
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY_CONFIG), encoding="utf-8")
    invoke("train_t2tt", "--manifest", manifest, "--config", config, "--backend", "smoke",
           "--max-updates", 1, "--batch-size", 1, "--update-frequency", 1, "--output", tmp_path / "t2tt")
    return tmp_path


def _args(root, *extra):
    return parser_for("evaluate_t2tt").parse_args(list(map(str, [
        "--manifest", root / "manifest.jsonl", "--checkpoint", root / "t2tt/checkpoint_best.pt",
        "--output", root / "dev-text.json", "--beam-size", 1, "--max-chars", 64, *extra])))


def test_source_only_generation_eos_and_unfinished_cap():
    source, lengths = torch.tensor([[5, 6]]), torch.tensor([2])
    translator = Translator({(5, 6): [7, 8, 9]}, 10)
    tokens, completed = text.generate_text(translator, source, lengths, beam_size=2, max_chars=8)
    assert tokens == [7, 8, 9, EOS] and completed
    tokens, completed = text.generate_text(translator, source, lengths, max_chars=2)
    assert tokens == [7, 8] and not completed
    empty = Translator({(5, 6): []}, 10)
    assert text.generate_text(empty, source, lengths) == ([EOS], True)


def test_dev_corpus_bleu_has_no_audio_dependency_and_resumes(text_data, monkeypatch):
    root = text_data
    saved = load_checkpoint(root / "t2tt/checkpoint_best.pt")
    source, target = (CharacterTokenizer(saved[key + "_vocabulary"]) for key in ("source", "target"))
    translations = {tuple(source.encode(ja)): target.encode(en) for ja, en in [
        ("あい", "a small bird can fly."), ("うえ", "a little cat can sit.")]}
    monkeypatch.setattr(text, "checkpoint_model", lambda *_: Translator(translations, len(target.vocabulary)))
    result = run("evaluate_t2tt", _args(root))
    assert result["split"] == "dev" and result["samples"] == 2 and result["diagnostic_only"]
    assert result["text_bleu"] == pytest.approx(100.) and result["eos_reached_ratio"] == 1
    assert "tok:13a" in result["bleu_signature"] and "case:mixed" in result["bleu_signature"]
    predictions = read_jsonl(root / "dev-text-predictions.jsonl")
    assert [r["pair_id"] for r in predictions] == ["1", "2"]
    assert all("reference" not in r and "en_text" not in r for r in predictions)
    monkeypatch.setattr(text, "checkpoint_model", lambda *_: pytest.fail("resume decoded again"))
    assert run("evaluate_t2tt", _args(root, "--resume")) == result
    # Resume also reuses per-sample results after interruption before metric publication.
    (root / "dev-text.json").unlink()
    assert run("evaluate_t2tt", _args(root, "--resume")) == result
    with pytest.raises(ValueError, match="resume contract"):
        run("evaluate_t2tt", _args(root, "--resume", "--beam-size", 2))
    with (root / "dev-text-predictions.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="predictions changed"):
        run("evaluate_t2tt", _args(root, "--resume"))


def test_text_evaluation_smoke_and_contract_guards(text_data):
    root = text_data
    result = run("evaluate_t2tt", _args(root, "--max-chars", 8, "--beam-size", 2))
    assert 0 <= result["text_bleu"] <= 100 and result["samples"] == 2
    with pytest.raises(FileExistsError):
        run("evaluate_t2tt", _args(root, "--max-chars", 8, "--beam-size", 2))
    with pytest.raises(ValueError, match="decoding"):
        run("evaluate_t2tt", _args(root, "--max-chars", 0))
    rows = read_jsonl(root / "manifest.jsonl")
    rows[1]["en_text"] = "changed dev reference"
    write_jsonl(root / "manifest.jsonl", rows, overwrite=True)
    with pytest.raises(ValueError, match="checkpoint selection dev"):
        run("evaluate_t2tt", _args(root, "--resume"))


def test_partial_text_cache_rejects_tampering(text_data, monkeypatch):
    root = text_data
    run("evaluate_t2tt", _args(root, "--max-chars", 4))
    (root / "dev-text.json").unlink()
    cache = next((root / "dev-text-cache").glob("*.json"))
    cached = read_json(cache)
    cached["hypothesis"] = "a forged good translation"
    write_json(cache, cached, overwrite=True)
    monkeypatch.setattr(text, "checkpoint_model", lambda *_: pytest.fail("must validate cached data first"))
    with pytest.raises(ValueError, match="cache contract or payload"):
        run("evaluate_t2tt", _args(root, "--max-chars", 4, "--resume"))
