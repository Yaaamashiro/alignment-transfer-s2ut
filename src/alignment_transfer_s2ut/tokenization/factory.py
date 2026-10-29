"""Checkpoint-owned tokenizers; absent metadata means legacy character mode."""
import json
from pathlib import Path
from .character import CharacterTokenizer, UNK
from ..io import atomic_write, digest, write_json
from ..progress import progress


def tokenizers_from_checkpoint(saved):
    if "tokenizers" not in saved:
        return tuple(CharacterTokenizer(saved[name + "_vocabulary"]) for name in ("source", "target"))
    from .sentencepiece import SentencePieceTokenizer
    result = tuple(SentencePieceTokenizer.from_state(saved["tokenizers"][name]) for name in ("source", "target"))
    for name, tokenizer in zip(("source", "target"), result):
        if tokenizer.vocabulary != saved[name + "_vocabulary"]:
            raise ValueError("checkpoint SentencePiece vocabulary differs: " + name)
    return result


def fit_tokenizers(rows, args, modality):
    kind = getattr(args, "tokenizer", "character") if modality == "text" else "character"
    last = Path(args.output) / "checkpoint_last.pt"
    existing = last if last.is_file() else Path(args.output) / "checkpoint_best.pt"
    if modality == "text" and existing.is_file():
        from ..training import load_checkpoint
        old_kind = "sentencepiece" if "tokenizers" in load_checkpoint(existing) else "character"
        if old_kind != kind:
            raise ValueError("tokenizer change must start at step 0 in a new stage; existing checkpoint is read-only")
    if kind == "character":
        return tuple(CharacterTokenizer.fit(rows, field) for field in ("ja_text", "en_text"))
    from .sentencepiece import SentencePieceTokenizer
    if kind != "sentencepiece" or getattr(args, "sentencepiece_model_type", "unigram") != "unigram":
        raise ValueError("only SentencePiece Unigram is supported")
    if args.resume and last.is_file():
        from ..training import load_checkpoint
        saved = load_checkpoint(last)
        if "tokenizers" not in saved:
            raise ValueError("SentencePiece must start at step 0 in a new stage; cannot resume a character checkpoint")
        result = tokenizers_from_checkpoint(saved)
        if [t.spec["requested_vocab_size"] for t in result] != [args.source_vocab_size, args.target_vocab_size]:
            raise ValueError("SentencePiece resume vocabulary request changed")
        return result
    result = []
    for field, size in (("ja_text", args.source_vocab_size), ("en_text", args.target_vocab_size)):
        progress("SentencePiece: fit train only", detail=f"field={field} requested_vocab_size={size} fixed input order, one thread")
        result.append(SentencePieceTokenizer.fit(rows, field, size))
    return tuple(result)


def tokenizer_metadata(source, target):
    if isinstance(source, CharacterTokenizer):
        return {}
    return {"tokenizers": dict(source=source.state(), target=target.state())}


def save_diagnostics(root, rows, source, target):
    if isinstance(source, CharacterTokenizer):
        return
    root = Path(root)
    for name, tokenizer in (("ja", source), ("en", target)):
        vocabulary = "".join(json.dumps(piece, ensure_ascii=False) + "\t" + str(tokenizer.processor.get_score(i)) + "\n"
                             for i, piece in enumerate(tokenizer.vocabulary)).encode("utf-8")
        for suffix, data in (("model", tokenizer.model), ("vocab", vocabulary)):
            path = root / "tokenizers" / (name + "." + suffix)
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("tokenizer asset differs from checkpoint: " + str(path))
            else:
                atomic_write(path, lambda stream, data=data: stream.write(data))
    path = root / "tokenization.json"
    if path.is_file():
        from ..io import read_json
        saved = read_json(path)
        if saved.get("tokenizer_hash") != digest([source.fingerprint, target.fingerprint]):
            raise ValueError("tokenization diagnostics differ from checkpoint")
        return
    stats = {}
    for split in ("train", "dev"):
        selected = [row for row in rows if row["split"] == split]
        for language, tokenizer in (("ja", source), ("en", target)):
            codepoints = tokens = unknown = exact = 0
            for index, row in enumerate(selected):
                text = row[language + "_text"]
                spans = tokenizer.encode_with_spans(text)
                ids = [s[0] for s in spans]
                codepoints += len(text); tokens += len(ids); unknown += sum(e-b for token,b,e in spans if token == UNK)
                restored = tokenizer.decode(ids)
                if UNK not in ids and restored != text:
                    raise ValueError("SentencePiece changed original text despite no unknown tokens")
                exact += restored == text
                if index % 1000 == 0:
                    progress("SentencePiece: tokenization audit " + split + "/" + language, index + 1, len(selected))
            stats[split + "_" + language] = dict(samples=len(selected), mean_codepoints=codepoints/max(1,len(selected)),
                mean_tokens=tokens/max(1,len(selected)), codepoints_per_token=codepoints/max(1,tokens),
                unknown_codepoints=unknown, roundtrip_exact_samples=exact)
    report = dict(tokenizer_type="sentencepiece", source=source.spec, target=target.spec,
        tokenizer_hash=digest([source.fingerprint, target.fingerprint]), statistics=stats,
        vocab_export="JSON-quoted pieces plus tab-separated scores", normalization_scope="original codepoints preserved; held-out UNK is reported")
    write_json(path, report)
    progress("SentencePiece: tokenization diagnostics", detail=json.dumps(report, ensure_ascii=False))
