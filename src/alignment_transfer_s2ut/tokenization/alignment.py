"""Subword text spans to unchanged character-CTC timing, without re-alignment."""
import numpy as np
from ..io import digest

FIELDS = ("tokenizer_type", "source_tokenizer_hash", "target_tokenizer_hash", "source_tokenizer_spec", "target_tokenizer_spec", "token_span_unit")


def valid_spans(spans, text):
    spans = np.asarray(spans)
    if (spans.ndim != 2 or spans.shape[1] != 2 or not np.issubdtype(spans.dtype, np.integer) or
            len(spans) == 0 or spans[0,0] != 0 or spans[-1,1] != len(text) or
            np.any(spans[:,0] >= spans[:,1]) or np.any(spans[1:,0] != spans[:-1,1])):
        raise ValueError("token spans must partition the original Unicode codepoints")
    return spans


def attention_token_spans(meta, packet, ja, en, key="attention"):
    kind = meta.get("tokenizer_type", "character")
    extra = {}
    if kind == "sentencepiece":
        if any(name not in meta for name in FIELDS) or meta["token_span_unit"] != "unicode-codepoint":
            raise ValueError("SentencePiece attention requires explicit tokenizer identity and codepoint spans")
        for side in ("source", "target"):
            spec = meta[side + "_tokenizer_spec"]
            if (spec.get("type") != "sentencepiece" or spec.get("normalization") != "identity" or
                    digest(spec) != meta[side + "_tokenizer_hash"]):
                raise ValueError("SentencePiece tokenizer metadata/hash mismatch")
        if digest([meta["source_tokenizer_hash"],meta["target_tokenizer_hash"]]) != meta["tokenizer_hash"]:
            raise ValueError("SentencePiece combined tokenizer hash mismatch")
        if any(side + "_token_spans" not in packet for side in ("source", "target")):
            raise ValueError("SentencePiece attention token spans are required; never infer subword spans")
        extra = {name:meta[name] for name in FIELDS}
    elif kind != "character" or any(name in meta for name in FIELDS[1:]):
        raise ValueError("character/SentencePiece tokenizer metadata mixed")
    spans = []
    for side, text in (("source",ja),("target",en)):
        literal = np.stack([np.arange(len(text)),np.arange(1,len(text)+1)],-1)
        value = valid_spans(packet.get(side + "_token_spans",literal),text)
        if kind == "character" and not np.array_equal(value,literal):
            raise ValueError("legacy character attention must match original codepoints")
        spans.append(value)
    if packet[key].shape != (len(spans[1]),len(spans[0])):
        raise ValueError("attention dimensions differ from token/codepoint spans")
    return *spans, extra


def aggregate_ctc_spans(characters, tokens):
    characters, tokens = np.asarray(characters), np.asarray(tokens)
    result = np.full((len(tokens),2),-1.,dtype=np.float64)
    for index,(begin,end) in enumerate(tokens):
        chars = characters[begin:end]
        valid = chars[(chars[:,0] >= 0) & (chars[:,1] > chars[:,0])]
        if len(valid):
            result[index] = valid[0,0],valid[-1,1]
    return result
