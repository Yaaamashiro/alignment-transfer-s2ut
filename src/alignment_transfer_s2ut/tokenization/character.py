from abc import ABC, abstractmethod
from ..io import digest

# Match fairseq Dictionary special indices; literal special-looking input stays characters.
SPECIAL = ["<bos>", "<pad>", "<eos>", "<unk>", "<space>"]
PAD, UNK, BOS, EOS = 1, 3, 0, 2


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text): ...

    @abstractmethod
    def encode(self, text): ...


class CharacterTokenizer(Tokenizer):
    def __init__(self, vocabulary):
        self.vocabulary = list(vocabulary)
        if self.vocabulary[:5] != SPECIAL or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("invalid character vocabulary")
        self.indices = {token: i for i, token in enumerate(vocabulary)}

    @staticmethod
    def tokenize(text):
        # Only U+0020 is represented by <space>; tabs, newlines and all other codepoints survive.
        return ["<space>" if ch == " " else ch for ch in text]

    @classmethod
    def fit(cls, rows, field):
        train = [r for r in rows if r["split"] == "train"]
        if not train:
            raise ValueError("tokenizer fitting requires train data")
        tokens = {c for r in train for c in cls.tokenize(r[field])}
        return cls(SPECIAL + sorted(tokens - set(SPECIAL)))

    def encode(self, text):
        return [self.indices.get(c, UNK) for c in self.tokenize(text)]

    def decode(self, indices):
        return "".join(" " if self.vocabulary[i] == "<space>" else self.vocabulary[i]
                       for i in indices if i not in {PAD, BOS, EOS})

    @property
    def fingerprint(self):
        return digest({"type": "unicode-codepoint-v1", "vocabulary": self.vocabulary})
