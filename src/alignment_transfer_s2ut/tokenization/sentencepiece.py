"""Pinned, lossless-normalization Unigram models and original-codepoint offsets."""
import base64
import hashlib
import io
from .character import Tokenizer, BOS, PAD, EOS, UNK
from ..io import digest

CONTROLS = ["\t", "\n", "\r"]


def codepoint_spans(text, pieces, unit="auto"):
    """Validate surfaces; explicitly translate UTF-8 byte offsets when necessary."""
    boundaries, count = {0: 0}, 0
    for index, char in enumerate(text):
        count += len(char.encode("utf-8"))
        boundaries[count] = index + 1
    for candidate in ("codepoint", "byte") if unit == "auto" else (unit,):
        spans, end = [], 0
        try:
            for piece in pieces:
                begin, finish = piece.begin, piece.end
                if candidate == "byte":
                    begin, finish = boundaries[begin], boundaries[finish]
                if not (begin == end and begin < finish <= len(text)) or text[begin:finish] != piece.surface:
                    raise ValueError("token surfaces do not preserve original codepoints")
                spans.append((piece.id, begin, finish)); end = finish
            if end != len(text):
                raise ValueError("token spans do not cover original text")
            return spans
        except (ValueError, KeyError):
            continue
    raise ValueError("SentencePiece offsets/surfaces cannot be mapped to original codepoints")


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model, requested_vocab_size, expected=None):
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2 as pb
        if spm.__version__ != "0.2.0":
            raise ValueError("SentencePiece 0.2.0 is required for reproducible tokenization")
        proto = pb.ModelProto(); proto.ParseFromString(model)
        normal, trainer = proto.normalizer_spec, proto.trainer_spec
        if (trainer.model_type != pb.TrainerSpec.UNIGRAM or normal.name != "identity" or normal.precompiled_charsmap or
                normal.add_dummy_prefix or normal.remove_extra_whitespaces or not normal.escape_whitespaces or
                trainer.byte_fallback or trainer.character_coverage != 1. or trainer.shuffle_input_sentence or
                trainer.input_sentence_size != 0 or trainer.num_threads != 1 or list(trainer.user_defined_symbols) != CONTROLS):
            raise ValueError("SentencePiece model violates the identity/deterministic Unigram contract")
        self.model = bytes(model)
        self.processor = spm.SentencePieceProcessor(model_proto=self.model)
        if [self.processor.bos_id(), self.processor.pad_id(), self.processor.eos_id(), self.processor.unk_id()] != [BOS, PAD, EOS, UNK]:
            raise ValueError("SentencePiece special token IDs differ")
        self.vocabulary = [self.processor.id_to_piece(i) for i in range(self.processor.get_piece_size())]
        if self.vocabulary[:4] != ["<bos>","<pad>","<eos>","<unk>"] or type(requested_vocab_size) is not int or requested_vocab_size < len(self.vocabulary):
            raise ValueError("SentencePiece vocabulary/special pieces differ")
        self.spec = dict(type="sentencepiece", model_type="unigram", normalization="identity", requested_vocab_size=requested_vocab_size,
            actual_vocab_size=len(self.vocabulary), model_hash=hashlib.sha256(self.model).hexdigest(), sentencepiece_version=spm.__version__,
            add_dummy_prefix=False, remove_extra_whitespaces=False, character_coverage=1., byte_fallback=False,
            user_defined_symbols=CONTROLS, shuffle_input_sentence=False, num_threads=1, input_sentence_size=0)
        if expected is not None and self.spec != expected:
            raise ValueError("SentencePiece checkpoint metadata/model hash differs")

    @staticmethod
    def check_text(text):
        if "▁" in text or "\x00" in text:
            raise ValueError("literal U+2581/NUL is not losslessly representable by this SentencePiece policy; text was not modified")

    @classmethod
    def fit(cls, rows, field, vocab_size=8000):
        import sentencepiece as spm
        texts = [row[field] for row in rows if row["split"] == "train"]
        if not texts or type(vocab_size) is not int or vocab_size < 8:
            raise ValueError("SentencePiece requires train data and a positive vocabulary budget >=8")
        for text in texts:
            cls.check_text(text)
            if len(text.encode("utf-8")) > 1048576:
                raise ValueError("SentencePiece sentence exceeds the explicit 1MiB limit; no sentence was skipped")
        alphabet = sorted({ch for text in texts for ch in text if ch not in " \t\r\n"})
        minimum = len(alphabet) + 7 + int(any(" " in text for text in texts))
        if vocab_size < minimum:
            raise ValueError(f"requested vocabulary {vocab_size} cannot cover {minimum} required symbols at coverage=1.0; corpus was not filtered")
        output = io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=iter(texts), model_writer=output,
            model_type="unigram", vocab_size=vocab_size, character_coverage=1., normalization_rule_name="identity",
            bos_id=BOS, pad_id=PAD, eos_id=EOS, unk_id=UNK, bos_piece="<bos>", pad_piece="<pad>", eos_piece="<eos>", unk_piece="<unk>",
            byte_fallback=False, hard_vocab_limit=False, input_sentence_size=0, shuffle_input_sentence=False, num_threads=1,
            add_dummy_prefix=False, remove_extra_whitespaces=False, split_by_whitespace=False, allow_whitespace_only_pieces=True,
            user_defined_symbols=CONTROLS, required_chars="".join(alphabet),
            max_sentence_length=1048576, minloglevel=1)
        return cls(output.getvalue(), vocab_size)

    def tokenize(self, text):
        self.check_text(text)
        return self.processor.encode(text, out_type=str)

    def encode(self, text):
        self.check_text(text)
        return self.processor.encode(text, out_type=int)

    def encode_with_spans(self, text):
        self.check_text(text)
        proto = self.processor.encode_as_immutable_proto(text)  # Keep the C++ owner alive while reading piece views.
        return codepoint_spans(text, list(proto.pieces))

    def decode(self, indices):
        return self.processor.decode([int(i) for i in indices if i not in {BOS, PAD, EOS}])

    @property
    def fingerprint(self):
        return digest(self.spec)

    def state(self):
        return dict(spec=self.spec, model_proto_base64=base64.b64encode(self.model).decode("ascii"))

    @classmethod
    def from_state(cls, state):
        return cls(base64.b64decode(state["model_proto_base64"], validate=True), state["spec"]["requested_vocab_size"], state["spec"])
