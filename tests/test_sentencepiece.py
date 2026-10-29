from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from alignment_transfer_s2ut.tokenization.character import CharacterTokenizer, BOS,PAD,EOS,UNK
from alignment_transfer_s2ut.tokenization.sentencepiece import SentencePieceTokenizer,codepoint_spans
from alignment_transfer_s2ut.tokenization.factory import tokenizers_from_checkpoint
from alignment_transfer_s2ut.tokenization.alignment import aggregate_ctc_spans,attention_token_spans


TEXTS=["私は猫が好き playing play playing ing", "　ＡＢＣ e\u0301 日本語 😀  A  B 　", "\tX\nY\rZ", "   ","<bos> <pad> <eos> <unk> <space>"]


@pytest.fixture
def spm():
    return SentencePieceTokenizer.fit([dict(split="train",text=t) for t in TEXTS]*10,"text",200)


def test_roundtrip_identity_and_original_spans(spm):
    import gc
    for text in TEXTS+["", "  私は猫   "]:
        assert spm.decode(spm.encode(text))==text
        for _ in range(3):
            gc.collect()
            spans=spm.encode_with_spans(text)
            assert [x[0] for x in spans]==spm.encode(text)
            assert "".join(text[b:e] for _,b,e in spans)==text
    assert [spm.processor.bos_id(),spm.processor.pad_id(),spm.processor.eos_id(),spm.processor.unk_id()]==[BOS,PAD,EOS,UNK]
    assert any(e-b>1 for _,b,e in spm.encode_with_spans(TEXTS[0]))
    assert "<space>" not in spm.vocabulary
    with pytest.raises(ValueError,match="U\\+2581"):
        spm.encode("a▁b")


def test_training_is_deterministic_and_excludes_held_out_text(spm):
    rows=[dict(split="train",text=t) for t in TEXTS]*10
    other=SentencePieceTokenizer.fit(rows+[dict(split="dev",text="Ω unseen"),dict(split="test",text="Ž unseen")],"text",200)
    assert other.model==spm.model
    assert UNK in other.encode("ΩŽ")
    assert other.encode_with_spans("ΩŽ")==[(UNK,0,2)]
    restored=SentencePieceTokenizer.from_state(spm.state())
    assert restored.fingerprint==spm.fingerprint and restored.vocabulary==spm.vocabulary
    invalid=deepcopy(spm.state());invalid["spec"]["model_hash"]="changed"
    with pytest.raises(ValueError,match="metadata/model hash"):
        SentencePieceTokenizer.from_state(invalid)


def test_offsets_explicitly_convert_utf8_bytes_and_reject_nonboundaries():
    values=[SimpleNamespace(id=4,begin=0,end=3,surface="あ"),SimpleNamespace(id=5,begin=3,end=7,surface="😀"),SimpleNamespace(id=6,begin=7,end=8,surface="a")]
    assert codepoint_spans("あ😀a",values)==[(4,0,1),(5,1,2),(6,2,3)]
    values[0].end=2
    with pytest.raises(ValueError,match="offsets/surfaces"):
        codepoint_spans("あ😀a",values)


def test_checkpoint_factory_preserves_legacy_and_validates_vocabulary(spm):
    char=CharacterTokenizer.fit([dict(split="train",text="私 は猫")],"text")
    saved=dict(source_vocabulary=char.vocabulary,target_vocabulary=char.vocabulary)
    assert tokenizers_from_checkpoint(saved)[0].fingerprint==char.fingerprint
    assert char.encode_with_spans("私 は猫")==[(token,i,i+1) for i,token in enumerate(char.encode("私 は猫"))]
    saved.update(source_vocabulary=spm.vocabulary,target_vocabulary=spm.vocabulary,tokenizers=dict(source=spm.state(),target=spm.state()))
    assert tokenizers_from_checkpoint(saved)[0].fingerprint==spm.fingerprint
    saved["source_vocabulary"]=[*spm.vocabulary,"changed"]
    with pytest.raises(ValueError,match="vocabulary differs"):tokenizers_from_checkpoint(saved)


def test_subword_ctc_aggregation_ignores_invalid_characters():
    chars=np.array([[.1,.14],[.14,.18],[.18,.22],[.22,.27],[.27,.31],[.31,.35],[.35,.4]])
    np.testing.assert_allclose(aggregate_ctc_spans(chars,np.array([[0,4],[4,7]])),[[.1,.27],[.27,.4]])
    chars[1:4]=-1
    np.testing.assert_allclose(aggregate_ctc_spans(chars,np.array([[0,4],[4,7]])),[[.1,.14],[.27,.4]])
    chars[:4]=-1
    np.testing.assert_allclose(aggregate_ctc_spans(chars,np.array([[0,4],[4,7]])),[[-1,-1],[.27,.4]])


def test_legacy_attention_requires_codepoint_shape():
    src,tgt,meta=attention_token_spans({},dict(attention=np.eye(2)),"私は","ab")
    assert not meta and src.tolist()==tgt.tolist()==[[0,1],[1,2]]
    with pytest.raises(ValueError,match="dimensions differ"):
        attention_token_spans({},dict(attention=np.ones((1,2))),"私は","ab")


def test_nonidentity_model_is_rejected_even_with_forged_metadata(spm):
    from sentencepiece import sentencepiece_model_pb2 as pb
    proto=pb.ModelProto();proto.ParseFromString(spm.model);proto.normalizer_spec.name="nfkc"
    with pytest.raises(ValueError,match="identity/deterministic"):
        SentencePieceTokenizer(proto.SerializeToString(),200)
