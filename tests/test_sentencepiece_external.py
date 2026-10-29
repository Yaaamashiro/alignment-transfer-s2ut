from copy import deepcopy
from alignment_transfer_s2ut import evaluate_t2tt_external as external
from alignment_transfer_s2ut.io import read_jsonl
from alignment_transfer_s2ut.tokenization.character import EOS,UNK
from test_sentencepiece import spm
from test_t2tt_external import evidence,evaluation


def test_fixed_300_evaluation_uses_embedded_spm_and_codepoint_oov(evidence,evaluation,spm,monkeypatch):
    args,_=evaluation
    saved=external.load_checkpoint(args.checkpoint)
    saved.update(source_vocabulary=spm.vocabulary,target_vocabulary=spm.vocabulary,
        tokenizers=dict(source=spm.state(),target=spm.state()))
    monkeypatch.setattr(external,"load_checkpoint",lambda _:saved)
    calls=[]
    def generate(model,source,lengths,**kwargs):
        assert kwargs==dict(beam_size=5,max_chars=1000,length_penalty=1.)
        calls.append(source.tolist())
        return spm.encode("playing play")+[EOS],True
    monkeypatch.setattr(external.text,"generate_text",generate)
    result=external.evaluate(args)
    assert result["samples"]==len(calls)==300 and result["source_unk_characters"]>0
    records=read_jsonl(result["predictions"])
    for row in records:
        unknown=sum(end-begin for token,begin,end in spm.encode_with_spans(row["source_text"]) if token==UNK)
        assert row["source_unk_characters"]==unknown
    assert sum(b["samples"] for b in result["source_length_bleu"])==300
    assert result["generation_diagnostics"]["unique_hypotheses"]==1
    assert external.evaluate(args)==result
