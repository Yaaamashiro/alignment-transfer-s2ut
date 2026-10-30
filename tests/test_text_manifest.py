from copy import deepcopy
from pathlib import Path
import pytest
from alignment_transfer_s2ut.data.manifest import (load_manifest,validate,text_manifest_hash,manifest_hash,speech_manifest_hash,
    TEXT_IDENTITY,checkpoint_manifest_rows,checkpoint_manifest_hash)
from alignment_transfer_s2ut.io import write_jsonl,file_hash,digest
from alignment_transfer_s2ut.text_manifest import prepare


def rows():
    return [dict(pair_id="a",split="train",ja_text="こんにちは。",en_text="Hello."),
        dict(pair_id="b",split="dev",ja_text="また明日。",en_text="See you.")]


def test_text_validation_and_loading_never_resolves_or_probes_audio(tmp_path,monkeypatch):
    values=rows();values[0].update(ja_audio="never-read-ja.wav",en_audio="never-read-en.wav")
    path=tmp_path/"text.jsonl";write_jsonl(path,values)
    resolve=Path.resolve
    def guarded(p,*a,**kw):
        assert p.suffix not in {".wav",".flac"},"audio path resolution is forbidden"
        return resolve(p,*a,**kw)
    def no_audio(*a,**kw):raise AssertionError("audio filesystem probe")
    monkeypatch.setattr(Path,"resolve",guarded)
    monkeypatch.setattr("alignment_transfer_s2ut.data.manifest.local_path",no_audio)
    assert validate(values,require_audio=False)==values
    assert load_manifest(path,require_audio=False)==values
    assert load_manifest(path,split="train",require_audio=False)==values[:1]


def test_speech_schema_remains_required_even_when_payload_checks_are_disabled(tmp_path):
    path=tmp_path/"text.jsonl";write_jsonl(path,rows())
    for kwargs in ({},{"check_audio":False}):
        with pytest.raises(ValueError,match="audio-required"):
            load_manifest(path,**kwargs)


@pytest.mark.parametrize("field",["pair_id","split","ja_text","en_text"])
def test_text_hash_mutations_change_identity(field):
    original=rows();changed=deepcopy(original)
    changed[0][field]="dev" if field=="split" else "changed"
    assert text_manifest_hash(original)!=text_manifest_hash(changed)


def test_text_identity_ignores_audio_and_speech_hash_keeps_legacy_formula():
    original=rows();attached=[dict(row,ja_audio="a.wav",en_audio="b.wav",ja_sha256="ja",en_sha256="en",metadata="ignored") for row in original]
    assert text_manifest_hash(original)==text_manifest_hash(attached)
    keys=("pair_id","split","ja_text","en_text","ja_audio","en_audio","ja_sha256","en_sha256")
    assert speech_manifest_hash(attached)==manifest_hash(attached)==digest([{k:r[k] for k in keys} for r in attached])
    changed=deepcopy(attached);changed[0]["ja_audio"]="changed.wav"
    assert speech_manifest_hash(attached)!=speech_manifest_hash(changed)
    assert text_manifest_hash(attached)==text_manifest_hash(changed)


def test_leakage_guards_are_retained_in_text_mode():
    for values,pattern in (([rows()[0],rows()[0]],"pair_id"),([rows()[0],dict(rows()[0],pair_id="z",split="dev")],"leakage")):
        with pytest.raises(ValueError,match=pattern):validate(values,require_audio=False)


def test_canonical_staging_is_idempotent_after_audio_is_attached(tmp_path):
    source,output=tmp_path/"external.jsonl",tmp_path/"local/text.jsonl"
    values=rows();write_jsonl(source,values)
    first=prepare(source,output);before=file_hash(output)
    write_jsonl(source,[dict(r,ja_audio="missing-ja.wav",en_audio="missing-en.wav") for r in values],overwrite=True)
    assert prepare(source,output,resume=True)==first
    assert file_hash(output)==before
    assert load_manifest(output,require_audio=False)==values
    with pytest.raises(ValueError,match="read-only"):prepare(source,source)
    values[0]["en_text"]="Changed.";write_jsonl(source,values,overwrite=True)
    with pytest.raises(ValueError,match="changed"):prepare(source,output,resume=True)


def test_legacy_checkpoint_requires_provable_original_metadata(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="legacy T2TT checkpoint"):
        checkpoint_manifest_rows({},rows(),tmp_path/"manifest.jsonl")
    canonical=[dict(r,ja_audio=str(tmp_path/"ja.wav"),en_audio=str(tmp_path/"en.wav")) for r in rows()]
    relative=[dict(r,ja_audio="ja.wav",en_audio="en.wav") for r in rows()]
    assert checkpoint_manifest_rows({},relative,tmp_path/"manifest.jsonl")==canonical
    assert checkpoint_manifest_hash({},canonical)==manifest_hash(canonical)
    assert checkpoint_manifest_rows(dict(manifest_identity=TEXT_IDENTITY),rows(),tmp_path/"manifest.jsonl")==rows()
