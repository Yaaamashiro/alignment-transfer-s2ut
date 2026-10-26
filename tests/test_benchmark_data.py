"""Literal fairseq TSV compatibility, with transcript identity kept strict."""
import csv
import io

import numpy as np
import pytest
import soundfile as sf

from alignment_transfer_s2ut.benchmark import tsv
from alignment_transfer_s2ut.benchmark_data import export_manifest
from alignment_transfer_s2ut.io import file_hash, write_json, write_jsonl


@pytest.mark.parametrize("text", ['"quoted"', '"unmatched', 'a""b', '「引用」 "ok"'])
def test_literal_quotes_survive_manifest_export_and_resume(tmp_path, text):
    common, prepared = tmp_path / "common", tmp_path / "prepared"
    common.mkdir()
    prepared.mkdir()
    write_json(common / "dataset-lock.json", {"source": "fixed corpus"})
    write_json(prepared / "data-lock.json", {
        "common_dataset_lock_sha256": file_hash(common / "dataset-lock.json")})
    for task in ("source_letter", "target_letter"):
        (prepared / task).mkdir()
    source = []
    for index, split in enumerate(("train", "dev")):
        row = dict(pair_id=split, split=split, ja_text=split + "あ", en_text=text + " " + split)
        for language in ("ja", "en"):
            path = tmp_path / f"{split}-{language}.wav"
            sf.write(path, np.arange(800 + index * 160, dtype=float) / 2000, 16000)
            row[language + "_audio"] = str(path)
        source.append(row)
        write_jsonl(common / f"{split}.jsonl", [row])
        (prepared / f"{split}.tsv").write_text(
            "id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n"
            f"{split}\t{row['ja_audio']}\t5\t0 1\t2\n", encoding="utf-8")
        for language, task in (("ja", "source_letter"), ("en", "target_letter")):
            # Exactly the benchmark's literal tab-separated output format.
            tokens = ["<space>" if c == " " else c for c in row[language + "_text"]]
            path = prepared / task / f"{split}.tsv"
            path.write_text(f"id\ttgt_text\n{split}\t{' '.join(tokens)}\n", encoding="utf-8")
            assert tsv(path) == [{"id": split, "tgt_text": " ".join(tokens)}]
    output = tmp_path / "exported/manifest.jsonl"
    result = export_manifest(common, prepared, output)
    assert [r["en_text"] for r in result] == [r["en_text"] for r in source]
    assert export_manifest(common, prepared, output, resume=True) == result
    # Fixing the reader must not allow a genuinely different transcript.
    path = prepared / "target_letter/train.tsv"
    path.write_text(path.read_text(encoding="utf-8").replace("t r a i n", "w r o n g"), encoding="utf-8")
    with pytest.raises(ValueError, match="split=train pair_id=train language=en"):
        export_manifest(common, prepared, output, resume=True)


def test_unmatched_quote_does_not_swallow_following_rows(tmp_path):
    content = 'id\ttgt_text\nfirst\t" h i\nsecond\tn e x t\n'
    # Reproduce why the default CSV reader cannot be used for fairseq labels.
    assert len(list(csv.DictReader(io.StringIO(content), delimiter="\t"))) == 1
    path = tmp_path / "train.tsv"
    path.write_text(content, encoding="utf-8")
    assert tsv(path) == [dict(id="first", tgt_text='" h i'), dict(id="second", tgt_text="n e x t")]


@pytest.mark.parametrize("content", [
    "id\ttgt_text\n", "id\ttgt_text\na\tx\na\ty\n",
    "id\ttgt_text\na\n", "id\ttgt_text\na\tx\textra\n",
    "id\ttgt_text\n\tx\n", "id\tid\na\tx\n", "tgt_text\nx\n",
])
def test_malformed_prepared_tsv_is_rejected(tmp_path, content):
    path = tmp_path / "train.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="prepared"):
        tsv(path)
