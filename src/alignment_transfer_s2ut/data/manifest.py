from pathlib import Path
from ..io import read_jsonl, write_jsonl, digest, file_hash

REQUIRED = ("pair_id", "split", "ja_text", "en_text", "ja_audio", "en_audio")


def validate(rows, check_audio=True):
    if not rows:
        raise ValueError("empty manifest")
    ids, texts, audio_pairs, audio_contents = set(), {}, {}, {}
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k] for k in REQUIRED):
            raise ValueError("canonical fields must be nonempty strings")
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError("split must be train/dev/test")
        if row["pair_id"] in ids:
            raise ValueError("duplicate pair_id or cross-split leakage: " + row["pair_id"])
        ids.add(row["pair_id"])
        for cache, key in ((texts, (row["ja_text"], row["en_text"])),
                           (audio_pairs, (row["ja_audio"], row["en_audio"]))):
            if key in cache and cache[key] != row["split"]:
                raise ValueError("cross-split duplicate pair leakage")
            cache[key] = row["split"]
        if row.get("ja_sha256") and row.get("en_sha256"):
            signature = (row["ja_sha256"], row["en_sha256"])
            if signature in audio_contents and audio_contents[signature] != row["split"]:
                raise ValueError("cross-split audio content pair leakage")
            audio_contents[signature] = row["split"]
        if check_audio:
            for key in ("ja_audio", "en_audio"):
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
    return rows


def load_manifest(path, split=None, check_audio=True):
    path = Path(path).resolve()
    rows = read_jsonl(path)
    for row in rows:
        for key in ("ja_audio", "en_audio"):
            if key in row:
                row[key] = str((path.parent / row[key]).resolve())
    validate(rows, check_audio)
    return [r for r in rows if split is None or r["split"] == split]


def manifest_hash(rows):
    return digest([{k: r[k] for k in (*REQUIRED, "ja_sha256", "en_sha256") if k in r} for r in rows])


def prepare_manifest(source, output, overwrite=False, corpus_root=None, resume=False):
    if corpus_root is None:
        rows = load_manifest(source)
    else:
        root = Path(corpus_root).resolve()
        if Path(output).resolve().is_relative_to(root):
            raise ValueError("derived output must be outside corpus_root")
        rows = read_jsonl(source)
        for row in rows:
            for language in ("ja", "en"):
                raw = row.get(language + "_audio_corpus_relative") or row.get(language + "_audio") or row.get(language + "_wav_16k")
                if not raw:
                    raise ValueError("missing audio path for " + row["pair_id"])
                path = root / raw
                if not path.is_file() and "audio_16k/" in str(raw).replace("\\", "/"):
                    # Colab manifests can contain paths from an earlier /content workspace.
                    suffix = str(raw).replace("\\", "/").split("audio_16k/", 1)[1]
                    candidates = [root / "production/audio_16k" / suffix, root / "audio_16k" / suffix]
                    matches = [p for p in candidates if p.is_file()]
                    if len(matches) != 1:
                        raise ValueError("audio path could not be uniquely relocated: " + str(raw))
                    path = matches[0]
                row[language + "_audio"] = str(path.resolve())
        validate(rows)
    from ..progress import progress
    for index, row in enumerate(rows):
        for language in ("ja", "en"):
            checksum = file_hash(row[language + "_audio"])
            key = language + "_sha256"
            if key in row and row[key] != checksum:
                raise ValueError("corpus audio hash mismatch: " + row["pair_id"])
            row[key] = checksum
        progress("manifest: audio hashes", index + 1, len(rows))
    validate(rows)
    if resume and Path(output).is_file():
        if load_manifest(output) != rows:
            raise ValueError("resume manifest differs from the source; use a new experiment directory")
    else:
        write_jsonl(output, rows, overwrite)
    return rows
