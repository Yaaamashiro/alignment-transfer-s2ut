from pathlib import Path, PurePosixPath, PureWindowsPath
import os
from ..io import read_jsonl, write_jsonl, digest, file_hash, local_path

TEXT_REQUIRED = ("pair_id", "split", "ja_text", "en_text")
AUDIO_REQUIRED = ("ja_audio", "en_audio")
REQUIRED = TEXT_REQUIRED + AUDIO_REQUIRED  # Historical speech schema and hash order.
TEXT_IDENTITY = "text-v1"


def kokoro_audio_path(raw, root, language):
    """Rebase the producer's audio/16k path without opening the Drive WAV."""
    if not isinstance(raw, str) or not raw or any(c in raw for c in "\0\t\r\n"):
        raise ValueError(f"missing Kokoro {language} WAV path")
    path = PurePosixPath(raw.replace("\\", "/"))
    parts = path.parts
    matches = [index for index in range(len(parts) - 3)
               if parts[index:index + 3] == ("audio", "16k", language)]
    if len(matches) != 1 or ".." in parts or path.suffix.lower() != ".wav":
        raise ValueError(f"invalid Kokoro {language} WAV path: {raw}")
    # Reject unexpected relative prefixes. Absolute paths may name an older VM.
    start = matches[0]
    if start and not (path.is_absolute() or parts[0].endswith(":") or parts[:start] == ("production",)):
        raise ValueError(f"invalid Kokoro relative WAV path: {raw}")
    relative = Path(*parts[start:])
    return str(Path(root) / relative), relative.as_posix()


def corpus_audio_path(raw, root, language):
    """Resolve a declared WAV path relative to the selected read-only corpus."""
    if not isinstance(raw, str) or not raw or any(c in raw for c in "\0\t\r\n"):
        raise ValueError(f"missing {language} WAV path")
    path = PurePosixPath(raw.replace("\\", "/"))
    if ".." in path.parts or path.suffix.lower() != ".wav":
        raise ValueError(f"invalid {language} WAV path: {raw}")
    root = Path(root).resolve()
    absolute = path.is_absolute() or PureWindowsPath(raw).is_absolute()
    candidate = Path(raw) if absolute else root / str(path)
    candidate = Path(os.path.abspath(candidate))
    if candidate.is_relative_to(root):
        return str(candidate), candidate.relative_to(root).as_posix()
    # Older release manifests may contain the producer's original VM prefix.
    return kokoro_audio_path(raw, root, language)


def speech_audio_path(value, parent=None):
    path = Path(value) if parent is None else parent / value
    logical = os.path.abspath(path)
    if os.environ.get("S2ST_INPUT_MAP") and logical.startswith("/content/drive/"):
        # The benchmark's verified map keys are logical absolute paths.
        # Resolving symlinks would stat every original Drive WAV.
        return Path(logical)
    return path.resolve()


def verified_teacher_subset(saved, rows, source_manifest):
    """Prove the frozen speech text is a split-preserving subset of T2TT input."""
    if saved.get("manifest_identity") != TEXT_IDENTITY:
        raise ValueError("a separate teacher training manifest requires a text-v1 T2TT checkpoint")
    source = load_manifest(source_manifest, require_audio=False)
    for split in ("train", "dev"):
        if saved[f"{split}_hash"] != text_manifest_hash([row for row in source if row["split"] == split]):
            raise ValueError(f"teacher's original {split} manifest differs from the checkpoint")
    index = {row["pair_id"]: row for row in source}
    for row in rows:
        original = index.get(row["pair_id"])
        if row["split"] not in ("train", "dev") or original is None or any(
                row[key] != original[key] for key in TEXT_REQUIRED):
            raise ValueError("speech pair ID, split or text differs from the teacher training corpus: " + row["pair_id"])
    return text_manifest_hash(source)


def validate(rows, check_audio=True, *, require_audio=True):
    if not rows:
        raise ValueError("empty manifest")
    indexed = None
    if check_audio and require_audio and os.environ.get("S2ST_INPUT_MAP"):
        from ..colab_runtime import stream_audio_mapping
        indexed = stream_audio_mapping()
    ids, texts, audio_pairs, audio_contents = set(), {}, {}, {}
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k] for k in TEXT_REQUIRED):
            raise ValueError("text manifest fields must be nonempty strings: " + ", ".join(TEXT_REQUIRED))
        if require_audio and any(not isinstance(row.get(k), str) or not row[k] for k in AUDIO_REQUIRED):
            raise ValueError("audio-required stage needs nonempty ja_audio and en_audio fields")
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError("split must be train/dev/test")
        if row["pair_id"] in ids:
            raise ValueError("duplicate pair_id or cross-split leakage: " + row["pair_id"])
        ids.add(row["pair_id"])
        pairs = [(texts, (row["ja_text"], row["en_text"]))]
        if require_audio:
            pairs.append((audio_pairs, (row["ja_audio"], row["en_audio"])))
        for cache, key in pairs:
            if key in cache and cache[key] != row["split"]:
                raise ValueError("cross-split duplicate pair leakage")
            cache[key] = row["split"]
        if require_audio and row.get("ja_sha256") and row.get("en_sha256"):
            signature = (row["ja_sha256"], row["en_sha256"])
            if signature in audio_contents and audio_contents[signature] != row["split"]:
                raise ValueError("cross-split audio content pair leakage")
            audio_contents[signature] = row["split"]
        if require_audio and check_audio:
            for key in ("ja_audio", "en_audio"):
                if indexed is not None:
                    entry = indexed.get(os.path.abspath(row[key]))
                    declared = row.get(key.replace("_audio", "_sha256"))
                    if entry is None or (declared and declared != entry["wav_sha256"]):
                        raise ValueError("manifest audio differs from the ZIP index: " + row[key])
                elif not local_path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
    return rows


def load_manifest(path, split=None, check_audio=True, *, require_audio=True):
    path = Path(path).resolve()
    rows = read_jsonl(path)
    for row in rows:
        for key in ("ja_audio", "en_audio"):
            if require_audio and isinstance(row.get(key), str) and row[key]:
                row[key] = str(speech_audio_path(row[key], path.parent))
    validate(rows, check_audio, require_audio=require_audio)
    return [r for r in rows if split is None or r["split"] == split]


def manifest_hash(rows):
    return digest([{k: r[k] for k in (*REQUIRED, "ja_sha256", "en_sha256") if k in r} for r in rows])


def speech_manifest_hash(rows):
    """Explicit alias retaining the historical S2UT/audio artifact identity."""
    return manifest_hash(rows)


def text_manifest_hash(rows):
    """Only ordered pair_id/split/original text; audio and extra fields are ignored."""
    return digest([{k: row[k] for k in TEXT_REQUIRED} for row in rows])


def checkpoint_manifest_rows(saved, rows, path):
    """Read old identities without opening/stat-ing optional audio payloads.

    Old checkpoints bind canonical audio metadata. A text-only manifest cannot
    prove that old contract; never guess or rewrite the saved hashes.
    """
    identity = saved.get("manifest_identity")
    if identity == TEXT_IDENTITY:
        return rows
    if identity is not None:
        raise ValueError("unsupported checkpoint manifest identity: " + str(identity))
    if any(not isinstance(row.get(k), str) or not row[k] for row in rows for k in AUDIO_REQUIRED):
        raise ValueError("legacy T2TT checkpoint requires its original audio metadata for identity verification; restore the original manifest")
    parent = Path(path).resolve().parent
    return [{**row, **{k: os.path.abspath(parent / row[k]) for k in AUDIO_REQUIRED}} for row in rows]


def checkpoint_manifest_hash(saved, rows):
    return text_manifest_hash(rows) if saved.get("manifest_identity") == TEXT_IDENTITY else manifest_hash(rows)


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
