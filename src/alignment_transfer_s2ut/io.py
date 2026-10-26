"""Atomic files, content-addressed pair names, and verified provenance."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def local_path(path):
    if os.environ.get("S2ST_INPUT_MAP"):
        from direct_s2st.drive_staging import local_path as resolve
        target = resolve(path)
        if (str(path).replace("\\", "/").startswith("/content/drive/") and target == Path(path)
                and Path(path).suffix.lower() in (".wav", ".flac", ".npz", ".units")):
            raise RuntimeError("Drive input was not staged locally: " + str(path))
        return target
    return Path(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(local_path(path), "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, writer, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, data, overwrite=False):
    atomic_write(path, lambda f: f.write((json.dumps(data, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + "\n").encode()), overwrite)


def read_json(path):
    return json.loads(local_path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with open(local_path(path), encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows, overwrite=False):
    atomic_write(path, lambda f: f.write("".join(json.dumps(row, ensure_ascii=False,
                allow_nan=False) + "\n" for row in rows).encode()), overwrite)


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, pair_id):
        return self.root / (digest(pair_id) + ".npz")

    def hash(self, pair_id):
        return file_hash(self.path(pair_id))

    def put(self, pair_id, metadata, arrays, overwrite=False):
        metadata = {**metadata, "pair_id": pair_id, "schema_version": 1}
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        for value in arrays.values():
            if not np.isfinite(value).all():
                raise ValueError("non-finite artifact")
        metadata["payload_hash"] = self.payload_hash(metadata, arrays)
        atomic_write(self.path(pair_id), lambda f: np.savez_compressed(f,
                     metadata=np.array(json.dumps(metadata, ensure_ascii=False, allow_nan=False)),
                     **arrays), overwrite)
        write_json(self.path(pair_id).with_suffix(".sha256.json"),
                   {"sha256": file_hash(self.path(pair_id))}, overwrite)

    def get(self, pair_id, expected=None):
        path = self.path(pair_id)
        checksum_path = path.with_suffix(".sha256.json")
        if checksum_path.exists() and file_hash(path) != read_json(checksum_path)["sha256"]:
            raise ValueError("artifact content hash mismatch: " + pair_id)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            arrays = {k: data[k].copy() for k in data.files if k != "metadata"}
        if meta.get("payload_hash") != self.payload_hash(meta, arrays):
            raise ValueError("artifact payload hash mismatch: " + pair_id)
        if meta["pair_id"] != pair_id:
            raise ValueError("pair_id mismatch")
        for key, value in (expected or {}).items():
            if meta.get(key) != value:
                raise ValueError("artifact provenance mismatch: " + key)
        if any(not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError("non-finite artifact")
        return meta, arrays

    def verify_for_resume(self, pair_id, expected=None):
        """Verify an owned stage output and finish an interrupted receipt write.

        Keep get() read-only for input artifacts. Only producers resuming their
        own output directory may publish this missing external checksum.
        """
        path = self.path(pair_id)
        receipt = path.with_suffix(".sha256.json")
        missing = not receipt.exists()
        before = path.stat() if missing else None
        result = self.get(pair_id, expected)
        if missing:
            checksum = file_hash(path)
            after = path.stat()
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, key) != getattr(after, key) for key in fields):
                raise ValueError("artifact changed during resume verification: " + pair_id)
            write_json(receipt, {"sha256": checksum})
            from .progress import progress
            progress("artifact resume: restored checksum", detail=pair_id)
        return result

    @staticmethod
    def payload_hash(metadata, arrays):
        metadata = {k: v for k, v in metadata.items() if k != "payload_hash"}
        signatures = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                           "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                      for key, value in arrays.items()}
        return digest({"metadata": metadata, "arrays": signatures})
