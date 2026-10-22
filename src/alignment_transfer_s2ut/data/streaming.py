"""Bounded ordered CPU loading and an owned, verified local feature/artifact cache.

Workers perform deterministic reads only. Sampling commits on consumption,
and augmentation/RNG and CUDA stay in the training thread.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import numpy as np
from filelock import FileLock, Timeout
from ..io import ArtifactStore, digest, file_hash, read_json, write_json


def stamp(path):
    path = Path(path).resolve()
    value = path.stat()
    return [str(path), value.st_size, value.st_mtime_ns]


class LocalCache:
    """Shared sequential-run cache with per-entry leases and bounded disk use.

    Entries are immutable NPZ payloads. Failed writes are never ready entries;
    eviction removes only our own unlocked, verified-name cache files.
    """
    def __init__(self, root, budget_bytes, reserve_bytes=1024**3, forbidden=()):
        self.root = Path(root).resolve()
        if budget_bytes < 1 or reserve_bytes < 0:
            raise ValueError("positive cache budget and nonnegative reserve required")
        if str(self.root).replace("\\", "/").startswith("/content/drive/"):
            raise ValueError("stream cache must be on the local VM, outside mounted Drive")
        if any(self.root.is_relative_to(Path(path).resolve()) for path in forbidden):
            raise ValueError("stream cache must be outside source/artifact directories")
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget, self.reserve = int(budget_bytes), int(reserve_bytes)
        if any((self.root / name).is_symlink() for name in (".cache.lock", "cache.json")):
            raise ValueError("symlink in owned stream cache")
        self.global_lock = FileLock(str(self.root / ".cache.lock"))
        marker = self.root / "cache.json"
        with self.global_lock:
            if marker.exists():
                if read_json(marker) != {"format": "alignment-transfer-s2ut-stream-cache", "version": 1}:
                    raise ValueError("not an owned stream cache")
            else:
                if any(p.name != ".cache.lock" for p in self.root.iterdir()):
                    raise ValueError("use an empty directory for the stream cache")
                write_json(marker, {"format": "alignment-transfer-s2ut-stream-cache", "version": 1})
            self.items = self.root / "items"
            self.locks = self.root / "locks"
            if any(p.is_symlink() or not p.resolve().is_relative_to(self.root) for p in (self.items, self.locks)):
                raise ValueError("cache directories escape the owned root")
            self.items.mkdir(exist_ok=True)
            self.locks.mkdir(exist_ok=True)
        self.store = ArtifactStore(self.items)

    def _lease(self, pair):
        path = self.locks / (digest(pair) + ".lock")
        if path.is_symlink():
            raise ValueError("symlink in cache lease")
        return FileLock(str(path))

    def _space(self, amount, current):
        """Called with the global lock; never wait for another entry's lease."""
        if amount > self.budget:
            return False
        entries = sorted((p for p in self.items.glob("*.npz") if re.fullmatch(r"[0-9a-f]{64}\.npz", p.name)),
                         key=lambda p: p.stat().st_mtime_ns)
        used = sum(p.stat().st_size + (p.with_suffix(".sha256.json").stat().st_size
                   if p.with_suffix(".sha256.json").exists() else 0) for p in entries)
        for path in entries:
            if used + amount <= self.budget and shutil.disk_usage(self.root).free >= amount + self.reserve:
                return True
            if path == current:
                continue
            lock_path = self.locks / (path.stem + ".lock")
            if lock_path.is_symlink():
                raise ValueError("symlink in cache lease")
            lease = FileLock(str(lock_path))
            try:
                with lease.acquire(timeout=0):
                    for item in (path, path.with_suffix(".sha256.json")):
                        if item.is_symlink():
                            raise ValueError("symlink in owned stream cache")
                        if item.exists():
                            used -= item.stat().st_size
                            item.unlink()
            except Timeout:
                continue
        return used + amount <= self.budget and shutil.disk_usage(self.root).free >= amount + self.reserve

    def get_or_create(self, kind, identity, create):
        pair = digest([kind, identity])
        path = self.store.path(pair)
        with self._lease(pair):
            if path.is_symlink() or path.with_suffix(".sha256.json").is_symlink():
                raise ValueError("symlink in owned stream cache")
            if path.is_file():
                metadata, arrays = self.store.get(pair, {"identity": identity, "kind": kind})
                os.utime(path, None)  # LRU; payload bytes/hash remain immutable.
                return metadata["original_metadata"], arrays
            metadata, arrays = create()
            arrays = {key: np.asarray(value) for key, value in arrays.items()}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("non-finite cached input")
            # Reserve a conservative bound for raw arrays plus JSON/ZIP headers.
            estimate = sum(value.nbytes for value in arrays.values()) + len(str(metadata).encode("utf-8")) + 8192
            with self.global_lock:
                if self._space(estimate, path):
                    self.store.put(pair, {"identity": identity, "kind": kind, "original_metadata": metadata}, arrays)
            # A large item still participates in the ordered stream, without disk caching.
            return metadata, arrays


class CachedArtifactStore:
    """Keep original artifact provenance/path hashes while caching numeric payloads."""
    def __init__(self, root, cache):
        self.original, self.cache = ArtifactStore(root), cache

    def path(self, pair):
        return self.original.path(pair)

    def _read(self, pair):
        path = self.path(pair)
        before = stamp(path)
        checksum = path.with_suffix(".sha256.json")
        identity = [before, stamp(checksum) if checksum.exists() else None]
        def create():
            metadata, arrays = self.original.get(pair)
            sha = file_hash(path)
            if stamp(path) != before:
                raise ValueError("artifact changed during cache loading")
            return {"metadata": metadata, "sha256": sha}, arrays
        return self.cache.get_or_create("artifact", identity, create)

    def get(self, pair, expected=None):
        wrapper, arrays = self._read(pair)
        metadata = wrapper["metadata"]
        for key, value in (expected or {}).items():
            if metadata.get(key) != value:
                raise ValueError("artifact provenance mismatch: " + key)
        return metadata, arrays

    def hash(self, pair):
        return self._read(pair)[0]["sha256"]


def sample_requests(rows, seed, epoch=0, cursor=0):
    """Infinite deterministic sampler; generating requests never commits a cursor."""
    while True:
        order = np.random.default_rng(seed + epoch).permutation(len(rows))
        for position in range(cursor, len(rows)):
            after = (epoch + 1, 0) if position + 1 == len(rows) else (epoch, position + 1)
            yield rows[int(order[position])], after
        epoch, cursor = epoch + 1, 0


class OrderedLoader:
    def __init__(self, requests, load, workers=0, capacity=16, position=(0, 0)):
        if workers < 0 or capacity < 1:
            raise ValueError("invalid loader workers/capacity")
        self.requests, self.load = iter(requests), load
        self.workers, self.capacity, self.position = workers, capacity, position
        self.pool = None
        self.pending, self.ready = deque(), deque()
        self.exhausted = False

    def __enter__(self):
        if self.workers:
            self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="s2ut-input")
            self._fill()
        return self

    def _fill(self):
        while not self.exhausted and len(self.pending) + len(self.ready) < self.capacity:
            try:
                row, after = next(self.requests)
            except StopIteration:
                self.exhausted = True
                break
            self.pending.append((self.pool.submit(self.load, row), after))

    def peek(self, count):
        if not 1 <= count <= self.capacity:
            raise ValueError("peek exceeds the bounded loader capacity")
        while len(self.ready) < count:
            if self.pool:
                if not self.pending:
                    raise StopIteration
                future, after = self.pending.popleft()
                value = future.result()  # Preserve sampler order, propagate errors.
            else:
                row, after = next(self.requests)
                value = self.load(row)
            self.ready.append((value, after))
        return [item[0] for item in list(self.ready)[:count]]

    def take(self, count):
        self.peek(count)
        result = []
        for _ in range(count):
            value, self.position = self.ready.popleft()
            result.append(value)
        if self.pool:
            self._fill()  # Start future reads before the consumer starts GPU work.
        return result

    def __exit__(self, *exception):
        for future, _ in self.pending:
            future.cancel()
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)


@contextmanager
def streaming_batches(*args, **kwargs):
    with OrderedLoader(*args, **kwargs) as loader:
        yield loader
