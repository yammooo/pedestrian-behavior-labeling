"""Immutable pickle-free feature entries, with optional global byte-bounded LRU."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import tempfile
import time

import numpy as np
import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def gb_bytes(value):
    if not np.isfinite(value) or value < 0:
        raise ValueError("Cache GB must be finite and nonnegative")
    return int(value * 10**9)


@contextmanager
def isolated_extraction():
    """Frozen sparse kernels may be nondeterministic; trainable RNG/flags may not change."""
    python, numpy = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            yield
    finally:
        random.setstate(python); np.random.set_state(numpy); torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def validate_arrays(arrays):
    if not arrays or "__identity__" in arrays:
        raise ValueError("Empty/reserved feature payload")
    for name, value in arrays.items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(value, np.ndarray):
            raise ValueError("Feature payload must contain named NumPy arrays")
        if value.dtype.kind not in "biufUS" or (value.dtype.kind == "f" and not np.isfinite(value).all()):
            raise ValueError(f"Nonfinite/unsafe feature array: {name}")


class FeatureCache:
    """One owner, zero workers. Capacity counts actual NPZ bytes, not index/temp files.

    Atomic rename precedes the index commit; interrupted unindexed writes are
    disposable cache data. Indexed missing/corrupt files always fail visibly.
    """

    def __init__(self, root, *, capacity_bytes=None, min_free_bytes=20 * 10**9):
        if capacity_bytes is not None and (type(capacity_bytes) is not int or capacity_bytes < 0):
            raise ValueError("Capacity must be nonnegative integer bytes or None")
        if type(min_free_bytes) is not int or min_free_bytes < 0:
            raise ValueError("Free-space guard must be nonnegative integer bytes")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "owner.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f"Feature cache already in use: {self.root}") from None
        self.db = None
        self.capacity_bytes, self.min_free_bytes = capacity_bytes, min_free_bytes
        self.counts = dict(hits=0, misses=0, read_bytes=0, write_bytes=0, evictions=0,
                           evicted_bytes=0, bypasses=0, read_seconds=0., write_seconds=0., miss_seconds=0.)
        try:
            self.payload = self.root / "entries"
            self.payload.mkdir(exist_ok=True)
            self.db = sqlite3.connect(self.root / "index.sqlite3")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS format (version INTEGER NOT NULL)")
            if not self.db.execute("SELECT * FROM format").fetchall():
                self.db.execute("INSERT INTO format VALUES (1)")
            if self.db.execute("SELECT * FROM format").fetchall() != [(1,)]:
                raise ValueError("Unsupported feature-cache index format")
            self.db.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, identity TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, used INTEGER NOT NULL)")
            self.db.execute("CREATE INDEX IF NOT EXISTS entries_lru ON entries(used, key)")
            self.db.commit()
            self.entry_count, self.resident = self.db.execute("SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM entries").fetchone()
            self.clock = self.db.execute("SELECT COALESCE(MAX(used), 0) FROM entries").fetchone()[0]
            indexed = {key for key, in self.db.execute("SELECT key FROM entries")}
            for path in self.payload.glob("*"):
                if path.name.endswith(".tmp") or (path.suffix == ".npz" and path.stem not in indexed):
                    path.unlink()
            for key in indexed:
                if not self.path(key).is_file():
                    raise ValueError(f"Indexed cache entry missing: {key}")
            self.trim(capacity_bytes)
            self.initial = self.snapshot()
        except BaseException:
            self.close()
            raise

    def path(self, key):
        return self.payload / (key + ".npz")

    def snapshot(self):
        return {**self.counts, "entries": self.entry_count, "resident_bytes": self.resident,
                "capacity_bytes": self.capacity_bytes, "min_free_bytes": self.min_free_bytes, "root": str(self.root)}

    def trim(self, capacity_bytes):
        if capacity_bytes is not None and (type(capacity_bytes) is not int or capacity_bytes < 0):
            raise ValueError("Invalid byte limit")
        self.capacity_bytes = capacity_bytes
        if capacity_bytes is None:
            return
        while self.resident > capacity_bytes:
            key, size = self.db.execute("SELECT key, bytes FROM entries ORDER BY used, key LIMIT 1").fetchone()
            with self.db:
                self.db.execute("DELETE FROM entries WHERE key=?", (key,))
            self.path(key).unlink()
            self.resident -= size
            self.entry_count -= 1
            self.counts["evictions"] += 1
            self.counts["evicted_bytes"] += size

    def contains(self, identity):
        return self.db.execute("SELECT 1 FROM entries WHERE key=?", (identity_digest(identity),)).fetchone() is not None

    def get(self, identity, produce, validate=validate_arrays):
        serialized = canonical(identity)
        key = identity_digest(identity)
        row = self.db.execute("SELECT identity, bytes, sha256 FROM entries WHERE key=?", (key,)).fetchone()
        if row is not None:
            before = time.perf_counter()
            path = self.path(key)
            if row[0] != serialized or not path.is_file() or path.stat().st_size != row[1]:
                raise ValueError(f"Cache identity/size mismatch: {key}")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row[2]:
                raise ValueError(f"Cache checksum mismatch: {key}")
            from io import BytesIO
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                if str(archive["__identity__"]) != serialized:
                    raise ValueError("Feature payload identity mismatch")
                arrays = {k: archive[k] for k in archive.files if k != "__identity__"}
            validate_arrays(arrays); validate(arrays)
            self.clock += 1
            with self.db:
                self.db.execute("UPDATE entries SET used=? WHERE key=?", (self.clock, key))
            self.counts["hits"] += 1
            self.counts["read_bytes"] += len(raw)
            self.counts["read_seconds"] += time.perf_counter() - before
            return arrays
        self.counts["misses"] += 1
        before = time.perf_counter()
        arrays = produce()
        self.counts["miss_seconds"] += time.perf_counter() - before
        validate_arrays(arrays); validate(arrays)
        # NPZ is uncompressed: exact size is known without allocating a second payload.
        saved = dict(arrays, __identity__=np.array(serialized))
        size = sum(30 + len((name + ".npy").encode()) + 20 + self._npy_size(value) +
                   46 + len((name + ".npy").encode()) for name, value in saved.items()) + 22
        if self.capacity_bytes is not None and size > self.capacity_bytes:
            self.counts["bypasses"] += 1
            return arrays
        if shutil.disk_usage(self.root).free < self.min_free_bytes + size:
            raise OSError(f"Insufficient atomic-write headroom at {self.root}: require payload + {self.min_free_bytes} free bytes")
        before = time.perf_counter()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.payload, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, **saved)
                stream.flush(); os.fsync(stream.fileno())
            actual = temporary.stat().st_size
            if actual != size:
                raise ValueError(f"NPZ size accounting changed: expected {size}, got {actual}")
            with temporary.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            # Check before eviction so a low-space failure preserves existing entries.
            if shutil.disk_usage(self.root).free < self.min_free_bytes:
                raise OSError("Free-space guard reached during cache write")
            if self.capacity_bytes is not None:
                capacity = self.capacity_bytes
                self.trim(capacity - actual)
                self.capacity_bytes = capacity
            os.replace(temporary, self.path(key))
            directory = os.open(self.payload, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self.clock += 1
            with self.db:
                self.db.execute("INSERT INTO entries VALUES (?, ?, ?, ?, ?)", (key, serialized, actual, digest, self.clock))
            self.resident += actual
            self.entry_count += 1
            self.counts["write_bytes"] += actual
            self.counts["write_seconds"] += time.perf_counter() - before
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return arrays

    @staticmethod
    def _npy_size(value):
        from io import BytesIO
        header = BytesIO()
        np.lib.format.write_array_header_1_0(header, np.lib.format.header_data_from_array_1_0(value))
        return header.tell() + value.nbytes

    def close(self):
        if self.db is not None:
            self.db.close(); self.db = None
        if not self.lock.closed:
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
