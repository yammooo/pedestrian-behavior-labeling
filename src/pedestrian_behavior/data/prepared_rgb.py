"""Hits-only, read-only access to the existing prepared RGB NPZ cache."""

from io import BytesIO
import hashlib
from pathlib import Path
import sqlite3
import time
from threading import Lock

import numpy as np

from .feature_cache import canonical, identity_digest, validate_arrays


class PreparedRGB:
    def __init__(self, root):
        self.root = Path(root).resolve()
        # One serialized read-only connection; producer must join before final close.
        self.lock = Lock()
        self.db = sqlite3.connect((self.root / "index.sqlite3").as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        if self.db.execute("SELECT * FROM format").fetchall() != [(1,)]:
            self.db.close()
            raise ValueError("Unsupported prepared RGB index")
        self.stats = {"hits": 0, "read_bytes": 0, "read_seconds": 0.}
        self.initial = self.snapshot()

    def get(self, identity, produce, validate=validate_arrays):
        with self.lock:
            return self._get(identity, validate)

    def _read(self, identity, fields=None):
        begin = time.perf_counter()
        serialized, key = canonical(identity), identity_digest(identity)
        row = self.db.execute("SELECT identity, bytes, sha256 FROM entries WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Prepared RGB cache miss; inference is disabled")
        path = self.root / "entries" / (key + ".npz")
        if row[0] != serialized or path.stat().st_size != row[1]:
            raise ValueError("Prepared RGB identity/size mismatch")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row[2]:
            raise ValueError("Prepared RGB checksum mismatch")
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            if str(archive["__identity__"]) != serialized:
                raise ValueError("Prepared RGB payload identity mismatch")
            names = fields or (k for k in archive.files if k != "__identity__")
            arrays = {k: archive[k] for k in names}
        return arrays, len(raw), time.perf_counter() - begin

    def _get(self, identity, validate):
        arrays, size, elapsed = self._read(identity)
        validate_arrays(arrays)
        validate(arrays)
        self.stats["hits"] += 1
        self.stats["read_bytes"] += size
        self.stats["read_seconds"] += elapsed
        return arrays

    def valid(self, identity):
        """Read only the cached availability vector before a model view is selected."""
        with self.lock:
            arrays, size, elapsed = self._read(identity, ("valid",))
            valid = arrays["valid"]
            if valid.ndim != 1 or valid.dtype != np.bool_:
                raise ValueError("Invalid prepared RGB validity metadata")
            self.stats["validity_reads"] = self.stats.get("validity_reads", 0) + 1
            self.stats["read_bytes"] += size
            self.stats["read_seconds"] += elapsed
            return valid.copy()

    def snapshot(self):
        return self.stats | {"root": str(self.root), "read_only": True}

    def close(self):
        with self.lock:
            self.db.close()
