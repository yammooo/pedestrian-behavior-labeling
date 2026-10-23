"""Target-specific coarse/fine point tables and immutable indexed shards."""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .frozen_features import validate_points
from .preparation import checksum
from pedestrian_behavior.inspection.feature_probe import box_membership, relative_xyz
from pedestrian_behavior.inspection.output_pooling import pool_outputs


PROTOCOL = "target-specific-20cm-5cm-fp16-v1"
DTYPES = {"features": ("<f2", 64), "coordinates": ("<f4", 3), "native_reference": ("<i8", 1),
          "cell": ("<u4", 1), "fine": ("u1", 1), "members": ("<u4", 1), "indices": ("<u4", 1)}
CHUNK_ROWS = 8192


def file_stamp(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev


def row_chunks(rows):
    chunks = rows // CHUNK_ROWS
    if len(chunks) < 2:
        return chunks
    if (chunks[1:] < chunks[:-1]).any():
        return np.unique(chunks)
    return chunks[np.r_[True, chunks[1:] != chunks[:-1]]]


def build_frame(arrays):
    """20cm means; retain original rows in whole cells refined by any target's .5m halo."""
    validate_points(arrays)
    xyz, features = arrays["coordinates"], arrays["features"]
    centers, means, inverse, counts = pool_outputs(xyz, features, .2)
    c = len(centers)
    refinements, union = [], np.zeros(c, bool)
    for box in arrays["boxes"]:
        marked = np.zeros(c, bool)
        marked[inverse[box_membership(xyz, box[:3], box[3:6] + 1., box[6])]] = True
        refinements.append(marked)
        union |= marked
    fine_rows = np.flatnonzero(union[inverse])
    cells = np.r_[np.arange(c), inverse[fine_rows]]
    order = np.argsort(cells, kind="stable")  # Cell order is lexicographic native XYZ, coarse then fine.
    representatives = np.full(c, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(representatives, inverse, arrays["native_indices"])
    full_features = np.r_[means, features[fine_rows]][order]
    half = full_features.astype(np.float16)
    if not np.isfinite(half).all() or len(half) >= 2**32:
        raise ValueError("FP16 overflow or uint32 row limit")
    result = {"features": half, "coordinates": np.r_[centers, xyz[fine_rows]][order],
        "native_reference": np.r_[representatives, arrays["native_indices"][fine_rows]][order],
        "cell": cells[order].astype(np.uint32), "fine": np.r_[np.zeros(c, np.uint8), np.ones(len(fine_rows), np.uint8)][order],
        "members": np.r_[counts, np.ones(len(fine_rows), np.int64)][order].astype(np.uint32)}
    selections, indices, offset = [], [], 0
    for locator, box, anchor, marked in zip(arrays["locators"], arrays["boxes"], arrays["anchors"], refinements):
        choose = result["fine"].astype(bool) == marked[result["cell"]]
        relative = relative_xyz(result["coordinates"], box[:3], arrays["world_from_native"], anchor)
        rows = np.flatnonzero(choose & (np.linalg.norm(relative[:, :2], axis=1) <= 20)).astype(np.uint32)
        # Independent checks: no cell supplies both tiers; every original target return survives.
        for level in (0, 1):
            selected_cells = result["cell"][rows[result["fine"][rows] == level]]
            if level == 0:
                coarse_cells = selected_cells
            elif np.intersect1d(coarse_cells, selected_cells).size:
                raise AssertionError("Duplicate coarse/fine cell")
        original_inside = box_membership(xyz, box[:3], box[3:6], box[6])
        original_inside &= np.linalg.norm(relative_xyz(xyz, box[:3], arrays["world_from_native"], anchor)[:, :2], axis=1) <= 20
        retained = result["native_reference"][rows[result["fine"][rows].astype(bool)]]
        if not np.isin(arrays["native_indices"][original_inside], retained).all():
            raise AssertionError("Target token lost")
        selections.append({"locator": str(locator), "index_start": offset, "count": len(rows),
                           "box": box.tolist(), "anchor": anchor.tolist(), "original_target_tokens": int(original_inside.sum())})
        indices.append(rows)
        offset += len(rows)
    result["indices"] = np.concatenate(indices) if indices else np.zeros(0, np.uint32)
    energy = float(np.square(full_features.astype(np.float64)).sum())
    metadata = {"world_from_native": arrays["world_from_native"].tolist(), "selections": selections,
        "original_tokens": len(xyz), "coarse_tokens": c, "fine_tokens": len(fine_rows),
        "fp16_relative_l2": (float(np.square(full_features.astype(np.float64) - half).sum()) / energy)**.5 if energy else 0.}
    return result, metadata


class ShardWriter:
    """Append bounded frames; publish the manifest last. Failed directories cannot be reopened."""

    def __init__(self, root, identity, min_free_bytes=0):
        if min_free_bytes < 0:
            raise ValueError("Negative free-space floor")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.min_free_bytes = min_free_bytes
        self.streams = {k: (self.root / (k + ".bin")).open("xb") for k in DTYPES}
        self.record = {"protocol": PROTOCOL, "identity": identity, "chunk_rows": CHUNK_ROWS, "frames": {}, "arrays": {}}
        self.rows, self.indices = 0, 0

    def add(self, key, arrays):
        if str(key) in self.record["frames"]:
            raise ValueError("Duplicate native frame")
        payload, metadata = build_frame(arrays)
        needed = sum(value.nbytes for value in payload.values())
        if shutil.disk_usage(self.root).free < self.min_free_bytes + needed:
            raise OSError("Insufficient disk space for indexed frame and free-space floor")
        for name, (dtype, width) in DTYPES.items():
            np.asarray(payload[name], dtype=dtype).tofile(self.streams[name])
        metadata.update(start=self.rows, count=len(payload["features"]), index_start=self.indices)
        self.record["frames"][str(key)] = metadata
        self.rows += metadata["count"]
        self.indices += len(payload["indices"])
        return metadata

    def finish(self):
        for stream in self.streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        for name, (dtype, width) in DTYPES.items():
            path = self.root / (name + ".bin")
            chunks, block = [], CHUNK_ROWS * np.dtype(dtype).itemsize * width
            with path.open("rb") as stream:
                while raw := stream.read(block):
                    chunks.append(hashlib.sha256(raw).hexdigest())
            self.record["arrays"][name] = {"dtype": dtype, "width": width, "rows": self.indices if name == "indices" else self.rows,
                "bytes": path.stat().st_size, "sha256": checksum(path), "chunks": chunks}
        temporary = self.root / "manifest.tmp"
        with temporary.open("w") as stream:
            stream.write(json.dumps(self.record, separators=(",", ":"), allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.root / "manifest.json")
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return checksum(self.root / "manifest.json")

    def close(self):
        for stream in self.streams.values():
            stream.close()


class IndexedShard:
    """Read selected rows, verify touched chunks, and bound verification memoization."""

    def __init__(self, root, manifest_sha256, max_verified_chunks=512, *, _record=None, _state=None):
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if _record is not None and _state is None:
            raise ValueError("Cached metadata requires pinned verification state")
        if _state is not None and file_stamp(manifest) != _state["manifest"]:
            raise ValueError("Immutable shard manifest changed")
        if max_verified_chunks <= 0 or (_record is None and checksum(manifest) != manifest_sha256):
            raise ValueError("Invalid verification budget or manifest hash")
        self.record = _record if _record is not None else json.loads(manifest.read_text())
        if self.record["protocol"] != PROTOCOL or set(self.record["arrays"]) != set(DTYPES) or self.record["chunk_rows"] != CHUNK_ROWS:
            raise ValueError("Unsupported indexed shard")
        self.maps, self.stamps, self.verified = {}, {}, OrderedDict()
        self.limit = max_verified_chunks
        self.state = _state
        self.stats = {"selected_bytes": 0, "verified_bytes": 0, "checked_chunks": 0, "requests": 0}
        for name, (dtype, width) in DTYPES.items():
            spec = self.record["arrays"][name]
            path = self.root / (name + ".bin")
            if (spec["dtype"], spec["width"]) != (dtype, width) or spec["bytes"] != spec["rows"] * width * np.dtype(dtype).itemsize:
                raise ValueError("Invalid indexed array shape")
            if len(spec["chunks"]) != (spec["rows"] + CHUNK_ROWS - 1) // CHUNK_ROWS:
                raise ValueError("Invalid chunk index")
            stamp = file_stamp(path)
            if stamp[0] != spec["bytes"]:
                raise ValueError("Truncated indexed array")
            if _state is not None and stamp != _state["arrays"][name]:
                raise ValueError("Immutable shard changed before reopening")
            self.stamps[name] = stamp
            shape = (spec["rows"], width) if width > 1 else (spec["rows"],)
            self.maps[name] = np.memmap(path, mode="r", dtype=dtype, shape=shape) if spec["rows"] else np.empty(shape, dtype=dtype)

    def take(self, name, rows, *, _chunks=None):
        rows = np.asarray(rows, dtype=np.int64)
        spec = self.record["arrays"][name]
        if rows.ndim != 1 or (rows < 0).any() or (rows >= spec["rows"]).any():
            raise ValueError("Indexed row outside shard")
        path = self.root / (name + ".bin")
        if file_stamp(path) != self.stamps[name]:
            raise ValueError("Immutable shard changed after opening")
        row_bytes = np.dtype(spec["dtype"]).itemsize * spec["width"]
        chunks = row_chunks(rows) if _chunks is None else _chunks
        flags = self.state["verified"].get(name) if self.state is not None else None
        if self.state is not None and flags is None:
            flags = self.state["verified"][name] = np.zeros(len(spec["chunks"]), bool)
        for chunk in chunks:
            key = name, int(chunk)
            if flags is not None:
                if flags[chunk]:
                    continue
            elif key in self.verified:
                self.verified.move_to_end(key)
                continue
            start = int(chunk) * CHUNK_ROWS
            raw = memoryview(self.maps[name][start:start + CHUNK_ROWS]).cast("B")
            if hashlib.sha256(raw).hexdigest() != spec["chunks"][int(chunk)]:
                raise ValueError("Indexed chunk checksum mismatch")
            self.stats["verified_bytes"] += len(raw)
            self.stats["checked_chunks"] += 1
            if flags is not None:
                flags[chunk] = True
            else:
                self.verified[key] = None
                if len(self.verified) > self.limit:
                    self.verified.popitem(last=False)
        self.stats["selected_bytes"] += len(rows) * row_bytes
        # Integer-array indexing already returns an owned copy, independent of mmap lifetime.
        return self.maps[name][rows]

    def crop(self, frame, locator):
        meta = self.record["frames"][str(frame)]
        found = [s for s in meta["selections"] if s["locator"] == locator]
        if len(found) != 1:
            raise ValueError("Target absent or duplicated in indexed frame")
        selection = found[0]
        start = meta["index_start"] + selection["index_start"]
        rows = self.take("indices", np.arange(start, start + selection["count"], dtype=np.int64)).astype(np.int64)
        if (rows >= meta["count"]).any() or (np.diff(rows) <= 0).any():
            raise ValueError("Invalid target row selection")
        rows += meta["start"]
        chunks = row_chunks(rows)
        result = np.empty((len(rows), 68), np.float32)
        result[:, :64] = self.take("features", rows, _chunks=chunks)
        coordinates = self.take("coordinates", rows, _chunks=chunks)
        box = np.array(selection["box"])
        relative = relative_xyz(coordinates, box[:3], np.array(meta["world_from_native"]), np.array(selection["anchor"]))
        inside = box_membership(coordinates, box[:3], box[3:6], box[6])
        result[:, 64:67] = relative / 20
        result[:, 67] = inside
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite indexed crop")
        self.stats["requests"] += 1
        return result

    def close(self):
        for array in self.maps.values():
            if isinstance(array, np.memmap):
                array._mmap.close()
        self.maps.clear()

    def evict_pages(self):
        """Best-effort eviction for this prototype's own files, never global drop_caches."""
        self.close()
        if not hasattr(os, "posix_fadvise"):
            return False
        for name in DTYPES:
            with (self.root / (name + ".bin")).open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return True
