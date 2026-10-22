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

    def __init__(self, root, manifest_sha256, max_verified_chunks=512):
        self.root = Path(root)
        if max_verified_chunks <= 0 or checksum(self.root / "manifest.json") != manifest_sha256:
            raise ValueError("Invalid verification budget or manifest hash")
        self.record = json.loads((self.root / "manifest.json").read_text())
        if self.record["protocol"] != PROTOCOL or set(self.record["arrays"]) != set(DTYPES) or self.record["chunk_rows"] != CHUNK_ROWS:
            raise ValueError("Unsupported indexed shard")
        self.maps, self.stamps, self.verified = {}, {}, OrderedDict()
        self.limit = max_verified_chunks
        self.stats = {"selected_bytes": 0, "verified_bytes": 0, "checked_chunks": 0, "requests": 0}
        for name, (dtype, width) in DTYPES.items():
            spec = self.record["arrays"][name]
            path = self.root / (name + ".bin")
            if (spec["dtype"], spec["width"]) != (dtype, width) or spec["bytes"] != spec["rows"] * width * np.dtype(dtype).itemsize:
                raise ValueError("Invalid indexed array shape")
            if len(spec["chunks"]) != (spec["rows"] + CHUNK_ROWS - 1) // CHUNK_ROWS:
                raise ValueError("Invalid chunk index")
            stat = path.stat()
            if stat.st_size != spec["bytes"]:
                raise ValueError("Truncated indexed array")
            self.stamps[name] = stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino
            shape = (spec["rows"], width) if width > 1 else (spec["rows"],)
            self.maps[name] = np.memmap(path, mode="r", dtype=dtype, shape=shape) if spec["rows"] else np.empty(shape, dtype=dtype)

    def take(self, name, rows):
        rows = np.asarray(rows, dtype=np.int64)
        spec = self.record["arrays"][name]
        if rows.ndim != 1 or (rows < 0).any() or (rows >= spec["rows"]).any():
            raise ValueError("Indexed row outside shard")
        path = self.root / (name + ".bin")
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino) != self.stamps[name]:
            raise ValueError("Immutable shard changed after opening")
        row_bytes = np.dtype(spec["dtype"]).itemsize * spec["width"]
        with path.open("rb") as stream:
            for chunk in np.unique(rows // CHUNK_ROWS):
                key = name, int(chunk)
                if key in self.verified:
                    self.verified.move_to_end(key)
                    continue
                stream.seek(int(chunk) * CHUNK_ROWS * row_bytes)
                raw = stream.read(CHUNK_ROWS * row_bytes)
                if hashlib.sha256(raw).hexdigest() != spec["chunks"][int(chunk)]:
                    raise ValueError("Indexed chunk checksum mismatch")
                self.stats["verified_bytes"] += len(raw)
                self.stats["checked_chunks"] += 1
                self.verified[key] = None
                if len(self.verified) > self.limit:
                    self.verified.popitem(last=False)
        self.stats["selected_bytes"] += len(rows) * row_bytes
        return np.array(self.maps[name][rows], copy=True)

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
        features = self.take("features", rows).astype(np.float32)
        coordinates = self.take("coordinates", rows)
        box = np.array(selection["box"])
        relative = relative_xyz(coordinates, box[:3], np.array(meta["world_from_native"]), np.array(selection["anchor"]))
        inside = box_membership(coordinates, box[:3], box[3:6], box[6])
        if not np.isfinite(features).all() or not np.isfinite(relative).all():
            raise ValueError("Nonfinite indexed crop")
        self.stats["requests"] += 1
        return np.c_[features, relative / 20, inside].astype(np.float32)

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
