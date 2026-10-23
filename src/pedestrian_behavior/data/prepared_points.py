"""Read-only source-level indexed points, complete-slot validation and packing costs."""

from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .indexed_points import DTYPES, IndexedShard, PROTOCOL, file_stamp


def metadata_size(record):
    """Account decoded JSON containers/values; shared objects are counted once per record."""
    stack, seen, total = [record], set(), 0
    while stack:
        value = stack.pop()
        key = id(value)
        if key in seen:
            continue
        seen.add(key)
        total += sys.getsizeof(value)
        if isinstance(value, dict):
            stack.extend(value.keys()); stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return total


class PreparedPoints:
    def __init__(self, root, dataset, population, tracks, encoder, *, allow_partial=False, max_open_scenes=8,
                 max_metadata_bytes=512 * 2**20):
        self.root = Path(root).resolve()
        path = self.root / "manifest.json"
        raw = path.read_bytes()  # An in-progress builder has no root manifest.
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.record = json.loads(raw)
        r = self.record
        if (r.get("format") != "e002a-indexed-source-v1" or r.get("protocol") != PROTOCOL or
                r.get("status") != "complete" or r.get("dataset") != dataset or r.get("population") != population or
                any(r.get("encoder", {}).get(k) != v for k, v in encoder.items())):
            raise ValueError("Indexed cache identity/population/encoder mismatch")
        if r.get("scope") != "full-frozen-source-train-validation" and not (
                allow_partial and r.get("scope") == "bounded-native-check"):
            raise ValueError("Full frozen source indexed cache required")
        if max_open_scenes < 1 or max_metadata_bytes < 0:
            raise ValueError("Positive open-scene budget required")
        self.stamp = self._stamp(path)
        self.protected = {str(path): self.sha256}
        self.limit, self.opened = max_open_scenes, OrderedDict()
        self.states, self.metadata = {}, OrderedDict()
        self.metadata_bytes, self.metadata_limit = 0, max_metadata_bytes
        self.counts, self.references = {}, {}
        self.stats = {"track_reads": 0, "requests": 0, "scene_opens": 0, "selected_bytes": 0,
                      "verified_bytes": 0, "checked_chunks": 0, "metadata_loads": 0, "metadata_hits": 0}
        entries = {t["locator"]: t for t in r["tracks"]}
        if len(entries) != len(r["tracks"]) or not set(entries).issubset(tracks):
            raise ValueError("Duplicate/unknown indexed tracks")
        if r["scope"] == "full-frozen-source-train-validation" and set(entries) != set(tracks):
            raise ValueError("Indexed cache changed frozen source population")
        expected = defaultdict(lambda: defaultdict(set))
        for locator, t in entries.items():
            frozen = tracks[locator]
            if any(t[k] != frozen[k] for k in ("archive", "group", "split", "slots", "sha256", "source_keys", "times_us")):
                raise ValueError("Indexed complete-track references differ from frozen tracks")
            usable = frozen["point_usable"].tolist()
            if t["point_usable"] != usable or len(t["point_frames"]) != t["slots"]:
                raise ValueError("Indexed slot validity mismatch")
            refs = t["point_frames"]
            for slot, ref in enumerate(refs):
                wanted = str(t["source_keys"][slot]) if usable[slot] else None
                if ref != wanted and not (allow_partial and ref is None):
                    raise ValueError("Missing/invalid indexed slot reference")
                if ref is not None:
                    expected[t["group"]][ref].add(locator)
            self.references[locator] = t
            self.counts[locator] = np.zeros(t["slots"], np.int64)
        if set(r["scenes"]) != {t["group"] for t in entries.values()}:
            raise ValueError("Indexed scene population mismatch")
        # Scan only small manifests once for all packing costs, not feature payloads.
        for group, spec in r["scenes"].items():
            folder = self.root / spec["path"]
            if not folder.resolve().is_relative_to(self.root) or folder.resolve() == self.root:
                raise ValueError("Unsafe indexed scene path")
            manifest = folder / "manifest.json"
            manifest_stamp = self._stamp(manifest)
            raw = manifest.read_bytes()
            if hashlib.sha256(raw).hexdigest() != spec["manifest_sha256"] or self._stamp(manifest) != manifest_stamp:
                raise ValueError("Indexed scene manifest checksum mismatch")
            self.protected[str(manifest)] = spec["manifest_sha256"]
            meta = json.loads(raw)
            self.stats["metadata_loads"] += 1
            if (meta.get("protocol") != PROTOCOL or meta["identity"]["dataset"] != dataset or
                    meta["identity"]["group"] != group or meta["identity"]["population"] != population or
                    set(meta["frames"]) != set(expected[group]) or set(meta["arrays"]) != set(DTYPES)):
                raise ValueError("Indexed scene identity/frames mismatch")
            for name, array in meta["arrays"].items():
                if (folder / (name + ".bin")).stat().st_size != array["bytes"]:
                    raise ValueError("Missing/truncated prepared point array")
            self.states[group] = {"manifest": manifest_stamp,
                "arrays": {name: self._stamp(folder / (name + ".bin")) for name in DTYPES}, "verified": {}}
            self._remember(group, meta)
            lookup = {}
            for key, frame in meta["frames"].items():
                selections = {s["locator"]: s for s in frame["selections"]}
                if len(selections) != len(frame["selections"]) or set(selections) != expected[group][key]:
                    raise ValueError("Indexed frame target population mismatch")
                for locator, selection in selections.items():
                    count = selection["count"]
                    if type(count) is not int or count < 0:
                        raise ValueError("Invalid indexed token count")
                    lookup[key, locator] = count
            for locator, t in entries.items():
                if t["group"] == group:
                    self.counts[locator] = np.array([lookup[ref, locator] if ref is not None else 0
                                                    for ref in t["point_frames"]], np.int64)
        self.identity = {"protocol": PROTOCOL, "manifest_sha256": self.sha256, "root": str(self.root)}
        self.initial = self.snapshot()

    @staticmethod
    def _stamp(path):
        return file_stamp(path)

    def _remember(self, group, record):
        size = metadata_size(record)
        if size > self.metadata_limit:
            return
        while self.metadata_bytes + size > self.metadata_limit:
            _, (_, old_size) = self.metadata.popitem(last=False)
            self.metadata_bytes -= old_size
        self.metadata[group] = record, size
        self.metadata_bytes += size

    def _metadata(self, group):
        spec = self.record["scenes"][group]
        path = self.root / spec["path"] / "manifest.json"
        if self._stamp(path) != self.states[group]["manifest"]:
            raise ValueError("Immutable shard manifest changed")
        if group in self.metadata:
            self.metadata.move_to_end(group)
            self.stats["metadata_hits"] += 1
            return self.metadata[group][0]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["manifest_sha256"] or self._stamp(path) != self.states[group]["manifest"]:
            raise ValueError("Indexed scene manifest checksum mismatch")
        record = json.loads(raw)
        self.stats["metadata_loads"] += 1
        self._remember(group, record)
        return record

    def require_track(self, locator):
        if self._stamp(self.root / "manifest.json") != self.stamp:
            raise ValueError("Immutable prepared source manifest changed")
        t = self.references[locator]
        if any(v and ref is None for v, ref in zip(t["point_usable"], t["point_frames"])):
            raise ValueError("Partial cache does not cover the complete selected track")
        return t

    def track_cost(self, locator):
        self.require_track(locator)
        return int(self.counts[locator].sum())

    def _reader(self, group):
        if self._stamp(self.root / "manifest.json") != self.stamp:
            raise ValueError("Immutable prepared source manifest changed")
        if group in self.opened:
            self.opened.move_to_end(group)
            return self.opened[group]
        if len(self.opened) >= self.limit:
            _, reader = self.opened.popitem(last=False)
            self._retire(reader)
        spec = self.record["scenes"][group]
        reader = IndexedShard(self.root / spec["path"], spec["manifest_sha256"],
                              _record=self._metadata(group), _state=self.states[group])
        self.opened[group] = reader
        self.stats["scene_opens"] += 1
        return reader

    def iter_track(self, locator):
        t = self.require_track(locator)
        self.stats["track_reads"] += 1
        for slot, ref in enumerate(t["point_frames"]):
            if ref is None:
                yield np.zeros((0, 68), np.float32)
            else:
                crop = self._reader(t["group"]).crop(ref, locator)
                if len(crop) != self.counts[locator][slot]:
                    raise ValueError("Prepared point count changed")
                yield crop

    def _retire(self, reader):
        for key in ("requests", "selected_bytes", "verified_bytes", "checked_chunks"):
            self.stats[key] += reader.stats[key]
        reader.close()

    def snapshot(self):
        result = dict(self.stats)
        for reader in self.opened.values():
            for key in ("requests", "selected_bytes", "verified_bytes", "checked_chunks"):
                result[key] += reader.stats[key]
        return result | {"root": str(self.root), "read_only": True, "open_scenes": len(self.opened),
                         "metadata_bytes": self.metadata_bytes, "metadata_limit_bytes": self.metadata_limit,
                         "verification_bytes": sum(v.nbytes for s in self.states.values() for v in s["verified"].values())}

    def reset_verification(self):
        """Explicit diagnostic reset; normal reader eviction/epoch boundaries retain trust."""
        self.close()
        for state in self.states.values():
            state["verified"].clear()

    def close(self):
        for reader in self.opened.values():
            self._retire(reader)
        self.opened.clear()
