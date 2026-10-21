"""Preparation-only bounded decoded-image reuse; scientific extraction stays per track."""

from collections import OrderedDict, defaultdict

from pedestrian_behavior.inspection.native_features import native_camera


class CameraReuse:
    def __init__(self, catalogs, selected, capacity_bytes=2_000_000_000):
        self.groups = defaultdict(list)
        for dataset, key in selected:
            track = catalogs[dataset].tracks[key]
            self.groups[dataset, track["group"]].append(track)
        self.capacity = capacity_bytes
        self.active = None
        self.images = OrderedDict()
        self.bytes = self.peak_bytes = self.decoded = self.hits = 0

    def snapshot(self):
        return {"capacity_bytes": self.capacity, "peak_decoded_bytes": self.peak_bytes,
                "decoded_frames": self.decoded, "reused_frames": self.hits}

    def __call__(self, reader, group, keys, tracks, frozen):
        site = reader.dataset, group
        keys = sorted(set(keys), key=int)
        if self.active != site:
            self.images.clear(); self.bytes = 0; self.active = site
            # One scene traversal, even when its images serve many separate tracks.
            members = self.groups[site]
            requested = sorted({k for t in members for k in t["source_keys"] if k is not None}, key=int)
            selected_ids = [t["track_id"] for t in members]
        else:
            requested = [k for k in keys if k not in self.images]
            selected_ids = [t["track_id"] for t in self.groups[site]]
            self.hits += len(keys) - len(requested)
        for key, image, boxes in native_camera(reader, group, requested, selected_ids, frozen) if requested else ():
            size = image.width * image.height * 3 if image is not None else 0
            if size > self.capacity:
                raise ValueError("Native image exceeds preparation-only RAM cache capacity")
            while self.images and self.bytes + size > self.capacity:
                _, (_, _, old) = self.images.popitem(last=False)
                self.bytes -= old
            self.images[key] = image, boxes, size
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.decoded += 1
        for key in keys:
            if key not in self.images:
                # A scene larger than RAM capacity may evict requested earlier frames.
                for k, image, boxes in native_camera(reader, group, [key], tracks, frozen):
                    yield k, image, boxes
            else:
                image, boxes, _ = self.images[key]
                self.images.move_to_end(key)
                yield key, image, {t: boxes[t] for t in tracks if t in boxes}
