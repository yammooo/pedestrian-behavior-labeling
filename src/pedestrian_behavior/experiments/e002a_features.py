"""E002a frozen 3x RGB CLS and shared 20 m D-DITR feature supply; no behavior training."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.feature_cache import FeatureCache, canonical, gb_bytes, identity_digest, isolated_extraction
from pedestrian_behavior.data.frozen_features import empty_crop, point_crop, point_union, validate_points, validate_rgb
from pedestrian_behavior.data.preparation import checksum, load_track, source_times, valid_pose
from pedestrian_behavior.datasets import loki, road_waymo
from pedestrian_behavior.inspection.feature_probe import normalized_crop, square_crop
from pedestrian_behavior.inspection.native_features import native_camera, native_scene
from pedestrian_behavior.inspection.pretrained import (DINO_REVISION, DINO_PTH_SHA256, DINO_HF_SHA256,
    DITR_REVISION, DITR_SHA256, DITR_CONFIG_SHA256, ImageEncoder, PointEncoder)
from .e002a_probe import write_json


RGB_BATCH = 16


def backend_identity(device):
    packages = {}
    for name in ("torch", "numpy", "Pillow", "spconv-cu126", "cumm-cu126", "flash-attn", "torch-scatter", "timm"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"packages": packages, "device": str(device), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
            "cudnn": torch.backends.cudnn.version()}


def extraction_identity(device, config=None):
    # Relevant source content, not the Git revision: doc-only commits do not expire features.
    from pedestrian_behavior.inspection import pretrained, feature_probe, native_features
    from pedestrian_behavior.data import feature_cache, frozen_features
    return {"schema": 1, "backend": backend_identity(device),
            "rgb_checkpoint_sha256": config["rgb_checkpoint_sha256"] if config else DINO_PTH_SHA256,
            "implementation": {m.__name__: checksum(m.__file__) for m in
                               (pretrained, feature_probe, native_features, feature_cache, frozen_features)},
            "policy_sha256": checksum(__file__)}


class FrozenEncoders:
    """Lazy, one resident encoder; init and inference are both isolated from fitting RNG."""

    def __init__(self, config, device):
        self.config, self.device = config, device
        self.encoder = None
        self.kind = None
        self.calls = {"rgb": 0, "points": 0}
        self.loads = {"rgb": 0, "points": 0}
        self.seconds = {"rgb": 0., "points": 0.}

    def __call__(self, kind, inputs):
        before = time.perf_counter()
        with isolated_extraction():
            if self.kind != kind:
                self.release()
                c = self.config
                self.encoder = (ImageEncoder(c["dino_code"], c["dino_weights"], self.device) if kind == "rgb" else
                    PointEncoder(c["ditr_code"], c["ditr_config"], c["ditr_weights"], self.device))
                self.kind = kind
                self.loads[kind] += 1
            result = self.encoder(inputs)
        self.calls[kind] += 1
        self.seconds[kind] += time.perf_counter() - before
        return result

    def release(self):
        self.encoder = None
        self.kind = None
        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


class FeatureCatalog:
    """The entire frozen eligible train/validation population; never reads test archives."""

    def __init__(self, outputs_root, dataset, config, transform):
        self.dataset, self.config = dataset, config
        root = Path(outputs_root).resolve()
        self.collection = root / "experiments/E001/reader-preparation" / dataset
        setup_path = root / "experiments/E001/data-setup" / dataset / "setup.json"
        manifest_path = self.collection / "manifest.json"
        self.setup, self.manifest = (json.loads(p.read_text()) for p in (setup_path, manifest_path))
        if self.setup["status"] != "complete" or self.setup["collection"]["manifest_sha256"] != checksum(manifest_path):
            raise ValueError("Frozen source setup/collection mismatch")
        self.protected = {str(p): checksum(p) for p in (setup_path, manifest_path)}
        self.tracks = {}
        self.frames = defaultdict(list)
        for entry in self.setup["tracks"]:
            if entry["exclusions"] or entry["split"] not in ("training", "validation"):
                continue
            if Path(entry["archive"]).name != entry["archive"] or not entry["archive"].endswith(".npz"):
                raise ValueError("Unsafe saved-track reference")
            path = self.collection / "tracks" / entry["archive"]
            m, arrays = load_track(path)
            if (m["track_locator"] != entry["locator"] or len(m["source_frames"]) != entry["slots"] or
                    self.setup["groups"][entry["group"]] != entry["split"] or entry["locator"] in self.tracks):
                raise ValueError("Frozen track identity/split/length mismatch")
            self.protected[str(path)] = checksum(path)
            identity = json.loads(entry["locator"])
            keys = m["source_frames"]
            usable = np.array([k is not None and m["anchor_valid"] and arrays["ped_position_valid"][i]
                and arrays["ego_pose_valid"][i] and m["availability"][i].get("native_3d", False)
                for i, k in enumerate(keys)], bool)
            t = {**entry, "track_id": identity.get("track_id", identity.get("tube_uid")),
                 "source_keys": keys, "times_us": source_times(m), "point_usable": usable,
                 "sha256": self.protected[str(path)]}
            self.tracks[entry["locator"]] = t
            for slot in np.flatnonzero(usable):
                self.frames[entry["group"], str(keys[slot])].append(entry["locator"])
        self.identity = {"dataset": dataset, "manifest_sha256": checksum(manifest_path),
            "setup_sha256": checksum(setup_path), "tracks_sha256": identity_digest(
                sorted((k, t["sha256"]) for k,t in self.tracks.items()))}
        self.transform = Path(transform)
        if dataset == "loki" and json.loads(self.transform.read_text()) != self.manifest["sources"]["transform"]:
            raise ValueError("LOKI transform differs from frozen collection")
        self.reader = None
        self.active_group = None
        self.source_snapshots = {}
        self.anchors = {}

    def source_identity(self, group):
        """Raw bytes are immutable during an attempt; stat changes invalidate the namespace.

        Native annotation/association SHA checks still run before decoding. Raw
        image/LiDAR stat snapshots avoid rehashing TB of native data every epoch.
        """
        if group not in self.source_snapshots:
            if self.dataset == "loki":
                root = Path(self.config["loki_root"] or self.manifest["sources"]["root"])
                paths = sorted(p for _, fields in loki.frame_files(root/group) for p in fields.values() if p)
            else:
                scene = self.manifest["scenes"][group]
                native_root = Path(self.config["waymo_root"]) if self.config.get("waymo_root") else None
                paths = [road_waymo.component_path(scene, c, native_root) for c in
                    ("camera_image", "camera_box", "camera_calibration", "vehicle_pose", "lidar", "lidar_pose",
                     "lidar_calibration", "lidar_box", "camera_to_lidar_box_association")]
                paths = [p for p in paths if p is not None]
            stamps = []
            for path in paths:
                if not path.is_file():
                    raise ValueError(f"Required native source missing: {path}")
                stat = path.stat()
                stamps.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
            self.source_snapshots[group] = identity_digest(stamps)
        return self.source_snapshots[group]

    def native_reader(self, group):
        if self.reader is None:
            if self.dataset == "loki":
                self.reader = loki.TrackReader(self.config["loki_root"] or self.manifest["sources"]["root"], self.transform)
            else:
                self.reader = road_waymo.TrackReader(self.config["road_index"] or self.manifest["sources"]["index"],
                    Path(self.config["waymo_root"]) if self.config.get("waymo_root") else None)
                if self.reader.sources["sha256"] != self.manifest["sources"]["sha256"]:
                    raise ValueError("Frozen ROAD index mismatch")
        if self.active_group != group:
            actual = self.reader.index_scene(group)["provenance"]
            frozen = self.manifest["scenes"][group]
            fields = ("annotation_odometry_sha256",) if self.dataset == "loki" else ("component_sha256", "association_ids")
            if any(actual[k] != frozen[k] for k in fields):
                raise ValueError("Native scene changed from frozen collection")
            self.active_group = group
        return self.reader


class FeatureSupply:
    """Lazy per-track RGB / per-scene-frame points, shared by all experiment arms."""

    def __init__(self, catalogs, rgb_cache, point_cache, encoders, identity):
        self.catalogs, self.rgb_cache, self.point_cache = catalogs, rgb_cache, point_cache
        self.encoders, self.identity = encoders, identity
        self._point_stream = None

    def rgb(self, dataset, locator):
        catalog = self.catalogs[dataset]
        track = catalog.tracks[locator]
        group = track["group"]
        identity = {**self.identity, **catalog.identity, "kind": "rgb-cls", "locator": locator,
            "archive_sha256": track["sha256"], "source": catalog.source_identity(group),
            "encoder": DINO_REVISION, "checkpoint": self.identity.get("rgb_checkpoint_sha256", DINO_PTH_SHA256), "crop": [3,256,"bicubic-mean-padding"],
            "batch": RGB_BATCH, "precision": "fp16-cuda-autocast-float32-storage"}
        def produce():
            n = track["slots"]
            values, valid = np.zeros((n,768), np.float32), np.zeros(n, bool)
            info = [{"valid": False, "reason": "no-native-frame"} for _ in range(n)]
            slots = defaultdict(list)
            for i,k in enumerate(track["source_keys"]):
                if k is not None:
                    slots[str(k)].append(i)
            reader = catalog.native_reader(group)
            crops, indices = [], []
            seen = set()
            def flush():
                if crops:
                    values[indices] = self.encoders("rgb", torch.stack(crops))
                    valid[indices] = True
                    crops.clear(); indices.clear()
            for key,image,boxes in native_camera(reader, group, [k for k in track["source_keys"] if k is not None], [track["track_id"]], catalog.manifest["scenes"][group]):
                key = str(key)
                if key not in slots or key in seen:
                    raise ValueError("Unexpected/duplicate native camera frame")
                seen.add(key)
                pixels, detail = square_crop(image, boxes.get(track["track_id"]), expansion=3)
                for slot in slots[key]:
                    info[slot] = detail
                    if pixels is not None:
                        crops.append(normalized_crop(pixels)); indices.append(slot)
                        if len(crops) == RGB_BATCH:
                            flush()
            if seen != set(slots):
                raise ValueError("Missing native camera timestamp")
            flush()
            return {"cls": values, "valid": valid, "source_keys": np.array([str(k) if k is not None else "" for k in track["source_keys"]]),
                "times_us": np.array([int(t) if t is not None else -1 for t in track["times_us"]], np.int64),
                "crop_info": np.array([canonical(i).encode("utf-8") for i in info], dtype="S")}
        return self.rgb_cache.get(identity, produce, validate_rgb)

    def point_identity(self, dataset, locator, slot):
        catalog = self.catalogs[dataset]
        track = catalog.tracks[locator]
        if type(slot) is not int or not 0 <= slot < track["slots"]:
            raise ValueError("Point slot outside complete track")
        if not track["point_usable"][slot]:
            return None
        group, key = track["group"], str(track["source_keys"][slot])
        members = sorted(catalog.frames[group,key])
        return {**self.identity, **catalog.identity, "kind": "point-union", "group": group, "key": key,
            "source": catalog.source_identity(group), "participants": members, "radius_m": 20,
            "encoder": DITR_REVISION, "checkpoint": DITR_SHA256, "config": DITR_CONFIG_SHA256,
            "preprocessing": "full-sweep;rotated-xyz-strength;5cm-first-native;fixed-serialization;float32"}

    def point_cost(self, dataset, locator, slot):
        """None means not measured; zero means genuinely unavailable/empty, never a guessed cost."""
        identity = self.point_identity(dataset,locator,slot)
        if identity is None:
            return {"tokens":0,"inside_box_tokens":0}
        return self.point_cache.metadata({"kind":"point-cost","entry":identity_digest(identity),"locator":locator})

    def iter_points(self, dataset, locator):
        """All slots, one native traversal for ordered misses, no decoded-frame RAM cache."""
        if self._point_stream is not None:
            raise RuntimeError("Nested point supply is unsupported; use zero workers")
        catalog = self.catalogs[dataset]
        track = catalog.tracks[locator]
        native = None
        try:
            for slot in range(track["slots"]):
                identity = self.point_identity(dataset, locator, slot)
                if identity is not None and not self.point_cache.contains(identity):
                    key = str(track["source_keys"][slot])
                    if native is None or self._point_stream["next"] != key:
                        if native is not None:
                            native.close()
                        # Capacity eviction may make an earlier hit a later miss. Replan only then.
                        keys = [track["source_keys"][i] for i in range(slot,track["slots"])
                                if (ref := self.point_identity(dataset,locator,i)) is not None
                                and not self.point_cache.contains(ref)]
                        keys = sorted(set(keys),key=int)
                        members = {k for key_ in keys for k in catalog.frames[track["group"],str(key_)]}
                        reader = catalog.native_reader(track["group"])
                        native = iter(native_scene(reader,track["group"],keys,
                            [catalog.tracks[k]["track_id"] for k in sorted(members)],
                            catalog.manifest["scenes"][track["group"]],include_rgb=False))
                        self._point_stream = {"iterator":native,"keys":iter(map(str,keys)),"next":str(keys[0])}
                        next(self._point_stream["keys"])
                yield self.points(dataset,locator,slot)
        finally:
            if native is not None:
                native.close()
            self._point_stream = None

    def points(self, dataset, locator, slot):
        identity = self.point_identity(dataset,locator,slot)
        if identity is None:
            return empty_crop()
        catalog = self.catalogs[dataset]
        track = catalog.tracks[locator]
        group, key = track["group"], str(track["source_keys"][slot])
        members = sorted(catalog.frames[group,key])
        def produce():
            reader = catalog.native_reader(group)
            tracks = [catalog.tracks[k] for k in members]
            first_keys = {t["source_keys"][0] for t in tracks if t["locator"] not in catalog.anchors}
            if None in first_keys:
                raise ValueError("Usable point track has no initial native anchor")
            records = reader.read_frames(group, first_keys) if first_keys else {}
            for t in tracks:
                if t["locator"] not in catalog.anchors:
                    catalog.anchors[t["locator"]] = records[t["source_keys"][0]]["ego_pose"]
            anchors = {t["locator"]: catalog.anchors[t["locator"]] for t in tracks}
            if self._point_stream is not None and self._point_stream["next"] == key:
                frame = next(self._point_stream["iterator"])
                self._point_stream["next"] = next(self._point_stream["keys"],None)
            else:
                frame, = native_scene(reader, group, [track["source_keys"][slot]], [t["track_id"] for t in tracks],
                                      catalog.manifest["scenes"][group], include_rgb=False)
            if str(frame["key"]) != key:
                raise ValueError("Native point timestamp mismatch")
            boxes = {}
            for t in tracks:
                box = frame["boxes"].get(t["track_id"])
                if box is None:
                    raise ValueError("Frozen usable 3D observation lost its native cuboid")
                boxes[t["locator"]] = box
            points = frame["points"]
            if points is None or not len(points):
                points = np.zeros((0,4), np.float32)
                features, selected = np.zeros((0,64), np.float32), np.zeros(0, np.int64)
            else:
                features, _, selected, _ = self.encoders("points", points)
            return point_union(points, features, selected, boxes, frame["world_from_native"], anchors)
        arrays = self.point_cache.get(identity, produce, validate_points)
        crop = point_crop(arrays, locator)
        self.point_cache.record_metadata({"kind":"point-cost","entry":identity_digest(identity),"locator":locator},
            {"tokens":len(crop["features"]),"inside_box_tokens":int(crop["inside_box"].sum())})
        return crop


def load_configuration(probe):
    probe = Path(probe).resolve()
    record = json.loads((probe/"probe.json").read_text())
    if record["status"] != "complete":
        raise ValueError("Completed pinned native probe required")
    config = record["config"]
    if checksum(config["ditr_config"]) != DITR_CONFIG_SHA256 or checksum(config["ditr_weights"]) != DITR_SHA256:
        raise ValueError("Changed point encoder assets")
    expected = DINO_HF_SHA256 if Path(config["dino_weights"]).suffix == ".safetensors" else DINO_PTH_SHA256
    if checksum(config["dino_weights"]) != expected:
        raise ValueError("Changed RGB encoder assets")
    config = dict(config, rgb_checkpoint_sha256=expected)
    return config, probe/"loki-transform.json"


def selected_tracks(catalogs, selection):
    if selection is None:
        return [(d,k) for d,c in catalogs.items() for k in c.tracks]
    record = json.loads(Path(selection).read_text())
    result = [(t["dataset"],t["locator"]) for t in record["tracks"] if t["dataset"] in catalogs]
    if not result or len(set(result)) != len(result):
        raise ValueError("Empty/duplicate cache selection")
    for d,k in result:
        if k not in catalogs[d].tracks:
            raise ValueError("Selection outside frozen source train/validation")
    return result


def validate_roots(args, config):
    rgb, points, output = (p.resolve() for p in (args.rgb_cache_root,args.point_cache_root,args.output))
    root = args.outputs_root.resolve()
    if not output.is_relative_to(root/"experiments/E002a"):
        raise ValueError("Evidence must be under outputs-root/experiments/E002a")
    locations = (rgb,points,output)
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(locations) for b in locations[i+1:]):
        raise ValueError("RGB cache, point cache and evidence must have separate non-overlapping roots")
    protected = [root/"experiments"/name for name in ("E001","E001b","E001c")]
    protected += [Path(config[k]).resolve() for k in ("loki_root","waymo_root","road_index") if config.get(k)]
    for location in locations:
        if any(location.is_relative_to(p) or p.is_relative_to(location) for p in protected):
            raise ValueError("Cache/evidence must not overlap native data or established E001 artifacts")


def run(args):
    validate_roots(args,{})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "command": sys.orig_argv, "revision": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "diff": subprocess.check_output(["git","diff","HEAD"],text=True), "tracks": [], "failures": []}
    write_json(output/"cache.json", record)
    protected = {}
    begin = time.perf_counter()
    encoders = None
    try:
        config, transform = load_configuration(args.probe)
        validate_roots(args,config)
        datasets = ("loki","road-waymo") if args.dataset == "both" else (args.dataset,)
        catalogs = {d: FeatureCatalog(args.outputs_root,d,config,transform) for d in datasets}
        for catalog in catalogs.values():
            protected.update(catalog.protected)
        for path in (args.probe/"probe.json", transform, Path(config["dino_weights"]), Path(config["ditr_weights"]), Path(config["ditr_config"])):
            protected[str(path.resolve())] = checksum(path)
        selected = selected_tracks(catalogs,args.selection)
        identity = extraction_identity(args.device,config)
        record.update(extraction=identity, population={d:c.identity for d,c in catalogs.items()},
            host=platform.node(), selection=str(args.selection) if args.selection else "full-frozen-source-train-validation")
        encoders = FrozenEncoders(config,args.device)
        with FeatureCache(args.rgb_cache_root, min_free_bytes=gb_bytes(args.min_free_gb)) as rgb, \
             FeatureCache(args.point_cache_root, capacity_bytes=gb_bytes(args.point_cache_gb), min_free_bytes=gb_bytes(args.min_free_gb)) as points:
            record["initial"] = {"rgb":rgb.initial,"points":points.initial}
            supply = FeatureSupply(catalogs,rgb,points,encoders,identity)
            for d,k in selected:
                t = catalogs[d].tracks[k]
                row = {"dataset":d,"locator":k,"slots":t["slots"],"split":t["split"]}
                if args.kind in ("rgb","both"):
                    values = supply.rgb(d,k)
                    row["rgb_valid"] = int(values["valid"].sum())
                if args.kind in ("points","both"):
                    counts = [len(crop["features"]) for crop in supply.iter_points(d,k)]
                    row["point_counts"] = counts
                record["tracks"].append(row)
                record["cache"] = {"rgb":rgb.snapshot(),"points":points.snapshot()}
                write_json(output/"cache.json",record)
                print(d,len(record["tracks"]),"/",len(selected),args.kind,flush=True)
            record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"; record["failures"].append(traceback.format_exc()); raise
    finally:
        if encoders is not None:
            encoders.release()
            record["encoders"] = {"loads":encoders.loads,"calls":encoders.calls,"seconds":encoders.seconds}
        changed = [p for p,h in protected.items() if not Path(p).is_file() or checksum(p) != h]
        record.update(protected_files=len(protected),changed_files=changed,elapsed_seconds=time.perf_counter()-begin,
            finished_utc=datetime.now(timezone.utc).isoformat())
        if changed:
            record["status"] = "failed"
        write_json(output/"protected.json",protected); write_json(output/"cache.json",record)
        if changed:
            raise ValueError("Protected input changed")


def add_arguments(parser):
    parser.add_argument("--outputs-root",type=Path,default=Path("outputs"))
    parser.add_argument("--probe",type=Path,default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    parser.add_argument("--dataset",choices=("loki","road-waymo","both"),default="both")
    parser.add_argument("--rgb-cache-root",type=Path,default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--point-cache-root",type=Path,default=Path("outputs/experiments/E002a/point-cache"))
    parser.add_argument("--point-cache-gb",type=float,default=450)
    parser.add_argument("--min-free-gb",type=float,default=20)
    parser.add_argument("--selection",type=Path,help="Explicit bounded track record; omitted means full frozen train/validation")
    parser.add_argument("--output",type=Path,required=True,help="Fresh evidence directory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--kind",choices=("rgb","points","both"),default="rgb")
    run(parser.parse_args())
