"""Matched frozen D-DITR/Utonia PCA videos on existing development tracks."""

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import html
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum, load_track, source_times, valid_pose
from pedestrian_behavior.datasets import loki, road_waymo
from pedestrian_behavior.inspection.feature_probe import relative_xyz
from pedestrian_behavior.inspection.feature_video import PCA_VIDEO_SIZE, render_pca_frame
from pedestrian_behavior.inspection.native_features import native_scene
from pedestrian_behavior.inspection.point_pca import calibration_frames, fit_pca, pca_colors
from pedestrian_behavior.inspection.pretrained import (PointEncoder, UtoniaEncoder,
    DITR_REVISION, DITR_SHA256, UTONIA_REVISION, UTONIA_SHA256)
from .e002a_probe import write_json


def gallery(output, record):
    lines = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen LiDAR feature comparison</title>',
        '<style>body{font:16px system-ui;background:#171b22;color:white;margin:24px}video,img{width:100%;max-width:2048px}a{color:#8cf}article{margin:35px 0}</style>',
        '<h1>D-DITR versus Utonia: fixed feature PCA</h1>',
        f'<p>Status: {html.escape(record["status"])}. Same ten development tracks and native points; complete 5 Hz grids. '
        'RGB 3x | D-DITR | Utonia, with target boxes and 15/20/30 m rings.</p>',
        '<p>One independent PCA per encoder, fitted once on equal training-frame/point counts from both datasets. '
        'Colors remain fixed across videos/datasets; they have no semantic class meaning and do not correspond between models. '
        'Display colors are inverse-mapped from encoder grids to native points; this does not restore detail lost by downsampling.</p>',
        '<p>D-DITR: 64 decoded channels, XYZ/strength, nuScenes semantic fine-tuning. Utonia: official outdoor '
        'last-three-level features (1224 channels), XYZ with absent color/normals, 0.2 coordinate scale. '
        'Both use effective 5 cm input grids, with deterministic first-return selection and fixed serialization. '
        'Different pretraining, feature levels and input channels: this is not a controlled architecture ablation.</p>',
        '<p>Utonia pretraining includes Waymo train/validation. Pooled training-only visual calibration also uses both datasets. '
        'This is a qualitative development diagnostic, not strict zero-shot evaluation or a behavior/segmentation score.</p>',
        '<p><a href="comparison.json">Provenance and numerical checks</a> · <a href="pca.npz">Fixed PCA bases/bounds</a></p>']
    for item in record["tracks"]:
        name = html.escape(item["name"])
        lines.append(f'<article><h2>{name}</h2><p>{html.escape(item["locator"])} | {item["split"]} | {item["slots"]} slots</p>')
        if item["status"] == "complete":
            lines.append(f'<video controls preload="metadata" poster="{name}-middle.png" src="{name}.mp4"></video>'
                         f'<p><a href="{name}.mp4">MP4</a> · <a href="{name}-first.png">First</a> · '
                         f'<a href="{name}-middle.png">Middle</a> · <a href="{name}-last.png">Last</a></p>')
        lines.append('</article>')
    (output/"index.html").write_text("\n".join(lines)+"</html>\n")


def context(dataset, root, config, probe, protected):
    collection = root/"experiments/E001/reader-preparation"/dataset
    setup_path = root/"experiments/E001/data-setup"/dataset/"setup.json"
    manifest = json.loads((collection/"manifest.json").read_text())
    setup = json.loads(setup_path.read_text())
    if setup["collection"]["manifest_sha256"] != checksum(collection/"manifest.json"):
        raise ValueError("Frozen collection/setup mismatch")
    for path in (collection/"manifest.json", setup_path):
        protected[str(path)] = checksum(path)
    if dataset == "loki":
        reader = loki.TrackReader(config["loki_root"] or manifest["sources"]["root"], probe/"loki-transform.json")
    else:
        reader = road_waymo.TrackReader(config["road_index"] or manifest["sources"]["index"],
            Path(config["waymo_root"]) if config["waymo_root"] else None)
        if reader.sources["sha256"] != manifest["sources"]["sha256"]:
            raise ValueError("Frozen ROAD index mismatch")
    return collection, manifest, setup, reader


def extract(encoder, points, indices, pca, device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = encoder.features(points, indices, None if pca is None else lambda x: pca_colors(x, pca))
    if values.ndim != 2 or len(values) != len(indices) or not np.isfinite(values).all():
        raise ValueError("Invalid matched point features/colors")
    return values, {"seconds": time.perf_counter()-started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}


def run(args):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe, videos, output = args.probe.resolve(), args.videos.resolve(), args.output.resolve()
    root = probe.parents[2]
    if not output.is_relative_to(root/"experiments/E002a"):
        raise ValueError("Output must be under outputs/experiments/E002a")
    source = json.loads((probe/"probe.json").read_text())
    previous = json.loads((videos/"videos.json").read_text())
    if source["status"] != "complete" or previous["status"] != "complete" or len(previous["tracks"]) != 10:
        raise ValueError("A completed probe and ten-track video record are required")
    output.mkdir(parents=True, exist_ok=False)
    (output/"calibration").mkdir()
    (output/"frames").mkdir()
    config = source["config"]
    protected = source["protected"].copy()
    for path in [probe/"probe.json", Path(config["ditr_weights"]), Path(config["ditr_config"]),
                 args.utonia_weights.resolve(), *videos.glob("*")]:
        if path.is_file():
            protected[str(path)] = checksum(path)
    record = {"status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.orig_argv, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device,
        "dependencies": sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()),
        "seed": 0, "probe_sha256": checksum(probe/"probe.json"), "videos_sha256": checksum(videos/"videos.json"),
        "encoders": {"dditr": {"code": DITR_REVISION, "sha256": DITR_SHA256, "channels": 64},
                     "utonia": {"code": UTONIA_REVISION, "sha256": UTONIA_SHA256, "channels": 1224}},
        "pca_policy": "one basis/encoder; 4 training scenes/dataset, 2 frames/scene, 1024 identical native points/frame; pooled across datasets; fixed 1st/99th percentile bounds; no per-frame fitting",
        "display_policy": "same native XYZ rows within target-centered 30 m world-horizontal radius; inverse grid/hierarchy mapping; 15/20/30 m rings; initial ego heading; no height cut or temporal accumulation",
        "measurement_policy": "seconds include grid preparation, inference, feature gathering/projection and CPU transfer; exclude scene I/O and artifact encoding; CUDA peaks include both resident encoders; no throughput or architecture speed ranking",
        "preprocessing": {"dditr": "unchanged 5 cm rotated XYZ+strength recipe",
            "utonia": "official outdoor .2 scale, .01 scaled grid, native ego axes, no center shift; absent color and normals 0 (official NormalizeColor divides by 255); last 3 hierarchy levels as official outdoor PCA demo; first native return/grid and all shuffle_orders False"},
        "calibration": [], "tracks": [], "failures": []}
    started = time.monotonic()
    write_json(output/"comparison.json", record)
    try:
        for path, digest in protected.items():
            if checksum(path) != digest:
                raise ValueError(f"Protected source changed: {path}")
        encoders = {"dditr": PointEncoder(config["ditr_code"], config["ditr_config"], config["ditr_weights"], args.device),
                    "utonia": UtoniaEncoder(args.utonia_code, args.utonia_weights, args.device)}
        record["hardware"] = torch.cuda.get_device_name() if args.device == "cuda" else "CPU"
        contexts = {d: context(d, root, config, probe, protected) for d in ("loki", "road-waymo")}
        rows = {name: [] for name in encoders}
        for dataset, (collection, manifest, setup, reader) in contexts.items():
            samples = calibration_frames(source, dataset)
            for group in dict.fromkeys(g for g, _ in samples):
                if setup["groups"][group] != "training":
                    raise ValueError("Calibration scene is not training")
                identities = [json.loads(e["locator"]) for e in setup["tracks"]
                              if e["group"] == group and not e["exclusions"]]
                tracks = [i.get("track_id", i.get("tube_uid")) for i in identities]
                reader.index_scene(group)
                keys = [k for g, k in samples if g == group]
                for frame in native_scene(reader, group, keys, tracks, manifest["scenes"][group]):
                    points = frame["points"]
                    if points is None or not valid_pose(frame["world_from_native"]):
                        raise ValueError("Calibration frame lacks scan/pose")
                    near = np.zeros(len(points), dtype=bool)
                    for box in frame["boxes"].values():
                        offsets = relative_xyz(points[:, :3], box[0], frame["world_from_native"], np.eye(4))
                        near |= np.linalg.norm(offsets[:, :2], axis=1) <= 30
                    available = np.flatnonzero(near)
                    if len(available) < 1024:
                        raise ValueError("Insufficient native calibration points near tracks")
                    indices = np.sort(np.random.default_rng(0).choice(available, 1024, replace=False))
                    item = {"dataset": dataset, "group": group, "key": frame["key"], "split": "training", "points": 1024}
                    fixture = {"native_indices": indices, "xyz": points[indices, :3]}
                    for name, encoder in encoders.items():
                        features, evidence = extract(encoder, points, indices, None, args.device)
                        fixture[name] = features
                        rows[name].append(features)
                        item[name] = evidence
                        if len(record["calibration"]) in (0, 8):
                            repeated, _ = extract(encoder, points, indices, None, args.device)
                            item[name]["repeat_max_absolute_error"] = float(np.abs(features-repeated).max())
                    filename = f"{dataset}-{group}-{frame['key']}.npz"
                    np.savez_compressed(output/"calibration"/filename, **fixture)
                    item["fixture"] = "calibration/"+filename
                    item["sha256"] = checksum(output/item["fixture"])
                    record["calibration"].append(item)
                    print("Calibration", dataset, group, frame["key"], flush=True)
        pcas = {name: fit_pca(np.concatenate(values)) for name, values in rows.items()}
        np.savez(output/"pca.npz", **{name+"_"+key: value for name, pca in pcas.items() for key, value in pca.items()})
        record["pca"] = {name: {"fit_rows": sum(len(x) for x in rows[name]),
            "explained_variance_ratio": pca["explained_variance_ratio"].tolist()} for name, pca in pcas.items()}
        del rows
        write_json(output/"comparison.json", record)
        print("Fixed PCA fitted", record["pca"], flush=True)
        for original in previous["tracks"]:
            dataset = "loki" if original["name"].startswith("loki-") else "road-waymo"
            collection, manifest, setup, reader = contexts[dataset]
            matches = [e for e in setup["tracks"] if e["archive"] == original["archive"] and not e["exclusions"]]
            if len(matches) != 1 or matches[0]["split"] not in ("training", "validation") or any(matches[0][k] != original[k] for k in ("locator", "group", "split")):
                raise ValueError("Video selection differs from frozen development split")
            path = collection/"tracks"/original["archive"]
            protected.setdefault(str(path), checksum(path))
            metadata, _ = load_track(path)
            if metadata["track_locator"] != original["locator"]:
                raise ValueError("Video archive identity changed")
            keys = metadata["source_frames"]
            if keys != [f["key"] for f in original["frames"]] or len(keys) != original["slots"]:
                raise ValueError("Video/native grid differs from previous video")
            times = source_times(metadata)
            if times != [f["source_time_us"] for f in original["frames"]]:
                raise ValueError("Native grid times changed")
            group = original["group"]
            identity = json.loads(original["locator"])
            track = identity.get("track_id", identity.get("tube_uid"))
            anchor = np.asarray(original["anchor_ego_pose"])
            item = {k: original[k] for k in ("name", "locator", "archive", "group", "split", "slots")}
            item.update(status="incomplete", frames=[])
            record["tracks"].append(item)
            reader.index_scene(group)
            stream = iter(native_scene(reader, group, [k for k in keys if k is not None], [track], manifest["scenes"][group]))
            print("Rendering", item["name"], "slots", len(keys), flush=True)
            with (output/(item["name"]+"-ffmpeg.log")).open("wb") as log:
                process = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", f"{PCA_VIDEO_SIZE[0]}x{PCA_VIDEO_SIZE[1]}", "-framerate", "5",
                    "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output/(item["name"]+".mp4"))], stdin=subprocess.PIPE, stderr=log)
                try:
                    for slot, key in enumerate(keys):
                        frame = next(stream) if key is not None else None
                        if frame and frame["key"] != key:
                            raise ValueError("PCA video/native synchronization mismatch")
                        offsets, colors, evidence = None, {}, {}
                        box = frame["boxes"].get(track) if frame else None
                        if box is not None and frame["points"] is not None and len(frame["points"]) and valid_pose(frame["world_from_native"]):
                            all_offsets = relative_xyz(frame["points"][:, :3], box[0], frame["world_from_native"], anchor)
                            indices = np.flatnonzero(np.linalg.norm(all_offsets[:, :2], axis=1) <= 30)
                            offsets = all_offsets[indices]
                            fixture = {"native_indices": indices, "offsets": offsets}
                            for name, encoder in encoders.items():
                                colors[name], evidence[name] = extract(encoder, frame["points"], indices, pcas[name], args.device)
                                fixture[name+"_colors"] = colors[name]
                            filename = f"{item['name']}-{slot:03d}.npz"
                            np.savez_compressed(output/"frames"/filename, **fixture)
                            evidence["fixture"] = "frames/"+filename
                            evidence["points"] = len(indices)
                        heading = f"{item['name']} | {group} | t={slot/5:.1f}s | slot {slot+1}/{len(keys)} | {key}"
                        image, panel = render_pca_frame(frame, track, anchor, offsets, colors, heading)
                        evidence.update(panel, slot=slot, key=key, source_time_us=times[slot])
                        item["frames"].append(evidence)
                        process.stdin.write(image.tobytes())
                        for at, label in ((0, "first"), (len(keys)//2, "middle"), (len(keys)-1, "last")):
                            if slot == at:
                                image.save(output/(item["name"]+"-"+label+".png"))
                finally:
                    with suppress(BrokenPipeError):
                        process.stdin.close()
                    if process.wait() != 0:
                        raise RuntimeError(f"ffmpeg failed; see {item['name']}-ffmpeg.log")
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", str(output/(item["name"]+".mp4"))], text=True))["streams"][0]
            if int(info["nb_read_frames"]) != len(keys) or info["r_frame_rate"] != "5/1" or (info["width"], info["height"]) != PCA_VIDEO_SIZE:
                raise ValueError("Encoded PCA video lost grid frames or changed dimensions/rate")
            item.update(status="complete", video=info)
            write_json(output/"comparison.json", record)
            gallery(output, record)
            print("Completed", item["name"], flush=True)
        record["status"] = "complete"
    except Exception:
        record["status"] = "failed"
        record["failures"].append(traceback.format_exc())
        raise
    finally:
        changed = [p for p, digest in protected.items() if checksum(p) != digest]
        record.update(protected=protected, protected_files=len(protected), changed_files=changed, seconds=time.monotonic()-started,
            ended_utc=datetime.now(timezone.utc).isoformat())
        if changed:
            record["status"] = "failed"
        write_json(output/"comparison.json", record)
        gallery(output, record)
        if changed:
            raise ValueError("Protected source files changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "videos", "utonia-code", "utonia-weights", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
