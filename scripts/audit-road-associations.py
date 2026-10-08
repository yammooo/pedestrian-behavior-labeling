"""Recheck the saved ROAD population against native official links and freeze review views."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from html import escape
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from pedestrian_behavior.data.preparation import checksum, load_manifest, load_track, source_times
from pedestrian_behavior.datasets import road_waymo as road
from pedestrian_behavior.inspection.prepared_tracks import native_previews


def audit(reader, collection, frozen_cases):
    collection = Path(collection)
    manifest = load_manifest(collection)
    if manifest["dataset"] != "road-waymo" or reader.sources["sha256"] != manifest["sources"]["sha256"]:
        raise ValueError("Native index/saved collection source mismatch")
    archives = {}
    for filename in manifest["tracks"]:
        metadata, arrays = load_track(collection / "tracks" / filename)
        speed = np.linalg.norm(arrays["ped_velocity"][arrays["ped_velocity_valid"]], axis=1)
        archives[metadata["track_locator"]] = {"archive": filename, "metadata": metadata,
            "maximum_selected_speed_mps": float(speed.max()) if len(speed) else 0.,
            "has_position": bool(arrays["ped_position_valid"].any()),
            "position_times": [source_times(metadata)[i] for i in np.flatnonzero(arrays["ped_position_valid"])]}
    if len(archives) != len(manifest["tracks"]):
        raise ValueError("Repeated saved identity")
    counts, tracks, action_rows = Counter(), [], Counter()
    for scene_id in sorted(manifest["scenes"]):
        index = reader.index_scene(scene_id)
        if index["provenance"]["component_sha256"] != manifest["scenes"][scene_id]["component_sha256"]:
            raise ValueError(f"{scene_id}: native component checksum mismatch")
        camera, lidar = defaultdict(set), defaultdict(set)
        for obj, ts in reader.camera:
            camera[obj].add(ts)
        for obj, ts in reader.lidar:
            lidar[obj].add(ts)
        frames = {f["time_us"]: f for f in index["frames"]}
        # Counts describe native identities, not a new association or eligibility policy.
        for tube, candidate in sorted(index["tracks"].items()):
            address = candidate["locator"]
            if address not in archives:
                raise ValueError("Native candidate missing from saved collection")
            saved = archives[address]["metadata"]
            if candidate["counts"] != saved["native_counts"] or candidate["extent_us"] != saved["native_extent_us"]:
                raise ValueError("Native/saved inventory or extent mismatch")
            if candidate["laser_id"] != manifest["scenes"][scene_id]["association_ids"][tube]:
                raise ValueError("Native/saved official association mismatch")
            observations = reader.road[scene_id][tube]
            times = sorted(observations)
            laser = candidate["laser_id"]
            ct, lt = camera[tube], lidar[laser] if laser else set()
            pairs = {t for t, o in observations.items() if o["flags"]["has_3d_box"]}
            if not pairs <= lt:
                raise ValueError("Export pair missing from native linked 3D")
            if any(frames[t]["rgb"] is not True for t in times):
                raise ValueError("ROAD observation lacks native FRONT image")
            for ts in pairs:
                if int(observations[ts]["flags"]["export_3d_type"]) != reader.lidar[laser, ts]["[LiDARBoxComponent].type"]:
                    raise ValueError("Export/native 3D type mismatch")
            disagreements = [ts for ts, o in observations.items() if o["flags"]["semantic_disagreement"]]
            counts.update({"native_candidates": 1, "with_unique_lidar_id": laser is not None,
                "without_lidar_id": laser is None, "with_native_3d": bool(lt), "native_front_identity_missing": not ct,
                "native_union_observations": candidate["counts"]["observations"], "native_3d_observations": len(lt),
                "native_front_observations": len(ct), "unique_road_observations": len(times),
                "duplicate_annotation_rows": candidate["counts"]["duplicate_annotations"],
                "export_pairs": len(pairs), "additional_3d_timestamps": len(lt-pairs),
                "additional_front_timestamps": len(ct-set(times)), "gain_3d_without_export_pair": bool(lt) and not pairs,
                "3d_before_road": bool(lt) and min(lt) < times[0], "3d_after_road": bool(lt) and max(lt) > times[-1],
                "front_before_road": bool(ct) and min(ct) < times[0], "front_after_road": bool(ct) and max(ct) > times[-1],
                "disagreement_tracks": bool(disagreements), "unique_disagreement_observations": len(disagreements)})
            for ts, o in observations.items():
                action_rows[scene_id, ts] += 1
            tracks.append({"locator": address, "clip": scene_id, "tube_uid": tube, "laser_id": laser,
                "archive": archives[address]["archive"], "native_road_times": times, "disagreement_times": disagreements,
                "native_front_times": sorted(ct), "native_3d_times": sorted(lt),
                "has_position": archives[address]["has_position"],
                "maximum_selected_speed_mps": archives[address]["maximum_selected_speed_mps"],
                "position_times": archives[address]["position_times"]})
        print(f"{scene_id}: native links verified", flush=True)
    if len(tracks) != len(archives):
        raise ValueError("Saved candidate absent from native audit")
    rows, box_areas = Counter(), defaultdict(list)
    for row in road.annotation_rows(reader.index):
        rows[row["association_status"]] += 1
        rows["semantic_disagreement_rows"] += row["semantic_disagreement"] == "True"
        box = json.loads(row["road_box_normalized_json"])
        box_areas[row["road_clip_id"], row["road_tube_uid"]].append((box[2]-box[0])*(box[3]-box[1]))
    for track in tracks:
        track["median_road_box_area"] = float(np.median(box_areas[track["clip"], track["tube_uid"]]))
        track["maximum_same_frame_road_pedestrians"] = max(action_rows[track["clip"], t] for t in track["native_road_times"])
    selected = {case["locator"]: ["frozen native case: " + case["reason"]] for case in frozen_cases}
    for track in tracks:
        if track["disagreement_times"]:
            selected.setdefault(track["locator"], []).append("all Pedestrian/Cyclist disagreement tracks")
    usable = [t for t in tracks if t["has_position"]]
    for name, ordered in (
        ("largest selected velocity", sorted(usable, key=lambda t: (-t["maximum_selected_speed_mps"], t["locator"]))),
        ("smallest median ROAD box", sorted(usable, key=lambda t: (t["median_road_box_area"], t["locator"]))),
        ("most same-frame ROAD pedestrians", sorted(usable, key=lambda t: (-t["maximum_same_frame_road_pedestrians"], t["locator"]))),
    ):
        clips = set()
        for track in ordered:
            if track["clip"] in clips:
                continue
            selected.setdefault(track["locator"], []).append(name + " (three distinct clips)")
            clips.add(track["clip"])
            if len(clips) == 3:
                break
    cases = []
    for track in tracks:
        if track["locator"] not in selected:
            continue
        snapshots = set()
        for times in (track["native_road_times"], track["disagreement_times"], track["position_times"]):
            if times:
                snapshots.update((times[0], times[len(times)//2], times[-1]))
        extensions = sorted(set(track["native_3d_times"]) - set(track["native_road_times"]))
        if extensions:
            snapshots.update((extensions[0], extensions[-1]))
        cases.append(track | {"reasons": selected[track["locator"]], "snapshot_times": sorted(snapshots)})
    return {"status": "structural-pass", "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.orig_argv, "collection_manifest_sha256": checksum(collection/"manifest.json"),
        "source_sha256": reader.sources["sha256"], "native_component_checksums": "all matched saved manifest",
        "counts": dict(counts), "csv_rows": dict(rows), "cases": cases,
        "selection": "Eight frozen native cases; all disagreement tracks; top three distinct clips each by maximum selected speed, minimum median ROAD box area, maximum same-frame ROAD pedestrian count. Ties: locator.",
        "limitations": "Purposive visual sample, not an estimated population association error rate. Occlusion is assessed visually, not inferred from missing boxes."}


def render(reader, report, output):
    selected = [(case["clip"], case["tube_uid"]) for case in report["cases"]]
    observations = road.read_tracks(reader.index, selected)
    for number, case in enumerate(report["cases"]):
        folder = output/f"case-{number:02}"
        metadata = {"track_locator": case["locator"], "source_frames": case["snapshot_times"]}
        case["previews"] = native_previews(reader, metadata, folder)
        case["crops"] = []
        scene = reader.scenes[case["clip"]]
        crop_paths = {}
        for row in road.component_rows(road.component_path(scene, "camera_image", reader.waymo_root), scene["segment_context_name"]):
            ts = row["key.frame_timestamp_micros"]
            if row["key.camera_name"] != 1 or ts not in case["snapshot_times"]:
                continue
            observation = observations[case["clip"], case["tube_uid"]].get(ts)
            if observation is None:
                continue
            with Image.open(BytesIO(row["[CameraImageComponent].image"])) as raw:
                x1,y1,x2,y2 = observation["box2d"]
                margin = .01
                bounds = (max(0,int((x1-margin)*raw.width)),max(0,int((y1-margin)*raw.height)),
                          min(raw.width,int((x2+margin)*raw.width)+1),min(raw.height,int((y2+margin)*raw.height)+1))
                crop = raw.crop(bounds).convert("RGB")
                draw = ImageDraw.Draw(crop)
                draw.rectangle((x1*raw.width-bounds[0],y1*raw.height-bounds[1],x2*raw.width-bounds[0],y2*raw.height-bounds[1]), outline="yellow", width=2)
                crop.save(folder/f"crop-{ts}.jpg")
                crop_paths[ts] = f"case-{number:02}/crop-{ts}.jpg"
        case["crops"] = [crop_paths.get(ts) for ts in case["snapshot_times"]]
        print(f"review {number}: {case['locator']}; {len(case['previews'])} snapshots",flush=True)
    rows=[]
    for number,case in enumerate(report["cases"]):
        frames=[]
        for ts,preview,crop in zip(case["snapshot_times"],case["previews"],case["crops"]):
            frames.append(f'<figure><figcaption>{ts}</figcaption><img width="622" src="{preview}" alt="Native RGB and same-frame BEV">'
                          +(f'<img style="max-width:320px;image-rendering:auto" src="{crop}" alt="Original-resolution ROAD box crop">' if crop else '<p>Native extension; no ROAD box/GT</p>')+'</figure>')
        rows.append(f'<h2>{number}: {escape(case["clip"])} / {escape(case["tube_uid"])}</h2><p>{escape("; ".join(case["reasons"]))}</p><div style="display:flex;flex-wrap:wrap">'+''.join(frames)+'</div>')
    (output/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>ROAD association review</title>'
        '<style>body{font:16px system-ui;margin:2rem}figure{margin:.4rem;max-width:622px}img{max-width:100%}</style>'
        '<h1>ROAD association review</h1><p>Yellow: original ROAD box/native linked 3D footprint. Crops retain original pixels; no inferred association or GT.</p>'+''.join(rows)+'</html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",type=Path,required=True)
    parser.add_argument("--collection",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    reader=road.TrackReader(args.index)
    cases=json.loads(Path("experiments/E001-kinematic-transfer/reader-cases.json").read_text())["road-waymo"]
    try:
        report=audit(reader,args.collection,cases)
        (args.output/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
        render(reader,report,args.output)
        (args.output/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    except Exception as error:
        (args.output/"failure.json").write_text(json.dumps({"status":"failed","error":str(error)},indent=2)+"\n")
        raise


if __name__=="__main__":
    main()
