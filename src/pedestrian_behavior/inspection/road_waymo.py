"""Whole-scene ROAD-Waymo pedestrian RGB + LiDAR BEV galleries."""

import argparse
from contextlib import suppress
from io import BytesIO
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from pedestrian_behavior.datasets import road_waymo as road
from pedestrian_behavior.inspection.__main__ import select
from pedestrian_behavior.inspection.render import BEV_SIZE, BEV_SPAN, RGB_SIZE, bev_pixel, footprint, write_index


def aligned_frames(groups, timestamps):
    current = next(groups, None)
    for ts in timestamps:
        while current is not None and current[0] < ts:
            current = next(groups, None)
        if current is not None and current[0] == ts:
            yield current[1]
            current = next(groups, None)
        else:
            yield []


def render_track(scene: dict, track_id: str, observations: dict, output: Path, waymo_root=None):
    name = scene["segment_context_name"]
    def rows(component):
        return road.component_rows(road.component_path(scene, component, waymo_root), name)

    images = [r for r in rows("camera_image") if r["key.camera_name"] == 1]
    timestamps = [r["key.frame_timestamp_micros"] for r in images]
    if not images or len(set(timestamps)) != len(timestamps):
        raise ValueError(f"{scene['road_clip_id']}: missing or duplicate FRONT frames")
    if not set(observations).issubset(timestamps):
        raise ValueError("ROAD timestamps absent from original FRONT images")
    poses = {}
    for r in rows("vehicle_pose"):
        ts = r["key.frame_timestamp_micros"]
        if ts in poses:
            raise ValueError("Duplicate vehicle pose")
        poses[ts] = np.asarray(r["[VehiclePoseComponent].world_from_vehicle.transform"]).reshape(4, 4)
    if any(ts not in poses for ts in timestamps):
        raise ValueError("Missing vehicle poses; cannot align TOP LiDAR or view focus")
    calibrations = {}
    for r in rows("lidar_calibration"):
        laser = r["key.laser_name"]
        if laser in calibrations:
            raise ValueError("Duplicate LiDAR calibration")
        calibrations[laser] = r
    centers = road.focus_centers(timestamps, observations, poses)
    has_focus = any(o["box3d"] is not None for o in observations.values())
    clouds = aligned_frames(road.frame_groups(rows("lidar")), timestamps)
    pixel_poses = aligned_frames(road.frame_groups(rows("lidar_pose")), timestamps)
    process = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", "1564x604", "-framerate", "5", "-i", "-", "-an",
         "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", str(output)], stdin=subprocess.PIPE,
    )
    try:
        for n, (raw, center, cloud, pixel_rows) in enumerate(zip(images, centers, clouds, pixel_poses), 1):
            ts = raw["key.frame_timestamp_micros"]
            observation = observations.get(ts)
            with Image.open(BytesIO(raw["[CameraImageComponent].image"])) as source:
                fitted = ImageOps.contain(source.convert("RGB"), RGB_SIZE)
            rgb = Image.new("RGB", RGB_SIZE, "black")
            offset = ((RGB_SIZE[0] - fitted.width) // 2, (RGB_SIZE[1] - fitted.height) // 2)
            rgb.paste(fitted, offset)
            draw = ImageDraw.Draw(rgb)
            if observation is not None:
                x1, y1, x2, y2 = observation["box2d"]
                draw.rectangle((offset[0] + x1 * fitted.width, offset[1] + y1 * fitted.height,
                                offset[0] + x2 * fitted.width, offset[1] + y2 * fitted.height),
                               outline="yellow", width=3)
            bev = Image.new("RGB", (BEV_SIZE, BEV_SIZE), "#101010")
            bev_draw = ImageDraw.Draw(bev)
            pixel_by_sensor = {r["key.laser_name"]: r for r in pixel_rows}
            if len(pixel_by_sensor) != len(pixel_rows):
                raise ValueError("Duplicate LiDAR pixel pose")
            seen = set()
            warnings = []
            visible_points = 0
            for sensor in cloud:
                laser = sensor["key.laser_name"]
                if laser in seen:
                    raise ValueError("Duplicate LiDAR sensor/frame")
                seen.add(laser)
                if laser not in calibrations:
                    warnings.append(f"missing sensor {laser} calibration")
                    continue
                top_pose = pixel_by_sensor.get(laser)
                if laser == 1 and top_pose is None:
                    warnings.append("missing TOP motion pose")
                    continue
                for return_number in (1, 2):
                    prefix = f"[LiDARComponent].range_image_return{return_number}."
                    if sensor[prefix + "values"] is None:
                        warnings.append(f"missing sensor {laser} return {return_number}")
                        continue
                    pixel = None
                    if laser == 1:
                        p = "[LiDARPoseComponent].range_image_return1."
                        shape = sensor[prefix + "shape"]
                        if top_pose[p + "shape"] != [shape[0], shape[1], 6]:
                            raise ValueError("TOP pixel pose and range image shapes differ")
                        pixel = top_pose[p + "values"]
                    points = road.range_points(sensor[prefix + "values"], sensor[prefix + "shape"],
                                               calibrations[laser], pixel, poses[ts])
                    # Rasterize all valid same-frame points; no sweep accumulation or person segmentation.
                    px = np.rint(BEV_SIZE / 2 + (points[:, 1] - center[1]) * BEV_SIZE / BEV_SPAN).astype(int)
                    py = np.rint(BEV_SIZE / 2 - (points[:, 0] - center[0]) * BEV_SIZE / BEV_SPAN).astype(int)
                    valid = (px >= 0) & (px < BEV_SIZE) & (py >= 0) & (py < BEV_SIZE)
                    visible_points += int(valid.sum())
                    array = np.asarray(bev).copy()
                    array[py[valid], px[valid]] = (160, 160, 160)
                    bev = Image.fromarray(array)
            bev_draw = ImageDraw.Draw(bev)
            if not cloud:
                warnings.append("missing point cloud")
            elif set(calibrations) - seen:
                warnings.append("missing LiDAR sensors: " + ",".join(map(str, sorted(set(calibrations) - seen))))
            if cloud and visible_points == 0:
                warnings.append("no LiDAR points in view")
            box3d = observation["box3d"] if observation else None
            if box3d is not None:
                x, y, _z, sx, sy, _sz, yaw = box3d
                label = dict(pos_x=x, pos_y=y, dim_x=sx, dim_y=sy, yaw=yaw)
                corners = [bev_pixel(x, y, center) for x, y in footprint(label)]
                bev_draw.line(corners + corners[:1], fill="yellow", width=3)
            else:
                warnings.append("missing 3D box")
            if not has_focus:
                warnings.append("no 3D focus; ego-centered view")
            bev_draw.rectangle((0, 0, BEV_SIZE, 22 + 14 * len(warnings)), fill="black")
            bev_draw.text((8, 5), "+x up | +y right | 40 m wide | native vehicle frame", fill="white")
            for i, warning in enumerate(warnings):
                bev_draw.text((8, 23 + 14 * i), f"[{warning}]", fill="yellow")
            actions = ", ".join(observation["actions"]) if observation else "unknown [missing 2D box]"
            locations = ", ".join(observation["locations"]) if observation else "unknown"
            if observation and observation["disagreement"]:
                actions += f" | original 3D: {observation['original_type']} [class disagreement]"
            draw.rectangle((0, 0, RGB_SIZE[0], 48), fill="black")
            draw.text((6, 5), f"{scene['road_clip_id']} | track {track_id} | frame {n} | timestamp {ts}", fill="white")
            draw.text((6, 24), f"action: {actions} | location: {locations}", fill="white")
            combined = Image.new("RGB", (1564, 604))
            combined.paste(rgb, (0, 0))
            combined.paste(bev, (960, 0))
            process.stdin.write(combined.tobytes())
            if n % 50 == 0:
                print(f"  {n}/{len(images)} frames", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        with suppress(BrokenPipeError):
            process.stdin.close()
        output.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--waymo-root", type=Path, help="Relocate manifest components to this Waymo root")
    parser.add_argument("--action", required=True, help="Exact native action, e.g. Wait2X")
    parser.add_argument("--clip", help="Restrict selection to this ROAD clip")
    parser.add_argument("--track", help="Restrict selection to this ROAD tube_uid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1:
        parser.error("--offset must be nonnegative and --limit must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output directory is not empty")
    manifest = json.loads((args.index / "scene_manifest.json").read_text())
    scenes = {s["road_clip_id"]: s for s in manifest}
    if len(scenes) != len(manifest):
        parser.error("duplicate clips in manifest")
    matches = set()
    for row in road.annotation_rows(args.index):
        key = row["road_clip_id"], row["road_tube_uid"]
        if (not args.clip or args.clip == key[0]) and (not args.track or args.track == key[1]):
            if args.action in json.loads(row["action_labels_json"]):
                matches.add(key)
    selected = select(matches, args.seed, args.offset, args.limit)
    if not selected:
        parser.error("no matching tracks at this offset")
    tracks = road.read_tracks(args.index, selected)
    args.output.mkdir(parents=True, exist_ok=True)
    for clip, track in selected:
        print(f"rendering {clip} {track}", flush=True)
        render_track(scenes[clip], track, tracks[(clip, track)], args.output / f"{clip}_{track}.mp4", args.waymo_root)
    write_index(args.output, args.action, selected, dataset="ROAD-Waymo")
    print(f"{len(selected)} of {len(matches)} matching tracks: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
