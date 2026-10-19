"""Ten complete development-track videos with nested RGB and semantic BEV windows."""

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import html
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
from pedestrian_behavior.inspection.feature_video import VIDEO_SIZE, render_track_frame
from pedestrian_behavior.inspection.native_features import native_scene
from pedestrian_behavior.inspection.pretrained import PointEncoder
from .e002a_probe import write_json


def choose_tracks(candidates, count=5):
    """Prefer paired observation support, then distinct scenes; never use behavior."""
    ranked = sorted((c for c in candidates if c["paired_slots"] > 0),
                    key=lambda c: (-c["paired_slots"], -c["paired_slots"]/c["slots"], c["locator"]))
    selected, groups = [], set()
    for c in ranked:
        if c["group"] not in groups:
            selected.append(c)
            groups.add(c["group"])
        if len(selected) == count:
            return selected
    for c in ranked:
        if c not in selected:
            selected.append(c)
        if len(selected) == count:
            return selected
    raise ValueError("Insufficient development tracks with paired observations")


def video_gallery(record, output):
    lines = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Whole-track crop and LiDAR videos</title>',
        '<style>body{font:16px system-ui;background:#171b22;color:white;margin:24px}video{width:100%;max-width:1440px}a{color:#8cf}article{margin:35px 0}</style>',
        '<h1>10 whole-track RGB / semantic LiDAR videos</h1>',
        '<p>5 Hz frozen grid, complete tracks including missing context. One 3x RGB window with 1.5x/2x/3x outlines; target-centered semantic BEV with 15/20/30 m radii. Both move with the same target. No interpolation, training, held-out access or segmentation accuracy claim.</p>']
    for track in record["tracks"]:
        name=track["name"]
        lines.append(f'<article><h2>{html.escape(name)}</h2><p>{html.escape(track["locator"])} | {track["split"]} | {track["slots"]} slots ({track["slots"]/5:.1f}s)</p>')
        if track["status"] == "complete":
            lines.append(f'<video controls preload="metadata" poster="{name}-middle.png" src="{name}.mp4"></video><p><a href="{name}.mp4">Download MP4</a></p>')
        else:
            lines.append('<p>Not completed; see videos.json.</p>')
        lines.append('</article>')
    (output/"index.html").write_text("\n".join(lines)+"</html>\n")


def run(args):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    probe, output = args.probe.resolve(), args.output.resolve()
    root = probe.parents[2]
    if not output.is_relative_to(root/"experiments/E002a"):
        raise ValueError("Output must be under the probe's outputs/experiments/E002a")
    source = json.loads((probe/"probe.json").read_text())
    if source["status"] != "complete":
        raise ValueError("A completed probe is required")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status":"incomplete", "started_utc":datetime.now(timezone.utc).isoformat(),
        "command":sys.orig_argv, "revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "torch":torch.__version__, "cuda":torch.version.cuda, "device":args.device, "seed":0,
        "grid_hz":5, "rgb_scales":[1.5,2,3], "lidar_radii_m":[15,20,30],
        "selection":"five tracks/dataset from frozen probe groups; most paired slots, distinct groups preferred; no behavior labels/errors",
        "tracks":[], "failures":[]}
    config = source["config"]
    protected = source["protected"].copy()
    for p in (probe/"probe.json", Path(config["ditr_weights"]), Path(config["ditr_config"])):
        protected[str(p)] = checksum(p)
    started=time.monotonic()
    write_json(output/"videos.json",record)
    try:
        for path,digest in protected.items():
            if checksum(path)!=digest:
                raise ValueError(f"Protected source changed: {path}")
        encoder=PointEncoder(config["ditr_code"],config["ditr_config"],config["ditr_weights"],args.device)
        record["checkpoint_sha256"]=checksum(config["ditr_weights"])
        import matplotlib
        palette=np.uint8(matplotlib.colormaps["tab20"](np.arange(len(encoder.names)))[:,:3]*255)
        for dataset in ("loki","road-waymo"):
            collection=root/"experiments/E001/reader-preparation"/dataset
            manifest=json.loads((collection/"manifest.json").read_text())
            setup=json.loads((root/"experiments/E001/data-setup"/dataset/"setup.json").read_text())
            if setup["collection"]["manifest_sha256"]!=checksum(collection/"manifest.json"):
                raise ValueError("Frozen collection/setup mismatch")
            entries={e["archive"]:e for e in setup["tracks"] if not e["exclusions"] and e["split"] in ("training","validation")}
            candidates=[]
            for group,sample in source["samples"][dataset].items():
                if sample["split"] not in ("training","validation"):
                    raise ValueError("Test scene in probe sample")
                for archive in sample["archives"]:
                    entry=entries[archive]
                    metadata,arrays=load_track(collection/"tracks"/archive)
                    if entry["group"]!=group or metadata["track_locator"]!=entry["locator"]:
                        raise ValueError("Frozen candidate identity differs")
                    flag="loki_2d" if dataset=="loki" else "waymo_2d"
                    paired=sum(bool(a.get(flag)) and bool(a.get("native_3d")) for a in metadata["availability"])
                    candidates.append({"locator":entry["locator"],"archive":archive,"group":group,
                        "split":entry["split"],"slots":len(metadata["source_frames"]),"paired_slots":paired})
            selected=choose_tracks(candidates)
            if dataset=="loki":
                reader=loki.TrackReader(config["loki_root"] or manifest["sources"]["root"],probe/"loki-transform.json")
            else:
                waymo_root=Path(config["waymo_root"]) if config["waymo_root"] else None
                reader=road_waymo.TrackReader(config["road_index"] or manifest["sources"]["index"],waymo_root)
                if reader.sources["sha256"]!=manifest["sources"]["sha256"]:
                    raise ValueError("Frozen ROAD index mismatch")
            for number,selection in enumerate(selected,1):
                item=selection|{"name":f"{dataset}-{number:02d}","status":"incomplete","frames":[]}
                record["tracks"].append(item)
                metadata,_=load_track(collection/"tracks"/selection["archive"])
                identity=json.loads(selection["locator"])
                track=identity.get("track_id",identity.get("tube_uid"))
                group=selection["group"]
                keys=metadata["source_frames"]
                reader.index_scene(group)
                first=reader.read_frames(group,[keys[0]]) if keys[0] is not None else {}
                anchor=first.get(keys[0],{}).get("ego_pose")
                if not valid_pose(anchor):
                    raise ValueError("Selected track lacks its frozen initial ego anchor")
                item["anchor_ego_pose"]=np.asarray(anchor).tolist()
                stream=iter(native_scene(reader,group,[k for k in keys if k is not None],[track],manifest["scenes"][group]))
                print("Rendering",item["name"],group,"slots",len(keys),flush=True)
                with (output/(item["name"]+"-ffmpeg.log")).open("wb") as log:
                    process=subprocess.Popen(["ffmpeg","-nostdin","-loglevel","error","-n","-f","rawvideo",
                        "-pixel_format","rgb24","-video_size",f"{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}","-framerate","5",
                        "-i","-","-an","-c:v","libx264","-threads","2","-preset","veryfast","-crf","20",
                        "-pix_fmt","yuv420p","-movflags","+faststart",str(output/(item["name"]+".mp4"))],
                        stdin=subprocess.PIPE,stderr=log)
                    try:
                        for slot,key in enumerate(keys):
                            frame=next(stream) if key is not None else None
                            if frame and frame["key"]!=key:
                                raise ValueError("Video/native frame synchronization mismatch")
                            offsets=labels=None
                            box=frame["boxes"].get(track) if frame else None
                            if box is not None and frame["points"] is not None and len(frame["points"]) and valid_pose(frame["world_from_native"]):
                                _,logits,indices,_=encoder(frame["points"])
                                offsets=relative_xyz(frame["points"][indices,:3],box[0],frame["world_from_native"],anchor)
                                labels=logits.argmax(1)
                            heading=f"{item['name']} | {group} | t={slot/5:.1f}s | slot {slot+1}/{len(keys)} | native {key}"
                            image,evidence=render_track_frame(frame,track,anchor,offsets,labels,palette,encoder.names,heading)
                            process.stdin.write(image.tobytes())
                            evidence.update(slot=slot,key=key,source_time_us=source_times(metadata)[slot])
                            if box is not None:
                                evidence["native_box"]={"center":np.asarray(box[0]).tolist(),"size":np.asarray(box[1]).tolist(),"yaw":float(box[2])}
                            item["frames"].append(evidence)
                            for at,label in ((0,"first"),(len(keys)//2,"middle"),(len(keys)-1,"last")):
                                if slot==at:
                                    image.save(output/(item["name"]+"-"+label+".png"))
                    finally:
                        with suppress(BrokenPipeError):
                            process.stdin.close()
                        if process.wait()!=0:
                            raise RuntimeError(f"ffmpeg failed; see {item['name']}-ffmpeg.log")
                info=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                    "-show_entries","stream=width,height,nb_read_frames,r_frame_rate,duration","-of","json",str(output/(item["name"]+".mp4"))],text=True))["streams"][0]
                if int(info["nb_read_frames"])!=len(keys) or info["r_frame_rate"]!="5/1":
                    raise ValueError("Encoded video lost grid frames or changed playback rate")
                item.update(status="complete",video=info)
                write_json(output/"videos.json",record)
                video_gallery(record,output)
                print("Completed",item["name"],"frames",len(keys),flush=True)
        record["status"]="complete"
    except Exception:
        record["status"]="failed"
        record["failures"].append(traceback.format_exc())
        raise
    finally:
        changed=[p for p,digest in protected.items() if checksum(p)!=digest]
        record.update(protected_files=len(protected),changed_files=changed,seconds=time.monotonic()-started)
        if changed:
            record["status"]="failed"
        write_json(output/"videos.json",record)
        video_gallery(record,output)
        if changed:
            raise ValueError("Protected sources changed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
