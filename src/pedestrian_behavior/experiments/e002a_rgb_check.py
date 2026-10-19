"""Bounded DINO patch visualizations on the existing E002a development sample."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.datasets import loki, road_waymo
from pedestrian_behavior.inspection.feature_probe import (crop_examples, crop_sheet, dino_sheet,
    normalized_crop, patch_maps, square_crop)
from pedestrian_behavior.inspection.native_features import native_camera
from pedestrian_behavior.inspection.pretrained import ImageEncoder
from .e002a_probe import write_json


def gallery(record, output, probe, *, panels=None, native=False):
    panels = panels or output
    title = "Native RGB + LiDAR checks" if native else "DINO patch sanity checks"
    lines = [f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title>',
        '<style>body{font:16px system-ui;background:#202020;color:white;margin:24px}a{color:#8cf}img{max-width:100%}article{margin-bottom:35px}summary{cursor:pointer}</style>',
        f'<h1>{title}</h1>',
        '<p>Counts cover represented eligible tracks on their frozen grids, not every person in the image. Valid crops appear first; one missing example follows. A scene RGB image can exist without a target FRONT box.</p>']
    if native:
        lines.append('<p>Original 3D semantic predictions with corrected target-crop selection. Missing-crop examples are collapsed scene context, not empty pedestrian crops. No encoder inference was repeated.</p>')
    else:
        lines.append('<p>Development crops only. PCA colors are fitted independently per crop and cannot be compared across examples. Similarity is to the marked box-centre patch, which may contain background or occlusion. These are neither segmentation nor behavior predictions. Patch resolution is 16×16; overlays only enlarge it.</p>')
    for frame in record["frames"]:
        label = frame["label"]
        lines.append(f'<article><h2>{html.escape(label)}</h2><p><b>{frame["valid_crops"]}/{frame["tracks"]} valid crops</b>; {len(frame["examples"])} displayed examples.</p>')
        bev = probe/"panels"/(label+"-bev.png")
        if bev.exists():
            link = html.escape(os.path.relpath(bev, output), quote=True)
            image = f'<a href="{link}"><img loading="lazy" src="{link}"></a>'
            lines.append(image if native else f'<details><summary>Existing LiDAR BEV</summary>{image}</details>')
        if not frame["valid_crops"]:
            lines.append('<p><b>No valid target FRONT/RGB crop at this timestamp.</b></p>')
        for example in frame["examples"]:
            name = example["name"]
            missing = not example["crop"]["valid"]
            if missing:
                lines.append('<details><summary>Missing target crop — scene context only</summary>')
            lines.append(f'<p>{html.escape(example["locator"])} — {html.escape(example["crop"].get("reason", "valid crop"))}</p>')
            for suffix in (["crop", "dino"] if not missing and not native else ["crop"]):
                path = html.escape(os.path.relpath(panels/"panels"/f"{name}-{suffix}.png", output), quote=True)
                lines.append(f'<a href="{path}"><img loading="lazy" src="{path}"></a>')
            if missing:
                lines.append('</details>')
        lines.append('</article>')
    (output/"index.html").write_text("\n".join(lines)+"</html>\n")


def refresh_probe_gallery(probe, visual):
    """Replace only presentation HTML, retaining the former gallery as evidence."""
    probe, visual = Path(probe), Path(visual)
    source = json.loads((probe/"probe.json").read_text())
    corrected = json.loads((visual/"visual-check.json").read_text())
    if source["status"] != "complete" or corrected["status"] != "complete":
        raise ValueError("Only completed probe/visual evidence can refresh a gallery")
    frames = {f"{f['dataset']}-{f['group']}-{f['key']}": f for f in source["frames"] if f.get("rendered")}
    if set(frames) != {f["label"] for f in corrected["frames"]}:
        raise ValueError("Probe and corrected gallery frame sets differ")
    for frame in corrected["frames"]:
        original = frames[frame["label"]]["tracks"]
        by_id = {t["locator"]: t for t in original}
        if frame["tracks"] != len(original) or frame["valid_crops"] != sum(t["crop"]["valid"] for t in original):
            raise ValueError("Corrected crop counts differ from original numerical evidence")
        for example in frame["examples"]:
            if example["crop"] != by_id[example["locator"]]["crop"]:
                raise ValueError("Corrected example differs from original native crop")
            if not (visual/"panels"/(example["name"]+"-crop.png")).is_file():
                raise ValueError("Missing corrected crop panel")
    snapshot = probe/"index-before-crop-fix.html"
    if not snapshot.exists():
        snapshot.write_bytes((probe/"index.html").read_bytes())
    gallery(corrected, probe, probe, panels=visual, native=True)


def run(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    probe, output = args.probe.resolve(), args.output.resolve()
    root = probe.parents[2]
    if not output.is_relative_to(root/"experiments/E002a"):
        raise ValueError("Output must be under the same outputs/experiments/E002a root")
    source = json.loads((probe/"probe.json").read_text())
    if source["status"] != "complete":
        raise ValueError("A completed native probe is required")
    frames = [f for f in source["frames"] if f.get("rendered")]
    if any(f["split"] not in ("training", "validation") for f in frames):
        raise ValueError("Only development frames are allowed")
    output.mkdir(parents=True, exist_ok=False)
    (output/"panels").mkdir()
    (output/"fixtures").mkdir()
    record = {"status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.orig_argv, "probe_sha256": checksum(probe/"probe.json"),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "numpy": np.__version__, "device": args.device,
        "seed": 0, "frames": [], "failures": []}
    protected = source["protected"] | {str(probe/"probe.json"): record["probe_sha256"]}
    config = source["config"]
    protected.update({str(p): checksum(p) for p in [Path(config["dino_weights"]), *sorted((probe/"fixtures").glob("*-rgb.npz"))]})
    started = time.monotonic()
    write_json(output/"visual-check.json", record)
    try:
        for path, expected in protected.items():
            if checksum(path) != expected:
                raise ValueError(f"Protected input changed: {path}")
        encoder = ImageEncoder(config["dino_code"], config["dino_weights"], args.device)
        record["checkpoint_sha256"] = checksum(config["dino_weights"])
        for dataset in ("loki", "road-waymo"):
            collection = root/"experiments/E001/reader-preparation"/dataset
            manifest = json.loads((collection/"manifest.json").read_text())
            if dataset == "loki":
                reader = loki.TrackReader(config["loki_root"] or manifest["sources"]["root"], probe/"loki-transform.json")
            else:
                waymo_root = Path(config["waymo_root"]) if config["waymo_root"] else None
                reader = road_waymo.TrackReader(config["road_index"] or manifest["sources"]["index"], waymo_root)
                if reader.sources["sha256"] != manifest["sources"]["sha256"]:
                    raise ValueError("Frozen ROAD index changed")
            grouped = defaultdict(dict)
            for f in frames:
                if f["dataset"] == dataset:
                    grouped[f["group"]][f["key"]] = f
            for group, entries in grouped.items():
                tracks = {json.loads(t["locator"]).get("track_id", json.loads(t["locator"]).get("tube_uid"))
                          for f in entries.values() for t in f["tracks"]}
                for key, image, boxes in native_camera(reader, group, entries, tracks, manifest["scenes"][group]):
                    frame = entries[key]
                    label = f"{dataset}-{group}-{key}"
                    item = {"label": label, "split": frame["split"], "tracks": len(frame["tracks"]),
                        "valid_crops": sum(t["crop"]["valid"] for t in frame["tracks"]), "examples": []}
                    valid_indices = [i for i, t in enumerate(frame["tracks"]) if t["crop"]["valid"]]
                    descriptors = {}
                    # Match the original FP16 extraction batches, including crop order.
                    for start in range(0, len(valid_indices), 16):
                        indices = valid_indices[start:start+16]
                        inputs = []
                        for i in indices:
                            old = frame["tracks"][i]
                            identity = json.loads(old["locator"])
                            crop, info = square_crop(image, boxes.get(identity.get("track_id", identity.get("tube_uid"))))
                            if info != old["crop"]:
                                raise ValueError("Native crop differs from frozen probe")
                            inputs.append(normalized_crop(crop))
                        inputs = torch.stack(inputs)
                        cls, patches = encoder.descriptors(inputs, patches=True)
                        if "repeat_max_error" not in record:
                            again_cls, again_patches = encoder.descriptors(inputs, patches=True)
                            record["repeat_max_error"] = float(max(np.max(np.abs(cls-again_cls)), np.max(np.abs(patches-again_patches))))
                            if record["repeat_max_error"] != 0:
                                raise ValueError("Patch extraction repeatability failed")
                        descriptors.update({i: (cls[j], patches[j]) for j, i in enumerate(indices)})
                    for i in crop_examples(frame["tracks"], valid_limit=2):
                        old = frame["tracks"][i]
                        identity = json.loads(old["locator"])
                        box = boxes.get(identity.get("track_id", identity.get("tube_uid")))
                        crop, info = square_crop(image, box)
                        if info != old["crop"]:
                            raise ValueError("Native crop differs from frozen probe")
                        name = f"{label}-track-{i}"
                        example = {"name": name, "locator": old["locator"], "slot": old["slot"], "crop": info}
                        crop_sheet(image, box, crop, info, output/"panels"/(name+"-crop.png"))
                        if crop is not None:
                            cls, patches = descriptors[i]
                            # Select the centre of the image-visible part of the box.
                            b = np.asarray(box)*np.tile(image.size, 2)
                            center = (np.maximum(b[:2], 0)+np.minimum(b[2:], image.size))/2
                            bounds = np.array(info["bounds_pixels"])
                            xy = np.floor((center-bounds[:2])/(bounds[2]-bounds[0])*16).astype(int).clip(0, 15)
                            seed = (int(xy[1]), int(xy[0]))
                            colors, similarity = patch_maps(patches, seed)
                            with np.load(probe/"fixtures"/(label+"-rgb.npz"), allow_pickle=False) as saved:
                                match = np.flatnonzero(saved["locators"] == old["locator"])
                                if len(match) != 1:
                                    raise ValueError("Missing/duplicate saved descriptor identity")
                                error = float(np.max(np.abs(cls-saved["cls"][match[0]])))
                                if not np.allclose(cls, saved["cls"][match[0]], atol=1e-4, rtol=1e-4):
                                    raise ValueError("CLS differs from previous batched extraction")
                            example.update(seed_row_column=list(seed), cls_previous_max_error=error)
                            np.savez(output/"fixtures"/(name+".npz"), crop=crop, cls=cls, patches=patches,
                                pca_rgb=colors, cosine_similarity=similarity, seed_row_column=np.array(seed),
                                locator=np.array(old["locator"]), key=np.array(str(key)))
                            dino_sheet(crop, colors, similarity, seed, output/"panels"/(name+"-dino"))
                        item["examples"].append(example)
                    record["frames"].append(item)
                    write_json(output/"visual-check.json", record)
                    gallery(record, output, probe)
                    print(label, f"{item['valid_crops']}/{item['tracks']} valid", flush=True)
        if not record["frames"]:
            raise ValueError("No rendered development frames")
        record["status"] = "complete"
    except Exception:
        record["status"] = "failed"
        record["failures"].append(traceback.format_exc())
        raise
    finally:
        changed = [p for p, expected in protected.items() if checksum(p) != expected]
        record.update(protected_files=len(protected), changed_files=changed, seconds=time.monotonic()-started,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        if changed:
            record["status"] = "failed"
        write_json(output/"visual-check.json", record)
        gallery(record, output, probe)
        if changed:
            raise ValueError("Protected inputs changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
