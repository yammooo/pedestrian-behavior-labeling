"""Read-only real-result/native-context acceptance; writes only viewer evidence."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import prediction_metrics
from pedestrian_behavior.experiments.e001b_run import STRATA
from pedestrian_behavior.inspection.predictions import Viewer
from pedestrian_behavior.inspection.prediction_index import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loki-root", type=Path, required=True)
    parser.add_argument("--road-index", type=Path, required=True)
    parser.add_argument("--waymo-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    viewer = Viewer(args.outputs_root, args.output/"cache", STRATA, loki_root=args.loki_root,
                    road_index=args.road_index, waymo_root=args.waymo_root)
    try:
        catalog = viewer.index.catalog()
        candidates = sorted((r for r in catalog["runs"] if r["source"] == "loki" and r["architecture"] == "BiLSTM" and r["seed"] == 0),
                            key=lambda r:(r["epochs"] != 75, r["patience"] != 8, r["relative"]))
        selected = []
        for experiment in ("E001", "E001b", "E001c"):
            selected.append(next(r for r in candidates if r["experiment"] == experiment))
        selected.append(next(r for r in candidates if r["id"] not in {s["id"] for s in selected}))
        runs = [r["id"] for r in selected]
        protected = [p for r in selected for p in Path(r["path"]).rglob("*") if p.is_file() and p.suffix in (".json", ".npz", ".pt")]
        protected += list(args.outputs_root.glob("experiments/E001/reader-preparation/*/manifest.json"))
        protected += list(args.outputs_root.glob("experiments/E001/data-setup/*/setup.json"))
        protected += list(args.outputs_root.glob("experiments/E001b/bbox-native/*/manifest.json"))
        protected += list(args.outputs_root.glob("experiments/E001c/box-cache/*/manifest.json"))
        before = {str(p.resolve()): checksum(p) for p in protected}
        write_json(args.output/"protected-before.json", before)
        result = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "catalog_runs": len(catalog["runs"]), "warnings": catalog["warnings"], "runs": selected, "datasets": {}}
        for dataset in ("loki", "road-waymo"):
            started = time.monotonic()
            query = viewer.index.query(runs, dataset)
            cold = time.monotonic()-started
            started = time.monotonic()
            viewer.index.query(runs, dataset, metric="macro_f1")
            warm = time.monotonic()-started
            loaded = viewer.index.load(runs[0], dataset)
            metric = prediction_metrics(list(loaded["tracks"].values()), 4)
            assert np.isclose(np.mean([t["scores"]["accuracy"] for t in loaded["tracks"].values()]), metric["mean_track_accuracy"], atol=1e-15)
            examples = viewer.index.query(runs, dataset, axis="frame-2d", value=1, scope="whole", minimum=20)
            locator = examples["rows"][0]["locator"]
            track = loaded["tracks"][locator]
            print(f"{dataset}: {query['common']} shared tracks, index cold={cold:.2f}s / warm={warm:.2f}s; preparing {locator}", flush=True)
            context = viewer.context.prepare(dataset, track["archive"], locator, lambda *_:None, expected=track)
            cached = viewer.context.prepare(dataset, track["archive"], locator, lambda *_:None, expected=track)
            assert cached["cache_hit"] and cached["key"] == context["key"]
            assert len(context["frames"]) == len(track["targets"])
            frame = next(i for i, evidence in enumerate(context["frames"]) if evidence["rgb"] and evidence["native_box2d"])
            for image in ("rgb", "rgb-road", "bev", "crop"):
                shutil.copyfile(args.output/"cache/context"/context["key"]/f"{frame:04d}-{image}.jpg", args.output/f"{dataset}-{image}.jpg")
            result["datasets"][dataset] = {"common_tracks": query["common"], "missing": query["missing"],
                "cold_index_seconds": cold, "warm_query_seconds": warm, "locator": locator, "frame": frame,
                "context_seconds": context["seconds"], "context_cache_reused": True, "context_key": context["key"],
                "frames": len(context["frames"]), "rgb_frames": sum(f["rgb"] for f in context["frames"]),
                "lidar_frames": sum(f["lidar"] for f in context["frames"]), "track_accuracy_reconstructed": True}
            print(f"{dataset}: {len(context['frames'])} context frames in {context['seconds']:.2f}s; cache reused", flush=True)
        assert all(checksum(Path(path)) == digest for path, digest in before.items())
        result["protected_files"] = len(before)
        result["protected_unchanged"] = True
        write_json(args.output/"verification.json", result)
        print(f"Verified {len(before)} unchanged source artifacts; evidence: {args.output}", flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
