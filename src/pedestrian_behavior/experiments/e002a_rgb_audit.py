"""Read back every prepared RGB entry; quantify source split/GT coverage without inference."""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from pedestrian_behavior.data.feature_cache import FeatureCache
from pedestrian_behavior.data.preparation import checksum, load_track
from .e001 import targets
from .e002a_features import FeatureCatalog, FeatureSupply, extraction_identity, load_configuration
from .run import write_json


def support(y, gt, rgb):
    if y.shape != gt.shape or rgb.shape != gt.shape:
        raise ValueError("RGB/GT audit alignment differs")
    return {"tracks": 1, "context_slots": len(y), "gt_frames": int(gt.sum()),
            "class_gt_frames": np.bincount(y[gt], minlength=4).tolist(),
            "rgb_slots": int(rgb.sum()), "rgb_tracks": int(rgb.any()), "rgb_gt_frames": int((rgb & gt).sum()),
            "rgb_class_gt_frames": np.bincount(y[rgb & gt], minlength=4).tolist()}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"status": "incomplete", "preparation_sha256": checksum(args.preparation), "support": {}, "failures": []}
    protected = {}
    try:
        preparation = json.loads(args.preparation.read_text())
        config, transform = load_configuration(args.probe)
        identity = extraction_identity(args.device, config)
        if preparation["status"] != "complete" or preparation["extraction"] != identity:
            raise ValueError("Matching completed RGB preparation required")
        catalogs = {d: FeatureCatalog(args.outputs_root, d, config, transform) for d in ("loki", "road-waymo")}
        for c in catalogs.values():
            protected.update(c.protected)
        def no_inference(*_):
            raise ValueError("Full RGB audit encountered a missing entry; no inference is permitted")
        with FeatureCache(Path(preparation["config"]["rgb_cache_root"])) as rgb:
            supply = FeatureSupply(catalogs, rgb, None, no_inference, identity)
            for dataset, catalog in catalogs.items():
                sums, groups = {}, {}
                for key, entry in catalog.tracks.items():
                    values = supply.rgb(dataset, key)
                    metadata, _ = load_track(catalog.collection / "tracks" / entry["archive"])
                    y, gt = targets(metadata, dataset, catalog.setup["native_audit"]["verified_overrides"])
                    row = support(y, gt, values["valid"])
                    if row["class_gt_frames"] != entry["class_frames"] or row["gt_frames"] != entry["gt_frames"]:
                        raise ValueError("Cache audit changed frozen GT support")
                    split = entry["split"]
                    groups.setdefault(split, set()).add(entry["group"])
                    if split not in sums:
                        sums[split] = row
                    else:
                        sums[split] = {k: (np.asarray(sums[split][k]) + v).tolist() for k, v in row.items()}
                record["support"][dataset] = {k: v | {"groups": len(groups[k])} for k, v in sums.items()}
            record["cache"] = rgb.snapshot()
            if record["cache"]["misses"]:
                raise ValueError("RGB readback unexpectedly inferred features")
        changed = [p for p, h in protected.items() if checksum(p) != h]
        if changed:
            raise ValueError("Frozen source changed during RGB audit")
        record.update(status="complete", protected_files=len(protected), changed_files=changed,
                      active_payload_bytes=record["cache"]["read_bytes"], encoder_calls=0)
        return record
    except BaseException:
        record["status"] = "failed"; record["failures"].append(traceback.format_exc()); raise
    finally:
        write_json(args.output / "audit.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run(parser.parse_args())
