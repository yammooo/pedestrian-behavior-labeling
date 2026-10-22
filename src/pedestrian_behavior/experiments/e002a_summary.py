"""Paired source-validation track changes from saved pilot predictions, no inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from pedestrian_behavior.evaluation import load_predictions, prediction_metrics
from .e002a_run import ARMS, FUSION_PROTOCOL
from .run import write_json


def summarize(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    result = {"scope": "exploratory source validation, one seed; no held-out results or superiority claim", "sources": {}}
    protocols = set()
    point_protocols = set()
    for source in ("road-waymo", "loki"):
        arms = {}
        for arm in ARMS:
            run = root / f"e002a-{source}-{arm}-s0"
            if json.loads((run / "status.json").read_text())["status"] != "complete":
                raise ValueError("All six attempts must complete before paired summary")
            config = json.loads((run / "config.json").read_text())
            protocol = config.get("fusion_protocol", "shared-availability-v1")
            if arm == "g-rgb-dditr":
                point_protocols.add(config.get("point_storage_protocol") or "legacy-unpooled")
                if len(point_protocols) != 1:
                    raise ValueError("Paired sources use different point storage protocols")
            protocols.add(protocol)
            if len(protocols) != 1 or protocol not in (FUSION_PROTOCOL, "shared-availability-v1"):
                raise ValueError("Paired arms use different or unknown fusion protocols")
            tracks = load_predictions(run / f"{source}-validation/predictions.npz")
            arms[arm] = {t["locator"]: t for t in tracks}
        baseline = arms["g"]
        if any(set(rows) != set(baseline) for rows in arms.values()):
            raise ValueError("Paired arms have different frozen validation populations")
        scores, pairs, observed = {}, [], {}
        for arm, rows in arms.items():
            scores[arm] = prediction_metrics(list(rows.values()), 4)
            for key, track in rows.items():
                base = baseline[key]
                if (not np.array_equal(track["targets"], base["targets"]) or not np.array_equal(track["gt_valid"], base["gt_valid"])):
                    raise ValueError("Paired arms changed target alignment")
                expected = (("rgb_valid", "lidar_valid", "point_tokens") if protocol == "shared-availability-v1"
                            or arm == "g-rgb-dditr" else ("rgb_valid",) if arm == "g-rgb" else ())
                for flag in expected:
                    if flag not in track["source_flags"]:
                        raise ValueError("Missing enabled-branch availability evidence")
                    value = track["source_flags"][flag]
                    reference = observed.setdefault((key, flag), value)
                    if not np.array_equal(value, reference):
                        raise ValueError("Observed availability differs across enabled arms")
        for arm, reference in (("g-rgb", "g"), ("g-rgb-dditr", "g-rgb"), ("g-rgb-dditr", "g")):
            for key in baseline:
                track = arms[arm][key]
                a, b = (prediction_metrics([x], 4) for x in (track, arms[reference][key]))
                pairs.append({"locator": key, "arm": arm, "reference_arm": reference,
                              "accuracy_change": a["mean_track_accuracy"] - b["mean_track_accuracy"],
                              "ce_change": a["track_ce"] - b["track_ce"], "gt_frames": int(track["gt_valid"].sum())})
        result["sources"][source] = {"metrics": scores, "paired_track_changes": pairs}
    result["fusion_protocol"] = protocols.pop()
    result["point_storage_protocol"] = point_protocols.pop()
    write_json(root / "paired-validation.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (source, value) in zip(axes, result["sources"].items()):
        ax.bar(list(ARMS), [value["metrics"][a]["track_weighted"]["macro_f1"] for a in ARMS])
        ax.set(title=source + ": source validation", ylabel="Four-class track-weighted macro-F1", ylim=(0, 1))
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(root / ("validation-comparison." + suffix), dpi=150)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
