"""Read-only kinematic summaries for frozen E002 validation predictions."""

import json
from pathlib import Path

import numpy as np


ROOT = Path("outputs/experiments/E002/transfer-validation-20261102")
E003B = Path("outputs/experiments/E003b/runs/20261104-092202")
OUTPUT = Path("outputs/experiments/E002/transfer-confusion-diagnostic-20261110/summary.json")
CLASSES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
INPUTS = ("geometry", "geometry+rgb", "geometry+lidar", "geometry+rgb+lidar")
CELLS = ((1, 0, "STOPPED_TO_MOVING"), (2, 0, "WAITING_TO_MOVING"),
         (2, 1, "WAITING_TO_STOPPED"),
         (3, 0, "CROSSING_TO_MOVING"), (0, 1, "MOVING_TO_STOPPED"),
         (0, 0, "MOVING_CORRECT"), (1, 1, "STOPPED_CORRECT"),
         (2, 2, "WAITING_CORRECT"), (3, 3, "CROSSING_CORRECT"))
TRANSITIONS = ("near", "steady", "unknown")


def read_predictions(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        offsets = saved["offsets"]
        targets, valid = saved["targets"], saved["gt_valid"]
        logits = saved["logits"]
        fields = {name: saved[name] for name in saved.files
                  if name.startswith("condition_") or name.startswith("source_")}
        records = {}
        for i, item in enumerate(metadata):
            start, stop = offsets[i:i + 2]
            locator = item["locator"]
            if locator in records:
                raise ValueError(f"Duplicate prediction locator: {locator}")
            record = {"locator": locator, "archive": item["archive"],
                      "targets": targets[start:stop].copy(), "gt_valid": valid[start:stop].copy(),
                      "predictions": logits[start:stop].argmax(axis=1),
                      "source_times_us": item.get("source_times_us")}
            record["conditions"] = {k.removeprefix("condition_"): v[start:stop].copy()
                                    for k, v in fields.items() if k.startswith("condition_")}
            record["source_flags"] = {k.removeprefix("source_"): v[start:stop].copy()
                                      for k, v in fields.items() if k.startswith("source_")}
            if len(record["targets"]) != len(record["gt_valid"]) or not record["gt_valid"].any():
                raise ValueError(f"Invalid GT population: {locator}")
            records[locator] = record
    return records


def check_aligned(reference, candidate, label):
    if reference.keys() != candidate.keys():
        raise ValueError(f"Population differs for {label}")
    for locator, first in reference.items():
        second = candidate[locator]
        if (not np.array_equal(first["targets"], second["targets"])
                or not np.array_equal(first["gt_valid"], second["gt_valid"])):
            raise ValueError(f"GT or frame alignment differs for {label}: {locator}")
        if first.get("source_times_us") != second.get("source_times_us"):
            raise ValueError(f"Source timestamps differ for {label}: {locator}")


def attach_track_arrays(records, cache):
    for locator, record in records.items():
        if locator not in cache:
            archive = Path(record["archive"])
            with np.load(archive, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"]))
                if metadata["track_locator"] != locator:
                    raise ValueError(f"Track archive locator mismatch: {locator}")
                arrays = {key: saved[key].copy() for key in
                          ("ped_position", "ped_position_valid", "ped_velocity", "ped_velocity_valid",
                           "ego_velocity", "ego_velocity_valid")}
            cache[locator] = arrays
        arrays = cache[locator]
        if len(arrays["ped_velocity"]) != len(record["targets"]):
            raise ValueError(f"Track grid length differs from predictions: {locator}")
        record["ped_speed"] = np.linalg.norm(arrays["ped_velocity"], axis=1)
        record["ped_speed_valid"] = arrays["ped_velocity_valid"] & np.isfinite(record["ped_speed"])
        record["ego_speed"] = np.linalg.norm(arrays["ego_velocity"], axis=1)
        record["ego_speed_valid"] = arrays["ego_velocity_valid"] & np.isfinite(record["ego_speed"])
        record["ped_position"] = arrays["ped_position"]
        record["ped_position_valid"] = arrays["ped_position_valid"]


def weighted_quantile(values, weights, probability):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[min(np.searchsorted(np.cumsum(weights), probability * weights.sum(), side="left"),
                            len(values) - 1)])


def speed_stats(records, selections):
    frame_count = sum(int(mask.sum()) for _, mask in selections)
    track_count = sum(bool(mask.any()) for _, mask in selections)
    weighted_mass = sum(float(mask.sum() / max(int(record["gt_valid"].sum()), 1))
                        for record, mask in selections)
    result = {"frames": frame_count, "tracks": track_count,
              "track_weighted_mass": weighted_mass}
    for name in ("ped_speed", "ego_speed"):
        vals, ws, valid_mass = [], [], 0.0
        for record, mask in selections:
            base = 1.0 / max(int(record["gt_valid"].sum()), 1)
            valid_mask = mask & record[name.replace("speed", "speed_valid")]
            valid_mass += float(valid_mask.sum()) * base
            if valid_mask.any():
                vals.append(record[name][valid_mask])
                ws.append(np.full(int(valid_mask.sum()), base))
        result[name + "_valid_fraction"] = valid_mass / weighted_mass if weighted_mass else None
        if vals:
            values, weights = np.concatenate(vals), np.concatenate(ws)
            result[name + "_median"] = weighted_quantile(values, weights, .5)
            result[name + "_p90"] = weighted_quantile(values, weights, .9)
            if name == "ped_speed":
                result["ped_speed_below_0_2_fraction"] = float(weights[values < .2].sum() / weights.sum())
                result["ped_speed_above_0_5_fraction"] = float(weights[values > .5].sum() / weights.sum())
        else:
            result[name + "_median"] = result[name + "_p90"] = None
            if name == "ped_speed":
                result["ped_speed_below_0_2_fraction"] = result["ped_speed_above_0_5_fraction"] = None
    for flag in ("rgb_valid", "lidar_valid"):
        numerator = denominator = 0.0
        found = False
        for record, mask in selections:
            if flag in record["source_flags"]:
                found = True
                base = 1.0 / max(int(record["gt_valid"].sum()), 1)
                denominator += float(mask.sum()) * base
                numerator += float((mask & record["source_flags"][flag].astype(bool)).sum()) * base
        result[flag + "_fraction"] = numerator / denominator if found and denominator else None
    if any("transition" in record["conditions"] for record, _ in selections):
        transition = {}
        for i, name in enumerate(TRANSITIONS):
            numerator = 0.0
            denominator = 0.0
            for record, mask in selections:
                if "transition" not in record["conditions"]:
                    continue
                base = 1.0 / max(int(record["gt_valid"].sum()), 1)
                condition = record["conditions"]["transition"]
                denominator += float(mask.sum()) * base
                numerator += float((mask & (condition == i)).sum()) * base
            transition[name] = numerator / denominator if denominator else None
        result["transition_fraction"] = transition
    return result


def summarize_dataset(records):
    result = {}
    for class_id, name in enumerate(CLASSES):
        selections = [(record, record["gt_valid"] & (record["targets"] == class_id))
                      for record in records.values()]
        result[name] = speed_stats(records, selections)
    return result


def summarize_errors(records):
    result = {}
    for truth, prediction, name in CELLS:
        selections = [(record, record["gt_valid"] & (record["targets"] == truth)
                       & (record["predictions"] == prediction)) for record in records.values()]
        record = speed_stats(records, selections)
        class_mass = sum(float((item["gt_valid"] & (item["targets"] == truth)).sum()
                               / max(int(item["gt_valid"].sum()), 1)) for item in records.values())
        record["fraction_of_gt_class_mass"] = record["track_weighted_mass"] / class_mass if class_mass else None
        result[name] = record
    return result


def conditional_availability_confusions(records):
    result = {}
    for flag in ("rgb_valid", "lidar_valid"):
        if not any(flag in record["source_flags"] for record in records.values()):
            continue
        by_truth = {}
        for truth, truth_name in enumerate(CLASSES):
            by_flag = {}
            for available, label in ((True, "valid"), (False, "missing")):
                frames, mass, tracks = 0, 0.0, set()
                predicted_mass = np.zeros(len(CLASSES), dtype=np.float64)
                for locator, record in records.items():
                    if flag not in record["source_flags"]:
                        continue
                    mask = (record["gt_valid"] & (record["targets"] == truth)
                            & (record["source_flags"][flag].astype(bool) == available))
                    if not mask.any():
                        continue
                    frames += int(mask.sum())
                    tracks.add(locator)
                    weight = 1.0 / max(int(record["gt_valid"].sum()), 1)
                    mass += int(mask.sum()) * weight
                    for pred in range(len(CLASSES)):
                        predicted_mass[pred] += int((mask & (record["predictions"] == pred)).sum()) * weight
                by_flag[label] = {"frames": frames, "tracks": len(tracks),
                                  "track_weighted_gt_mass": mass,
                                  "prediction_fraction": {name: float(predicted_mass[i] / mass) if mass else None
                                                          for i, name in enumerate(CLASSES)}}
            by_truth[truth_name] = by_flag
        result[flag] = by_truth
    return result


def sustained_road_stop_motion(records, predicted_class):
    """Six-slot centered displacement speed where all seven labels/positions are valid."""
    values, weights, elapsed = [], [], []
    centers, tracks = 0, set()
    gt_stop_mass = 0.0
    for locator, record in records.items():
        gt = record["gt_valid"] & (record["targets"] == 1)
        track_weight = 1.0 / max(int(record["gt_valid"].sum()), 1)
        gt_stop_mass += float(gt.sum()) * track_weight
        times = record["source_times_us"]
        if times is None or len(times) != len(gt):
            raise ValueError(f"Missing actual source timestamps for {locator}")
        time_valid = np.array([t is not None for t in times], dtype=bool)
        time_us = np.array([int(t) if t is not None else 0 for t in times], dtype=np.int64)
        for center in range(3, len(gt) - 3):
            window = slice(center - 3, center + 4)
            if not (gt[window].all() and record["ped_position_valid"][window].all()
                    and time_valid[center - 3] and time_valid[center + 3]):
                continue
            if int(record["predictions"][center]) != predicted_class:
                continue
            delta_us = int(time_us[center + 3] - time_us[center - 3])
            if delta_us <= 0:
                raise ValueError(f"Nonpositive observed center-window duration: {locator}")
            speed = float(np.linalg.norm(record["ped_position"][center + 3]
                                         - record["ped_position"][center - 3]) / (delta_us / 1e6))
            if not np.isfinite(speed):
                raise ValueError(f"Nonfinite sustained speed: {locator}")
            values.append(speed)
            weights.append(track_weight)
            elapsed.append(delta_us / 1e6)
            centers += 1
            tracks.add(locator)
    eligible_mass = float(np.sum(weights))
    result = {"eligible_centers": centers, "eligible_tracks": len(tracks),
              "eligible_track_weighted_mass": eligible_mass,
              "eligible_fraction_of_all_gt_stopped_mass": eligible_mass / gt_stop_mass if gt_stop_mass else None,
              "window_duration_seconds_median": float(np.median(elapsed)) if elapsed else None}
    if values:
        value_array, weight_array = np.asarray(values), np.asarray(weights)
        result.update(net_speed_median=weighted_quantile(value_array, weight_array, .5),
                      net_speed_p90=weighted_quantile(value_array, weight_array, .9),
                      net_speed_below_0_2_fraction=float(weight_array[value_array < .2].sum() / weight_array.sum()),
                      net_speed_above_0_5_fraction=float(weight_array[value_array > .5].sum() / weight_array.sum()))
    else:
        result.update(net_speed_median=None, net_speed_p90=None,
                      net_speed_below_0_2_fraction=None, net_speed_above_0_5_fraction=None)
    return result


def normalizations(aggregate):
    import torch
    by_key = {(row["training_source"], row["input"]): row for row in aggregate["models"]}
    result = {}
    for source in ("road-waymo", "loki"):
        row = by_key[(source, "geometry")]
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        baseline = data["normalization"]["baseline"]
        # K+T+R order: v_ped xy is columns 0:2; v_rel xy is columns 6:8.
        result[source] = {"v_ped_xy_mean": baseline["mean"][:2],
                          "v_ped_xy_scale": baseline["scale"][:2],
                          "v_rel_xy_mean": baseline["mean"][6:8],
                          "v_rel_xy_scale": baseline["scale"][6:8],
                          "checkpoint": row["checkpoint"],
                          "checkpoint_sha256": row["checkpoint_sha256"]}
    return result


def main():
    aggregate = json.loads((ROOT / "source-validation-metrics.json").read_text())
    manifest = json.loads(Path("experiments/E002-multimodal-reference/transfer-comparison.json").read_text())
    manifest_rows = {(row["training_source"], row["input"]): row for row in manifest["models"]}
    own, transfer = {}, {}
    for row in aggregate["models"]:
        key = (row["training_source"], row["input"])
        frozen = manifest_rows[key]
        if (row["checkpoint"] != frozen["checkpoint"]
                or row["checkpoint_sha256"] != frozen["checkpoint_sha256"]):
            raise ValueError(f"Source prediction lineage differs from frozen manifest: {key}")
        metrics_path = Path(row["metrics_path"])
        own_path = metrics_path.parent / "predictions.npz"
        own.setdefault(row["training_source"], {})[row["input"]] = read_predictions(own_path)
        target = "loki" if row["training_source"] == "road-waymo" else "road-waymo"
        transfer[row["training_source"] + "-to-" + target, row["input"]] = read_predictions(
            ROOT / "cells" / row["training_source"] / row["input"] / "predictions.npz")

    populations = {dataset: models["geometry"] for dataset, models in own.items()}
    for dataset, models in own.items():
        reference = models["geometry"]
        for input_name, records in models.items():
            check_aligned(reference, records, f"{dataset} own-validation/{input_name}")
    for (direction, input_name), records in transfer.items():
        target = direction.split("-to-")[1]
        check_aligned(populations[target], records, f"{direction}/{input_name}")

    cache = {}
    for records in populations.values():
        attach_track_arrays(records, cache)
    dataset_summary = {dataset: summarize_dataset(records) for dataset, records in populations.items()}

    direct = {}
    target_profiles = {}
    for (direction, input_name), records in transfer.items():
        attach_track_arrays(records, cache)
        direct.setdefault(direction, {})[input_name] = summarize_errors(records)
        if input_name == "geometry+rgb+lidar":
            target_profiles[direction] = summarize_dataset(records)

    adaptation = {}
    conditional_availability = {}
    for source, target in (("road-waymo", "loki"), ("loki", "road-waymo")):
        direction = source + "-to-" + target
        conditional_availability[direction] = conditional_availability_confusions(
            transfer[(direction, "geometry+rgb+lidar")])
        e003b_path = E003B / f"e002d-{direction}" / f"{target}-validation" / "predictions.npz"
        adapted = read_predictions(e003b_path)
        check_aligned(populations[target], adapted, f"E003b adapted target {direction}")
        attach_track_arrays(adapted, cache)
        baseline = transfer[(direction, "geometry+rgb+lidar")]
        check_aligned(baseline, adapted, f"E003b vs direct fusion target {direction}")
        adaptation[direction] = {"direct_fusion": summarize_errors(baseline),
                                 "adapted_frozen_body_new_head": summarize_errors(adapted),
                                 "adapted_predictions": str(e003b_path)}

    road_full_fusion = transfer[("loki-to-road-waymo", "geometry+rgb+lidar")]
    sustained_motion = {"false_moving_prediction": sustained_road_stop_motion(road_full_fusion, 0),
                        "correct_stopped_prediction": sustained_road_stop_motion(road_full_fusion, 1)}

    summary = {"schema_version": 1,
               "method": {"data": "saved E002 source-validation and direct opposite-source validation predictions, frozen E003b adapted-head target predictions, and their referenced unnormalized E001 track arrays",
                          "metric_weight": "each accepted GT frame weight 1 / total accepted GT frames in its track; class groups are not independently rebalanced",
                          "speed": "Euclidean norm of native ped_velocity; missing values excluded by ped_velocity_valid; <0.2 and >0.5 m/s are diagnostic bins, not label definitions",
                          "ego_speed": "Euclidean norm of ego_velocity, separately masked; no imputation",
                          "transition_codes": {str(i): name for i, name in enumerate(TRANSITIONS)},
                          "script": str(Path(__file__)),
                          "predictions": "eight own-source validation files from source-validation-metrics.json; eight opposite-source cells; two E003b frozen-body/new-head outputs",
                          "no_training_or_inference": True,
                          "limitations": ["one frozen validation population per dataset", "velocity and ego-motion summaries are correlates, not independently verified motion truth", "no causal or annotation-error attribution"]},
               "datasets": dataset_summary, "target_gt_profiles_with_modality_flags": target_profiles,
               "direct_transfer_errors": direct,
               "full_fusion_conditional_availability_confusions": conditional_availability,
               "adaptation_comparison": adaptation,
               "road_stopped_sustained_motion_1_2s": sustained_motion,
               "source_normalization": normalizations(aggregate)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "dataset_gt_classes": dataset_summary,
                      "fusion_adaptation_cells": {direction: {
                          "direct_crossing_to_moving": values["direct_fusion"]["CROSSING_TO_MOVING"]["fraction_of_gt_class_mass"],
                          "adapted_crossing_to_moving": values["adapted_frozen_body_new_head"]["CROSSING_TO_MOVING"]["fraction_of_gt_class_mass"],
                          "direct_stopped_to_moving": values["direct_fusion"]["STOPPED_TO_MOVING"]["fraction_of_gt_class_mass"],
                          "adapted_stopped_to_moving": values["adapted_frozen_body_new_head"]["STOPPED_TO_MOVING"]["fraction_of_gt_class_mass"]}
                          for direction, values in adaptation.items()}}, indent=2))


if __name__ == "__main__":
    main()
