"""Posthoc error strata for saved E004 assessment predictions; no inference."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pedestrian_behavior.evaluation import load_predictions, prediction_metrics


ROOT = Path("outputs/experiments")
ASSESSMENT = ROOT / "E004a/runs/20261113-082847-737608Z/zero-source-head/assessment-predictions.npz"
PREDICTIONS = {
    "road-direct": ROOT / "E002/transfer-validation-20261102/cells/road-waymo/geometry+rgb+lidar/predictions.npz",
    "road-head": ROOT / "E003c/runs/source-adaptation-20261106-121040/head/loki-validation/predictions.npz",
    "loki-trained": ROOT / "E002d/runs/fusion-20261030-140937-loki-rgbdrop50/loki-validation/predictions.npz",
}
SOURCE_VALIDATION = {
    "road-source": ROOT / "E002d/runs/fusion-20261030-140937-road-waymo-rgbdrop50/road-waymo-validation/predictions.npz",
    "loki-source": ROOT / "E002d/runs/fusion-20261030-140937-loki-rgbdrop50/loki-validation/predictions.npz",
}
OUTPUT = ROOT / "E004/track-difficulty-20261116"
CLASSES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
DURATION = ("[0,1)", "[1,5)", "[5,10)", "[10,15)", "[15,inf)")
RANGE = ("[0,10)m", "[10,20)m", "[20,40)m", "[40,inf)m", "unknown")
TRANSITION = ("near", "steady", "unknown")
EVIDENCE = ("[0,.25)", "[.25,.75)", "[.75,1]")
EXPECTED_ASSESSMENT = (931, 30605, 47)

MODEL_LABELS = {
    "road-direct": "ROAD direct",
    "road-head": "ROAD + full-label head",
    "loki-trained": "LOKI trained",
}
COLORS = {"road-direct": "#4477AA", "road-head": "#EE6677", "loki-trained": "#228833"}


def read(path):
    tracks = load_predictions(path)
    result = {}
    for track in tracks:
        locator = track["locator"]
        if locator in result:
            raise ValueError(f"Duplicate track locator in {path}: {locator}")
        gt = np.asarray(track["gt_valid"], dtype=bool)
        if not gt.any() or len(track["targets"]) != len(gt):
            raise ValueError(f"Invalid scored track in {path}: {locator}")
        result[locator] = track
    return result


def check_aligned(reference, candidate, name):
    missing = reference.keys() - candidate.keys()
    if missing:
        raise ValueError(f"Tracks missing for {name}: {len(missing)}")
    for locator, first in reference.items():
        second = candidate[locator]
        for field in ("targets", "gt_valid"):
            if not np.array_equal(first[field], second[field]):
                raise ValueError(f"GT alignment differs for {name}: {locator}/{field}")
        first_times, second_times = first.get("source_times_us"), second.get("source_times_us")
        if first_times is not None and second_times is not None and first_times != second_times:
            raise ValueError(f"Timestamp alignment differs for {name}: {locator}")


def scenario(locator):
    metadata = json.loads(locator)
    value = metadata.get("scenario")
    if value is None:
        raise ValueError(f"Expected a LOKI scenario locator: {locator}")
    return value


def scalar_condition(track, axis, labels):
    values = np.unique(np.asarray(track["conditions"][axis])[track["gt_valid"]])
    if len(values) != 1 or not 0 <= int(values[0]) < len(labels):
        raise ValueError(f"Expected one valid {axis} value per track: {track['locator']}")
    return int(values[0])


def evidence_bin(fraction):
    return int(np.searchsorted((0.25, 0.75), fraction, side="right"))


def aligned_weighted_metrics(records, masks, total_error_mass):
    """Keep each selected frame's original 1 / whole-track-GT weight."""
    gt_frames = raw_errors = tracks = 0
    weighted_support = weighted_errors = 0.0
    groups = set()
    for locator, mask in masks.items():
        if locator not in records:
            continue
        track = records[locator]
        gt = np.asarray(track["gt_valid"], dtype=bool)
        selected = np.asarray(mask, dtype=bool) & gt
        count = int(gt.sum())
        if not selected.any():
            continue
        target = np.asarray(track["targets"])
        prediction = np.asarray(track["logits"]).argmax(axis=1)
        wrong = selected & (target != prediction)
        tracks += 1
        groups.add(scenario(locator))
        gt_frames += int(selected.sum())
        raw_errors += int(wrong.sum())
        weighted_support += float(selected.sum()) / count
        weighted_errors += float(wrong.sum()) / count
    return {
        "tracks": tracks,
        "groups": len(groups),
        "gt_frames": gt_frames,
        "weighted_gt_track_equivalent": weighted_support,
        "raw_errors": raw_errors,
        "weighted_error_contribution_track_equivalent": weighted_errors,
        "weighted_error_rate": weighted_errors / weighted_support if weighted_support else None,
        "raw_error_rate": raw_errors / gt_frames if gt_frames else None,
        "share_of_total_weighted_error_pct": (100 * weighted_errors / total_error_mass
                                                if total_error_mass else None),
    }


def masks_for_tracks(records, locators):
    return {locator: np.ones(len(records[locator]["targets"]), dtype=bool) for locator in locators}


def add_row(rows, view, stratum, model, metrics):
    rows.append({"view": view, "stratum": stratum, "model": model, **metrics})


def make_data():
    assessment = read(ASSESSMENT)
    locators = sorted(assessment)
    full_predictions = {name: read(path) for name, path in PREDICTIONS.items()}
    for name, records in full_predictions.items():
        check_aligned(assessment, records, name)
    predictions = {name: {locator: records[locator] for locator in locators}
                   for name, records in full_predictions.items()}
    reference = predictions["road-direct"]
    for name, records in predictions.items():
        check_aligned(reference, records, name)
    gt_frames = sum(int(assessment[k]["gt_valid"].sum()) for k in locators)
    groups = {scenario(k) for k in locators}
    if (len(locators), gt_frames, len(groups)) != EXPECTED_ASSESSMENT:
        raise ValueError(f"Unexpected E004 assessment population: {(len(locators), gt_frames, len(groups))}")
    class_support = [sum(int(((assessment[k]["targets"] == c) & assessment[k]["gt_valid"]).sum())
                         for k in locators) for c in range(len(CLASSES))]
    if not all(class_support):
        raise ValueError(f"Assessment lacks a GT class: {class_support}")

    overall_masks = {name: masks_for_tracks(records, locators) for name, records in predictions.items()}
    overall = {name: aligned_weighted_metrics(records, overall_masks[name], 0)
               for name, records in predictions.items()}
    total_error_mass = {name: values["weighted_error_contribution_track_equivalent"]
                        for name, values in overall.items()}
    for name in predictions:
        overall[name]["share_of_total_weighted_error_pct"] = 100.0 if total_error_mass[name] else None

    track_features = {}
    for locator in locators:
        track = reference[locator]
        duration_code = scalar_condition(track, "duration", DURATION)
        rgb = np.asarray(track["source_flags"]["rgb_valid"], dtype=bool)
        lidar = np.asarray(track["source_flags"]["lidar_valid"], dtype=bool)
        duration_us = track.get("native_extent_us")
        if duration_us is None or len(duration_us) != 2:
            raise ValueError(f"Missing native extent for {locator}")
        track_features[locator] = {
            "duration_code": duration_code,
            "duration_seconds": (int(duration_us[1]) - int(duration_us[0])) / 1e6,
            "rgb_fraction": float(rgb.mean()),
            "rgb_bin": evidence_bin(float(rgb.mean())),
            "lidar_fraction": float(lidar.mean()),
            "lidar_bin": evidence_bin(float(lidar.mean())),
            "scenario": scenario(locator),
        }

    strata_rows = []
    for model in predictions:
        add_row(strata_rows, "overall", "all", model, overall[model])
    figures = {}

    # Duration and sensor evidence are track cohorts. Distance and transition
    # are timestep cohorts, with the original whole-track weight retained.
    specs = (
        ("duration", DURATION, lambda loc: track_features[loc]["duration_code"], True),
        ("ped-ego-range", RANGE,
         lambda loc: np.asarray(reference[loc]["conditions"]["range"]), False),
        ("rgb-fraction", EVIDENCE, lambda loc: track_features[loc]["rgb_bin"], True),
        ("lidar-fraction", EVIDENCE, lambda loc: track_features[loc]["lidar_bin"], True),
    )
    for feature, labels, get_value, track_level in specs:
        group_summaries = {}
        for code, label in enumerate(labels):
            masks = {}
            for locator in locators:
                value = get_value(locator)
                masks[locator] = (np.full(len(assessment[locator]["targets"]), value == code, dtype=bool)
                                  if track_level else np.asarray(value) == code)
            group_summaries[label] = {}
            for model, records in predictions.items():
                metrics = aligned_weighted_metrics(records, masks, total_error_mass[model])
                group_summaries[label][model] = metrics
                add_row(strata_rows, feature, label, model, metrics)
        figures[feature] = group_summaries

    class_summary = {}
    class_feature_summary = {}
    transition_summary = {}
    for class_id, class_name in enumerate(CLASSES):
        class_summary[class_name] = {}
        class_feature_summary[class_name] = {}
        transition_summary[class_name] = {}
        for feature, labels, get_value, track_level in specs:
            class_feature_summary[class_name][feature] = {}
            for code, label in enumerate(labels):
                masks = {}
                for locator in locators:
                    track = assessment[locator]
                    class_mask = ((np.asarray(track["targets"]) == class_id)
                                  & np.asarray(track["gt_valid"], dtype=bool))
                    value = get_value(locator)
                    cohort = (np.full(len(track["targets"]), value == code, dtype=bool)
                              if track_level else np.asarray(value) == code)
                    masks[locator] = class_mask & cohort
                class_feature_summary[class_name][feature][label] = {}
                for model, records in predictions.items():
                    metrics = aligned_weighted_metrics(records, masks, total_error_mass[model])
                    class_feature_summary[class_name][feature][label][model] = metrics
                    add_row(strata_rows, f"{class_name}/{feature}", label, model, metrics)
        for transition_id, transition_name in enumerate(TRANSITION):
            masks = {}
            for locator in locators:
                track = assessment[locator]
                masks[locator] = ((np.asarray(track["targets"]) == class_id)
                                  & np.asarray(track["gt_valid"], dtype=bool)
                                  & (np.asarray(reference[locator]["conditions"]["transition"]) == transition_id))
            transition_summary[class_name][transition_name] = {}
            for model, records in predictions.items():
                metrics = aligned_weighted_metrics(records, masks, total_error_mass[model])
                transition_summary[class_name][transition_name][model] = metrics
                add_row(strata_rows, f"{class_name}/transition", transition_name, model, metrics)

        masks = {locator: ((np.asarray(assessment[locator]["targets"]) == class_id)
                           & np.asarray(assessment[locator]["gt_valid"], dtype=bool))
                 for locator in locators}
        for model, records in predictions.items():
            class_summary[class_name][model] = aligned_weighted_metrics(
                records, masks, total_error_mass[model])
            add_row(strata_rows, "gt-class", class_name, model, class_summary[class_name][model])

    track_rows = []
    for locator in locators:
        base = track_features[locator]
        target = np.asarray(assessment[locator]["targets"])
        gt = np.asarray(assessment[locator]["gt_valid"], dtype=bool)
        for model, records in predictions.items():
            record = records[locator]
            prediction = np.asarray(record["logits"]).argmax(axis=1)
            row = {
                "locator": locator,
                "scenario": base["scenario"],
                "model": model,
                "gt_frames": int(gt.sum()),
                "duration_seconds_nominal": base["duration_seconds"],
                "duration_bin": DURATION[base["duration_code"]],
                "rgb_evidence_fraction_all_slots": base["rgb_fraction"],
                "rgb_bin": EVIDENCE[base["rgb_bin"]],
                "lidar_evidence_fraction_all_slots": base["lidar_fraction"],
                "lidar_bin": EVIDENCE[base["lidar_bin"]],
                "raw_errors": int((gt & (target != prediction)).sum()),
                "track_error_rate": float((gt & (target != prediction)).sum() / gt.sum()),
            }
            for class_id, class_name in enumerate(CLASSES):
                class_gt = gt & (target == class_id)
                row[f"gt_{class_name.lower()}_frames"] = int(class_gt.sum())
                row[f"errors_{class_name.lower()}_frames"] = int((class_gt & (target != prediction)).sum())
            track_rows.append(row)

    source_context = {}
    for name, path in SOURCE_VALIDATION.items():
        source = list(read(path).values())
        metric = prediction_metrics(source, len(CLASSES))
        source_context[name] = {
            "tracks": metric["denominators"]["tracks"],
            "groups": metric["denominators"]["groups"],
            "gt_frames": metric["denominators"]["gt_frames"],
            "track_weighted_macro_f1": metric["track_weighted"]["macro_f1"],
            "track_ce": metric["track_ce"],
        }

    return {
        "population": {"tracks": len(locators), "groups": len(groups), "gt_frames": gt_frames,
                       "gt_class_frames": dict(zip(CLASSES, class_support))},
        "models": {
            name: {
                "label": MODEL_LABELS[name],
                "architecture": "E002d two-layer RGB + LiDAR + geometry Transformer, 128-wide temporal representation",
                "selection_access": ("direct source-selected checkpoint" if name == "road-direct" else
                                     "selected after full LOKI validation; this common subset was included in selection"),
                **overall[name],
                "track_weighted_macro_f1": prediction_metrics(
                    list(records[k] for k in locators), len(CLASSES))["track_weighted"]["macro_f1"],
            }
            for name, records in predictions.items()
        },
        "strata": figures,
        "gt_class_error": class_summary,
        "gt_class_feature_error": class_feature_summary,
        "gt_class_transition_error": transition_summary,
        "own_source_validation_context_not_same_population": source_context,
        "weighting": ("Primary endpoint is track-weighted timestep error (weighted misclassification rate); "
                      "macro-F1 is descriptive. All subgroup rates use each scored timestep's original weight "
                      "1 / total accepted GT frames in its complete track. Slices retain that whole-track "
                      "weight; they are not reweighted to each slice's frame count per track. Weighted error "
                      "contribution is the sum of weighted incorrect timesteps, in track-equivalent units."),
        "limitations": [
            "The 931 tracks are the E004a assessment identities, but E003c head and LOKI-trained E002d checkpoints were selected using all 1,861 LOKI validation tracks, including these tracks; this is descriptive posthoc analysis, not independent assessment.",
            "E004a's saved assessment predictions omit source timestamps; alignment to the three model archives uses exact track locator, track length, targets and GT mask. The three model archives also agree on saved timestamps.",
            "Difficulty strata and usable-input fractions come from the ROAD-direct archive, which retains the required E001 condition arrays; some optional condition arrays are absent from the E003c head archive.",
            "Track duration uses the saved LOKI native extent timestamps, whose physical timing is not independently verified; interpret duration as nominal recorded span.",
            "RGB and LiDAR fractions are actual usable model-input flags over all unpadded track slots, not raw sensor-stream availability.",
            "Pedestrian–ego range and near/steady transition use the existing E001 saved condition bins; they are associations, not causal effects or explanations of annotation differences.",
            "ROAD→LOKI uses the original E002d source checkpoint; full-target-label head adaptation uses E003c's same E002d body; LOKI-trained reference is the E002d LOKI source run. Single selected checkpoints; no retraining or inference.",
        ],
        "track_rows": track_rows,
        "summary_rows": strata_rows,
    }


def figure_difficulty(summary, path):
    panel_specs = (
        ("duration", "Nominal recorded track span", DURATION),
        ("ped-ego-range", "Pedestrian–ego distance (m)", RANGE),
        ("rgb-fraction", "Usable RGB fraction of track slots", EVIDENCE),
        ("lidar-fraction", "Usable LiDAR fraction of track slots", EVIDENCE),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, (feature, title, labels) in zip(axes.flat, panel_specs):
        values = summary["strata"][feature]
        supports = [max((values[label][m]["tracks"] for m in MODEL_LABELS), default=0)
                    for label in labels]
        x = np.arange(len(labels))
        for model in MODEL_LABELS:
            y = [values[label][model]["weighted_error_rate"] * 100
                 if values[label][model]["weighted_error_rate"] is not None else np.nan
                 for label in labels]
            ax.plot(x, y, marker="o", linewidth=1.8, label=MODEL_LABELS[model], color=COLORS[model])
        ax.set_title(title)
        ax.set_xticks(x, [f"{label}\n{n} tracks" for label, n in zip(labels, supports)], fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Track-weighted error (%)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("LOKI E004 assessment subset: error by track and input conditions\n"
                 "Model-selection exposure includes these tracks for head-adapted and LOKI-trained models",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def figure_transition(summary, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    x = np.arange(len(MODEL_LABELS))
    width = 0.34
    for class_id, (ax, class_name) in enumerate(zip(axes.flat, CLASSES)):
        values = summary["gt_class_transition_error"][class_name]
        for offset, transition in ((-width / 2, "near"), (width / 2, "steady")):
            y = [values[transition][model]["weighted_error_rate"] * 100
                 if values[transition][model]["weighted_error_rate"] is not None else np.nan
                 for model in MODEL_LABELS]
            ax.bar(x + offset, y, width, label=transition,
                   color="#CC6677" if transition == "near" else "#4477AA")
        ax.set_title(class_name.replace("_", " ").title())
        ax.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_LABELS], rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Within-class track-weighted error (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("GT-class error near versus steady label transitions\n"
                 "Near = within 0.4 s of a GT transition; unknown support is retained in JSON/CSV",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(data):
    OUTPUT.mkdir(parents=True, exist_ok=False)
    summary_rows = data.pop("summary_rows")
    track_rows = data.pop("track_rows")
    (OUTPUT / "summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    for name, rows in (("summary.csv", summary_rows), ("track-errors.csv", track_rows)):
        with (OUTPUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    figure_difficulty(data, OUTPUT / "difficulty.png")
    figure_transition(data, OUTPUT / "class-transition.png")


def self_check():
    # A two-track slice keeps the whole-track denominators: 1/2 + 1/4 errors.
    def track(locator, target, prediction, gt):
        return {"locator": locator, "targets": np.asarray(target), "gt_valid": np.asarray(gt, bool),
                "logits": np.eye(2)[np.asarray(prediction)]}

    records = {
        '{"scenario":"a"}': track("a", [0, 0], [1, 0], [1, 1]),
        '{"scenario":"b"}': track("b", [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]),
    }
    masks = {key: np.asarray([True] + [False] * (len(value["targets"]) - 1))
             for key, value in records.items()}
    metrics = aligned_weighted_metrics(records, masks, 1.0)
    assert metrics["tracks"] == 2 and metrics["gt_frames"] == 2
    assert np.isclose(metrics["weighted_gt_track_equivalent"], 0.75)
    assert np.isclose(metrics["weighted_error_contribution_track_equivalent"], 0.75)
    assert np.isclose(metrics["weighted_error_rate"], 1.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("Track-weighted slice checks passed")
    else:
        write_outputs(make_data())
        print(f"Wrote E004 track-difficulty summary to {OUTPUT}")
