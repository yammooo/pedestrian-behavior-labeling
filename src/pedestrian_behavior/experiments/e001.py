"""E001 native audit, temporary targets, eligibility and frozen setup (no training)."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from pedestrian_behavior.data.features import FEATURE_SETS, features, fit_normalization
from pedestrian_behavior.data.loading import TrackDataset
from pedestrian_behavior.data.preparation import checksum, load_manifest, load_track, locator, source_times
from pedestrian_behavior.data.splits import split_groups
from pedestrian_behavior.datasets import loki, road_waymo
from pedestrian_behavior.models.kinematic import KinematicClassifier

CLASSES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
MODEL_SETTINGS = {"embedding_dim": 64, "dropout": 0.1, "recurrent_hidden": 32}
MAPPING = {
    "road-waymo": {"Mov": 0, "MovAway": 0, "MovTow": 0, "Stop": 1, "Wait2X": 2,
                   "Xing": 3, "XingFmLft": 3, "XingFmRht": 3},
    "loki": {"Moving": 0, "Stopped": 1, "Waiting to cross": 2, "Crossing the road": 3},
}


def build_model(configuration, variant):
    """E001 A: framewise; B: one-layer BiLSTM; independent weights on each call."""
    if configuration not in FEATURE_SETS or variant not in ("A", "B"):
        raise ValueError("Unknown E001 feature configuration or model variant")
    return KinematicClassifier(3*len(FEATURE_SETS[configuration]), MODEL_SETTINGS["embedding_dim"],
                               len(CLASSES), MODEL_SETTINGS["dropout"],
                               recurrent_hidden=MODEL_SETTINGS["recurrent_hidden"] if variant == "B" else None)


def mapped_target(actions, dataset):
    if not isinstance(actions, list) or any(not isinstance(a, str) for a in actions):
        raise ValueError("Native actions must be a list of strings")
    states = {MAPPING[dataset][a] for a in actions if a in MAPPING[dataset]}
    if states == {0, 3}:
        return 3
    if len(states) > 1:
        raise ValueError(f"Unresolved mapped-state conflict: {actions}")
    return next(iter(states), -100)


def override_for(observation, overrides):
    """Both native frame and timestamp bounds must agree, with exact actions."""
    matches = []
    for override in overrides:
        if (observation["clip"], observation["tube_uid"]) != (override["clip"], override["tube_uid"]):
            continue
        in_frame = override["road_frames_inclusive"][0] <= observation["frame"] <= override["road_frames_inclusive"][1]
        in_time = override["timestamps_inclusive_us"][0] <= observation["timestamp"] <= override["timestamps_inclusive_us"][1]
        if in_frame != in_time:
            raise ValueError("Override frame/timestamp scope mismatch")
        if in_frame:
            if sorted(observation["actions"]) != sorted(override["expected_native_actions"]):
                raise ValueError("Override native actions mismatch")
            matches.append(override)
    if len(matches) > 1:
        raise ValueError("Overlapping overrides")
    return matches[0] if matches else None


def audit_native(manifest, native_input, override_path):
    """Audit complete native supervision before inspecting the eligible saved view."""
    dataset = manifest["dataset"]
    action_sets, class_counts, track_annotations = Counter(), Counter(), Counter()
    evidence = {"dataset": dataset, "native_input": str(Path(native_input).resolve()),
                "observations": 0, "duplicate_annotations": 0, "verified_overrides": []}
    def count(actions, target, address):
        action_sets[json.dumps(sorted(actions))] += 1
        class_counts[str(target)] += 1
        evidence["observations"] += 1
        track_annotations[address] += 1
    if dataset == "road-waymo":
        for name, expected in manifest["sources"]["sha256"].items():
            if checksum(Path(native_input) / name) != expected:
                raise ValueError(f"Native source checksum mismatch: {name}")
        policy = json.loads(Path(override_path).read_text())
        if (policy.get("schema_version") != 1 or policy.get("experiment") != "E001"
                or policy.get("dataset") != dataset or policy["native_csv_sha256"] != manifest["sources"]["sha256"]["pedestrians.csv.gz"]):
            raise ValueError("Override policy/source checksum mismatch")
        overrides = policy["overrides"]
        for o in overrides:
            frames, times = o["road_frames_inclusive"], o["timestamps_inclusive_us"]
            if (len(frames) != 2 or len(times) != 2 or frames[0] > frames[1] or times[0] > times[1]
                    or o["observation_count"] != frames[1]-frames[0]+1 or o["target"] not in CLASSES):
                raise ValueError("Invalid accepted override bounds/count/target")
        seen, matches = {}, [[] for _ in overrides]
        for row in road_waymo.annotation_rows(Path(native_input)):
            clip, tube = row["road_clip_id"], row["road_tube_uid"]
            if clip not in manifest["scenes"]:
                continue
            if row["merged_agent_label"] != "Ped" or row["camera_name"] != "1":
                raise ValueError("Native audit population mismatch")
            observation = {"clip": clip, "tube_uid": tube, "frame": int(row["road_frame_1based"]),
                           "timestamp": int(row["frame_timestamp_micros"]),
                           "actions": json.loads(row["action_labels_json"])}
            key = clip, tube, observation["timestamp"]
            signature = observation["frame"], tuple(sorted(observation["actions"]))
            if key in seen:
                if seen[key] != signature:
                    raise ValueError("Conflicting repeated native supervision")
                evidence["duplicate_annotations"] += 1
                # Repeated original annotation IDs remain part of override evidence.
                for verified in matches:
                    for item in verified:
                        if (item["clip"], item["tube_uid"], item["timestamp"]) == key:
                            item["annotation_ids"].append(row["road_annotation_id"])
                continue
            seen[key] = signature
            override = override_for(observation, overrides)
            if override is not None:
                target = CLASSES.index(override["target"])
                matches[overrides.index(override)].append(observation | {"target": target, "annotation_ids": [row["road_annotation_id"]]})
            else:
                target = mapped_target(observation["actions"], dataset)
            count(observation["actions"], target, locator(clip=clip, tube_uid=tube))
        for override, observations in zip(overrides, matches):
            if (len(observations) != override["observation_count"]
                    or sorted(o["frame"] for o in observations) != list(range(override["road_frames_inclusive"][0], override["road_frames_inclusive"][1]+1))
                    or [min((o["timestamp"] for o in observations), default=None), max((o["timestamp"] for o in observations), default=None)] != override["timestamps_inclusive_us"]):
                raise ValueError("Override identity/bounds/count mismatch")
        evidence["verified_overrides"] = [o for observations in matches for o in observations]
        evidence["policy"] = {"path": str(Path(override_path).resolve()), "sha256": checksum(override_path), "accepted": policy}
        evidence["source_sha256"] = manifest["sources"]["sha256"]
    elif dataset == "loki":
        reader = loki.TrackReader(native_input)
        verified_scenes = {}
        for scene, provenance in manifest["scenes"].items():
            actual = reader.index_scene(scene)["provenance"]["annotation_odometry_sha256"]
            if actual != provenance["annotation_odometry_sha256"]:
                raise ValueError(f"Native source checksum mismatch: {scene}")
            verified_scenes[scene] = actual
            for frame in loki.frames(Path(native_input) / scene):
                for person in frame.pedestrians.values():
                    if person.label3d is not None:
                        actions = [person.action] if person.action else []
                        count(actions, mapped_target(actions, dataset), locator(scenario=scene, track_id=person.track_id))
                        evidence["duplicate_annotations"] += len(person.label3d_record_indices)-1
        evidence["source_sha256"] = verified_scenes
        evidence["policy"] = {"mapping": MAPPING[dataset], "future_target_shift": False}
    else:
        raise ValueError("Unsupported E001 dataset")
    evidence["action_sets"] = dict(sorted(action_sets.items()))
    evidence["class_frames"] = {name: class_counts[str(i)] for i, name in enumerate(CLASSES)}
    evidence["unmapped_frames"] = class_counts["-100"]
    evidence["track_annotations"] = dict(track_annotations)
    return evidence


def targets(metadata, dataset, verified_overrides=()):
    identity = json.loads(metadata["track_locator"])
    verified = {(o["clip"], o["tube_uid"], o["timestamp"]): o for o in verified_overrides}
    result = np.full(len(metadata["native_annotations"]), -100, dtype=np.int64)
    for i, (annotations, time) in enumerate(zip(metadata["native_annotations"], source_times(metadata))):
        if dataset == "road-waymo":
            road = annotations.get("road")
            actions = road["actions"] if road else []
            override = verified.get((identity["clip"], identity["tube_uid"], time))
            if override is not None:
                if (road is None or sorted(actions) != sorted(override["actions"])
                        or sorted(road["annotation_ids"]) != sorted(override["annotation_ids"])
                        or road["annotation"].get("tube_uid") != identity["tube_uid"]):
                    raise ValueError("Saved override source/scope mismatch")
                result[i] = override["target"]
                continue
        elif dataset == "loki":
            action = annotations.get("label3d", {}).get("intended_actions")
            actions = [action] if action else []
        else:
            raise ValueError("Unsupported E001 dataset")
        result[i] = mapped_target(actions, dataset)
    return result, result != -100


def exclusion_reasons(metadata, arrays, gt_valid):
    return (["no-usable-position"] if not arrays["ped_position_valid"].any() else []) + (
        ["no-accepted-gt"] if not gt_valid.any() else []) + (["invalid-anchor"] if not metadata["anchor_valid"] else [])


def setup(collection, native_input, output, override_path):
    collection, output = Path(collection), Path(output)
    started = time.monotonic()
    manifest = load_manifest(collection)
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "experiment": "E001", "status": "incomplete", "dataset": manifest["dataset"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.orig_argv, "output": str(output.resolve()),
              "collection": {"path": str(collection.resolve()), "manifest_sha256": checksum(collection / "manifest.json"),
                             "preparation_id": manifest["preparation_id"]},
              "policy": {"classes": CLASSES, "mapping": MAPPING[manifest["dataset"]], "ignore_index": -100},
              "environment": {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                              "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
                              "code_sha256": {str(p): checksum(p) for p in sorted(Path("src").rglob("*.py"))},
                              "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                              "cuda_build": torch.version.cuda, "host": platform.node()},
              "tracks": [], "failures": []}
    def save():
        (output / "setup.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    save()
    try:
        record["native_audit"] = audit_native(manifest, native_input, override_path)
        exclusions, combinations = Counter(), Counter()
        for filename in manifest["tracks"]:
            metadata, arrays = load_track(collection / "tracks" / filename)
            if metadata["native_counts"]["annotations"] != record["native_audit"]["track_annotations"].get(metadata["track_locator"], 0):
                raise ValueError("Native/saved track supervision count mismatch")
            y, gt = targets(metadata, manifest["dataset"], record["native_audit"]["verified_overrides"])
            reasons = exclusion_reasons(metadata, arrays, gt)
            identity = json.loads(metadata["track_locator"])
            group = identity["clip"] if manifest["dataset"] == "road-waymo" else identity["scenario"]
            entry = {"archive": filename, "locator": metadata["track_locator"], "group": group,
                     "exclusions": reasons, "slots": len(y), "gt_frames": int(gt.sum()),
                     "class_frames": [int((y == i).sum()) for i in range(4)],
                     "has_2d": any(a.get("loki_2d") or a.get("road_2d") or a.get("waymo_2d") for a in metadata["availability"])}
            record["tracks"].append(entry)
            exclusions.update(reasons)
            if reasons:
                combinations[" + ".join(reasons)] += 1
        eligible = [t for t in record["tracks"] if not t["exclusions"]]
        if len({t["locator"] for t in record["tracks"]}) != len(record["tracks"]):
            raise ValueError("Repeated saved identity")
        if set(record["native_audit"]["track_annotations"]) - {t["locator"] for t in record["tracks"]}:
            raise ValueError("Native supervision identity missing from saved collection")
        record["groups"] = split_groups(t["group"] for t in eligible)
        record["split_policy"] = {"unit": "clip/scenario with eligible tracks", "seed": 0,
                                  "algorithm": "sorted IDs; fresh Python random.Random(0); ceil(0.15*N) validation then test; remainder training"}
        for entry in record["tracks"]:
            entry["split"] = record["groups"][entry["group"]] if not entry["exclusions"] else None
        record["population"] = {"candidates": len(record["tracks"]), "eligible": len(eligible),
                                "excluded": len(record["tracks"])-len(eligible), "exclusion_reasons": dict(exclusions),
                                "exclusion_combinations": dict(combinations), "eligible_3d_only": sum(not t["has_2d"] for t in eligible)}
        record["support"] = {}
        for split in ("training", "validation", "test"):
            entries = [t for t in eligible if t["split"] == split]
            record["support"][split] = {"groups": sum(s == split for s in record["groups"].values()), "tracks": len(entries),
                                        "slots": sum(t["slots"] for t in entries), "gt_frames": sum(t["gt_frames"] for t in entries),
                                        "class_frames": [sum(t["class_frames"][i] for t in entries) for i in range(4)],
                                        "class_tracks": [sum(t["class_frames"][i] > 0 for t in entries) for i in range(4)]}
        training = [t for t in eligible if t["split"] == "training"]
        def samples(configuration):
            for entry in training:
                _, arrays = load_track(collection / "tracks" / entry["archive"])
                yield features(arrays, configuration)
        record["normalization"] = {configuration: fit_normalization(samples(configuration)) for configuration in FEATURE_SETS}
        record["status"] = "complete"
    except Exception as error:
        record["failures"].append(str(error))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = time.monotonic() - started
        save()
    return record


def dataset_from_setup(collection, setup_path, split, configuration, statistics):
    """Supply the source's statistics explicitly, including for target evaluation."""
    record = json.loads(Path(setup_path).read_text())
    if record.get("schema_version") != 1 or record.get("experiment") != "E001" or record.get("status") != "complete":
        raise ValueError("Unsupported or incomplete E001 setup")
    if split not in ("training", "validation", "test"):
        raise ValueError("Invalid split")
    if checksum(Path(collection) / "manifest.json") != record["collection"]["manifest_sha256"]:
        raise ValueError("Setup/collection checksum mismatch")
    entries = [t for t in record["tracks"] if t["split"] == split and not t["exclusions"]]
    policy = partial(targets, dataset=record["dataset"], verified_overrides=record["native_audit"]["verified_overrides"])
    return TrackDataset(collection, entries, configuration, statistics, policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--native-input", type=Path, required=True, help="LOKI root or ROAD export index; used only for setup audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, default=Path("experiments/E001-kinematic-transfer/behavior-overrides.json"))
    args = parser.parse_args()
    record = setup(args.collection, args.native_input, args.output, args.overrides)
    print(json.dumps({k: record[k] for k in ("population", "support")}, indent=2))
    print("Guarded columns:", {k: v["guarded_columns"] for k, v in record["normalization"].items()})


if __name__ == "__main__":
    main()
