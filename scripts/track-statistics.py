"""Annotation-only, clip-scoped track extents; internal missing frames stay in context."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from pedestrian_behavior.datasets import loki


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    # Empirical inverse CDF (nearest rank); no interpolated fictitious track.
    result = {f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)]
              for p in (25, 50, 75, 90, 95, 99)}
    return dict(n=len(values), min=values[0], mean=statistics.mean(values),
                **result, max=values[-1])


def extent(frames):
    frames = sorted(set(frames))
    if not frames:
        return dict(observed=0, span=0, missing=0, max_gap=0, gap_runs=0)
    gaps = [b - a - 1 for a, b in zip(frames, frames[1:])]
    span = frames[-1] - frames[0] + 1
    return dict(observed=len(frames), span=span, missing=span - len(frames),
                max_gap=max(gaps, default=0), gap_runs=sum(g > 0 for g in gaps))


def record(clip, track, observations, hz, split):
    frames = sorted(observations)
    stats = extent(frames)
    duration = observations[frames[-1]]["time"] - observations[frames[0]]["time"]
    if duration < 0:
        raise ValueError("Non-monotonic track timestamps")
    result = dict(clip=clip, track=track, split=split, first_frame=frames[0],
                  last_frame=frames[-1], duration_seconds=duration,
                  # Conservative capacity of a track-start-aligned 5 Hz envelope.
                  context_steps_5hz=math.ceil(duration * 5 - 1e-8) + 1,
                  native_hz=hz, **stats)
    result["max_gap_boundary_interval_seconds"] = max(
        (observations[b]["time"] - observations[a]["time"]
         for a, b in zip(frames, frames[1:]) if b - a > 1), default=0)
    for modality in ("2d", "3d", "paired", "behavior"):
        selected = [f for f in frames if observations[f][modality]]
        for name, value in extent(selected).items():
            result[f"{modality}_{name}"] = value
        result[f"{modality}_missing_in_full_span"] = stats["span"] - len(selected)
    result["actions"] = sorted(set().union(*(o["actions"] for o in observations.values())))
    return result


def scan_loki(root):
    records, cadence = [], Counter()
    inventory = hashlib.sha256()
    scenes, total_frames = 0, 0
    for scene in loki.scenarios(root):
        scenes += 1
        tracks = defaultdict(dict)
        frames = list(loki.frame_files(scene))
        ids = sorted(int(frame) for frame, _ in frames)
        cadence.update(b - a for a, b in zip(ids, ids[1:]))
        # This release uses suffix 0,2,4,... at documented annotation rate 5 Hz.
        if any(f % 2 for f in ids) or any(b - a != 2 for a, b in zip(ids, ids[1:])):
            raise ValueError(f"{scene}: unexpected filename cadence; verify timing")
        for frame_id, paths in frames:
            total_frames += 1
            for kind in ("label2d", "label3d"):
                if paths[kind] is None:
                    raise ValueError(f"{scene}/{frame_id}: missing {kind}")
                inventory.update(str(paths[kind].relative_to(root)).encode())
                inventory.update(paths[kind].read_bytes())
            frame = loki.read_frame(scene, frame_id, paths)
            f = int(frame_id) // 2
            for person in frame.pedestrians.values():
                two, three = person.label2d is not None, person.label3d is not None
                action = person.action
                tracks[person.track_id][f] = dict(time=f / 5, **{
                    "2d": two, "3d": three, "paired": two and three,
                    "behavior": bool(action), "actions": {action} if action else set()})
        records.extend(record(scene.name, t, obs, 5, "unknown") for t, obs in tracks.items())
    return records, dict(root=str(root.resolve()), scenarios=scenes, frames=total_frames,
                         filename_step_counts=dict(cadence),
                         annotation_content_digest=inventory.hexdigest(),
                         timing="Nominal 5 Hz from published annotation cadence; suffix / 2 is frame ordinal. No physical timestamps verified.")


def scan_road(index):
    tracks, clip_times, splits = defaultdict(dict), defaultdict(dict), {}
    rows, duplicates = 0, 0
    source = index / "pedestrians.csv.gz"
    with gzip.open(source, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            if row["merged_agent_label"] != "Ped" or row["has_3d_box"] not in ("True", "False"):
                raise ValueError("Invalid authoritative class or 3D mask")
            clip, track = row["road_clip_id"], row["road_tube_uid"]
            key = clip, track
            f, ts = int(row["road_frame_1based"]), int(row["frame_timestamp_micros"])
            if f in clip_times[clip] and clip_times[clip][f] != ts:
                raise ValueError("Conflicting frame timestamps")
            clip_times[clip][f] = ts
            if key in splits and splits[key] != row["road_split"]:
                raise ValueError("Conflicting splits")
            splits[key] = row["road_split"]
            actions = set(json.loads(row["action_labels_json"]))
            paired = row["has_3d_box"] == "True"
            obs = dict(time=ts, **{"2d": True, "3d": paired, "paired": paired,
                                   "behavior": bool(actions), "actions": actions})
            if f in tracks[key]:
                if tracks[key][f] != obs:
                    raise ValueError("Conflicting duplicate observation")
                duplicates += 1
            tracks[key][f] = obs
    cadence, timing_outliers = [], []
    for clip, times in clip_times.items():
        frames = sorted(times)
        for a, b in zip(frames, frames[1:]):
            dt = (times[b] - times[a]) / 1e6
            if dt <= 0:
                raise ValueError("Non-monotonic scene timestamps")
            cadence.append(dt / (b - a))
            if dt / (b - a) > .15:
                timing_outliers.append(dict(clip=clip,
                    first_frame=a, last_frame=b, elapsed_seconds=dt,
                    seconds_per_frame=dt / (b - a)))
    result = []
    for (clip, track), obs in tracks.items():
        # Subtract integer timestamps before converting to seconds (avoid precision loss).
        start = min(o["time"] for o in obs.values())
        for o in obs.values():
            o["time"] = (o["time"] - start) / 1e6
        result.append(record(clip, track, obs, 10, splits[clip, track]))
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return result, dict(index=str(index.resolve()), csv_sha256=digest.hexdigest(),
                        raw_rows=rows, duplicate_observations=duplicates,
                        clips_with_pedestrians=len(clip_times),
                        timing_intervals_above_150ms_per_frame=timing_outliers,
                        seconds_per_frame_from_timestamp_differences=distribution(cadence),
                        timing="Extent uses exact verified Waymo endpoint timestamps. Gap counts use ROAD frame indices; native cadence nominally 10 Hz.")


def summarize(records):
    if not records:
        return dict(tracks=0)
    fields = ("duration_seconds", "context_steps_5hz", "observed", "span", "missing",
              "max_gap", "max_gap_boundary_interval_seconds", "3d_observed",
              "3d_max_gap", "3d_missing_in_full_span", "behavior_observed")
    windows = {}
    for steps in (16, 25, 32, 50, 64, 75, 100, 101, 102, 128):
        fits = [r for r in records if r["context_steps_5hz"] <= steps]
        windows[str(steps)] = dict(seconds=(steps - 1) / 5, tracks=len(fits),
            track_percent=100 * len(fits) / len(records),
            observed_frame_percent=100 * sum(r["observed"] for r in fits) / sum(r["observed"] for r in records))
    return dict(tracks=len(records), tracks_with_internal_gaps=sum(r["missing"] > 0 for r in records),
                missing_frame_percent=100 * sum(r["missing"] for r in records) / sum(r["span"] for r in records),
                paired_full_span_percent=100 * sum(r["paired_observed"] for r in records) / sum(r["span"] for r in records),
                distributions={f: distribution([r[f] for r in records]) for f in fields}, windows=windows)


def write(output, dataset, records, source):
    output.mkdir(parents=True, exist_ok=False)
    groups = dict(all=records, with_3d=[r for r in records if r["3d_observed"]],
                  only_2d=[r for r in records if not r["3d_observed"]],
                  only_3d=[r for r in records if not r["2d_observed"]])
    actions = sorted(set().union(*(set(r["actions"]) for r in records)))
    summary = dict(dataset=dataset, source=source,
        semantics="One (clip/scenario, track ID); first-to-last union of available 2D/3D pedestrian observations, INCLUDING internal gaps. No stitching across IDs/clips. Action cohorts overlap; lengths are whole tracks, not action episodes. 5 Hz steps are conservative envelope capacities, not observed/resampled labels.",
        cohorts={k: summarize(v) for k, v in groups.items()},
        actions={a: summarize([r for r in records if a in r["actions"]]) for a in actions},
        splits={s: summarize([r for r in records if r["split"] == s]) for s in sorted({r["split"] for r in records})})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "tracks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(dict(r, actions=json.dumps(r["actions"])) for r in records)
    print(json.dumps(dict(dataset=dataset, cohorts={k: v["tracks"] for k, v in summary["cohorts"].items()})))


def self_check():
    assert extent([1, 2, 2, 6]) == dict(observed=3, span=6, missing=3, max_gap=3, gap_runs=1)
    observations = {f: dict(time=(f - 1) / 5, **{"2d": True, "3d": f == 6,
                   "paired": f == 6, "behavior": f == 6, "actions": {"Stop"} if f == 6 else set()}) for f in (1, 2, 6)}
    r = record("clip", "person", observations, 5, "unknown")
    assert r["duration_seconds"] == 1 and r["context_steps_5hz"] == 6
    assert r["missing"] == 3 and r["3d_missing_in_full_span"] == 5
    assert r["3d_span"] == 1 and r["actions"] == ["Stop"]
    assert record("other-clip", "person", {1: observations[1]}, 5, "unknown")["context_steps_5hz"] == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("loki", "road-waymo"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("Track extent/gap/capacity checks passed")
    else:
        if not all((args.dataset, args.input, args.output)):
            parser.error("--dataset, --input and --output are required")
        records, source = (scan_loki if args.dataset == "loki" else scan_road)(args.input)
        write(args.output, args.dataset, records, source)
