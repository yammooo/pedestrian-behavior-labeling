"""LOKI action summary and track gallery."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random

from pedestrian_behavior.datasets import loki
from pedestrian_behavior.inspection.render import render_track, write_index


def inspect(root: Path):
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    scenes = list(loki.scenarios(root))
    if not scenes:
        raise ValueError(f"No scenario_* directories in {root}")
    frames = 0
    missing = Counter()
    actions = Counter()
    tracks = defaultdict(set)
    for scene in scenes:
        for frame in loki.frames(scene):
            frames += 1
            missing.update(kind for kind, path in frame.paths.items() if path is None)
            for person in frame.pedestrians.values():
                if person.label3d is None:
                    continue
                action = person.action
                actions[action] += 1
                tracks[action].add((scene.name, person.track_id))
    return len(scenes), frames, missing, actions, tracks


def select(matches: set[tuple[str, str]], seed: int, offset: int, limit: int):
    selected = sorted(matches)
    random.Random(seed).shuffle(selected)
    return selected[offset:offset + limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("summary", "gallery"):
        sub = commands.add_parser(command)
        sub.add_argument("--root", type=Path, required=True)
        if command == "gallery":
            sub.add_argument("--action", required=True)
            sub.add_argument("--offset", type=int, default=0)
            sub.add_argument("--limit", type=int, default=8)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "gallery":
        if args.offset < 0 or args.limit < 1:
            parser.error("--offset must be nonnegative and --limit must be positive")
        if args.output.exists():
            if not args.output.is_dir() or any(args.output.iterdir()):
                parser.error(f"output directory is not empty: {args.output}")
    scenes, frame_count, missing, actions, tracks = inspect(args.root)
    if args.command == "summary":
        print(f"scenarios: {scenes}\nframes: {frame_count}")
        print("missing aligned files:")
        for kind in loki.KINDS:
            print(f"  {kind}: {missing[kind]}")
        print("raw pedestrian action | frames | distinct (scenario, track_id) tracks")
        for action in sorted(actions, key=lambda value: value or ""):
            print(f"{action!r} | {actions[action]} | {len(tracks[action])}")
        return
    matches = select(tracks[args.action], args.seed, args.offset, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    for scenario, track_id in matches:
        print(f"rendering {scenario} {track_id}", flush=True)
        render_track(args.root / scenario, track_id, args.output / f"{scenario}_{track_id}.mp4")
    write_index(args.output, args.action, matches)
    print(f"{len(matches)} of {len(tracks[args.action])} matching tracks: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
