"""Replay frozen E005 models under small LiDAR context perturbations.

This is deliberately an inference-only diagnostic.  It reuses the complete
LOKI E005 cache for the normal, wedge, random and off conditions.  The raw
crop condition re-encodes only the selected validation panel with Tiny-Utonia.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.data.feature_cache import isolated_extraction
from pedestrian_behavior.evaluation import load_predictions, prediction_metrics
from pedestrian_behavior.experiments import e002a_features, e005_run
from pedestrian_behavior.experiments.e005_cache import TinyUtoniaEncoder, collate_e005, lidar_tokens
from pedestrian_behavior.inspection.feature_probe import relative_xyz
from pedestrian_behavior.inspection.native_features import native_scene
from pedestrian_behavior.models import e005
from pedestrian_behavior.training import seed_run


CLASSES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
CONDITIONS = ("full", "raw-crop", "wedge", "random", "off")
DEFAULT_CACHE = Path("outputs/experiments/E005/loki-cache-full-20261117")
DEFAULT_PROBE = Path("outputs/experiments/E002a/native-probe-07")
DEFAULT_UTONIA_CODE = Path("outputs/experiments/E002a/dependencies/utonia")
DEFAULT_TINY = Path("outputs/experiments/E005/online-probe-20261117/utonia_tiny.pth")
DEFAULT_LOKI_RUN = Path("outputs/experiments/E005/runs/loki-g-rgb-utonia-s0-20261118-074440")
DEFAULT_JOINT_RUN = Path("outputs/experiments/E005/runs/joint-g-rgb-utonia-s0-20261122-001910")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _seed(*values):
    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _load_prepared(args):
    ns = argparse.Namespace(
        source="loki", arm="g-rgb-utonia", outputs_root=args.outputs_root,
        cache=args.cache, loki_cache=None, road_cache=None, joint_profile=None,
        device=args.device, aalto_fast=False,
    )
    prepared = e005_run.assembly(ns)
    return prepared, prepared["dataset"]("loki", "validation")


def _load_model(run, device, expected_epoch):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != expected_epoch:
        raise ValueError(f"{run}: expected epoch {expected_epoch}, got {checkpoint.get('epoch')}")
    arm = config.get("arm", "g-rgb-utonia")
    profile = config.get("profile")
    if not isinstance(profile, dict):
        raise ValueError(f"{run}: missing E005 execution profile")
    model = e005.build_model(arm, profile).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, config, checkpoint_path


def _select_panel(dataset, size, seed):
    """Select groups from GT only; logits are not consulted."""
    grouped = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        grouped[entry["group"]].append(index)
    candidates = []
    for group, indices in sorted(grouped.items()):
        rows = []
        for index in indices:
            base = dataset.baseline[index]
            y = base["targets"].numpy()
            gt = base["gt_valid"].numpy().astype(bool)
            labels = y[gt]
            rows.append((index, int((labels == 3).any()), int((labels == 0).any()),
                         int(gt.sum()), dataset.entries[index]["locator"]))
        # Prefer one track containing both moving and crossing, then crossing.
        rows.sort(key=lambda row: (row[1] and row[2], row[1], row[2], row[3], row[4]), reverse=True)
        candidates.append((group, rows[0]))
    rng = np.random.default_rng(seed)
    def shuffled(rows):
        rows = list(rows)
        return [rows[int(i)] for i in rng.permutation(len(rows))]
    both = [row for row in candidates if row[1][1] and row[1][2]]
    crossing = [row for row in candidates if row[1][1] and row not in both]
    moving = [row for row in candidates if row[1][2] and row not in both]
    selected, seen = [], set()
    for pool in (both, crossing, moving, candidates):
        for row in shuffled(pool):
            if row[0] not in seen:
                selected.append(row); seen.add(row[0])
            if len(selected) == size:
                break
        if len(selected) == size:
            break
    if len(selected) < size:
        raise ValueError(f"LOKI validation has only {len(selected)} eligible groups for panel size {size}")
    selected.sort(key=lambda row: row[0])
    return [index for _, (index, *_rest) in selected]


def _panel_record(dataset, indices):
    rows = []
    for index in indices:
        entry = dataset.entries[index]
        base = dataset.baseline[index]
        y, gt = base["targets"].numpy(), base["gt_valid"].numpy().astype(bool)
        rows.append({"locator": entry["locator"], "group": entry["group"], "slots": len(y),
                     "gt_frames": int(gt.sum()),
                     "class_frames": [int((gt & (y == c)).sum()) for c in range(4)]})
    return rows


def _coords(tokens):
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or tokens.shape[1] < 5:
        raise ValueError("Expected E005 LiDAR tokens with spatial cues")
    return tokens[:, -5:-2].astype(np.float32) * 20.


def _base_masks(tokens, direction, random_seed):
    coords = _coords(tokens)
    radius = np.linalg.norm(coords[:, :2], axis=1)
    eligible = (radius >= 2.) & (radius <= 20.)
    angle = np.arctan2(coords[:, 1], coords[:, 0])
    delta = np.arctan2(np.sin(angle - direction), np.cos(angle - direction))
    wedge = eligible & (np.abs(delta) <= np.pi / 4.)
    candidates = np.flatnonzero(eligible)
    rng = np.random.default_rng(random_seed)
    random = np.zeros(len(tokens), dtype=bool)
    if int(wedge.sum()):
        random[rng.choice(candidates, size=int(wedge.sum()), replace=False)] = True
    return wedge, random, int(eligible.sum())


def _mask_plan(sample, panel_seed):
    locator = sample["locator"]
    levels = sample["lidar_levels"]
    direction = np.random.default_rng(_seed(panel_seed, locator, "direction")).uniform(-np.pi, np.pi)
    wedge, random, eligible = [], [], []
    for level, frames in enumerate(levels):
        level_wedge, level_random, level_eligible = [], [], []
        for slot, frame in enumerate(frames):
            array = frame.numpy()
            if not len(array):
                level_wedge.append(np.zeros(0, bool)); level_random.append(np.zeros(0, bool)); level_eligible.append(0)
                continue
            w, r, count = _base_masks(array, direction, _seed(panel_seed, locator, slot, level, "random"))
            level_wedge.append(w); level_random.append(r); level_eligible.append(count)
        wedge.append(level_wedge); random.append(level_random); eligible.append(level_eligible)
    return {"direction_rad": float(direction), "wedge": wedge, "random": random, "eligible": eligible}


def _empty_like(frame):
    return torch.zeros((0, frame.shape[1]), dtype=frame.dtype)


def _condition_sample(base, condition, raw_levels, masks):
    sample = dict(base)
    sample["flags"] = base["flags"].clone()
    if condition == "full":
        levels = tuple([list(level) for level in base["lidar_levels"]])
    elif condition == "raw-crop":
        levels = tuple([list(level) for level in raw_levels])
    elif condition == "off":
        levels = tuple([[ _empty_like(frame) for frame in level] for level in base["lidar_levels"]])
    else:
        levels = []
        selected_masks = masks[condition]
        for level, frames in enumerate(base["lidar_levels"]):
            values = []
            for slot, frame in enumerate(frames):
                remove = selected_masks[level][slot]
                values.append(frame[~torch.from_numpy(remove)] if len(frame) else frame)
            levels.append(values)
        levels = tuple(levels)
    valid = torch.tensor([len(a) + len(b) > 0 for a, b in zip(*levels)], dtype=torch.float32)
    sample["lidar_levels"] = levels
    sample["flags"][:, 2] = valid if condition != "off" else 0.
    return sample


def _raw_levels(catalog, tiny, entry):
    group, locator = entry["group"], entry["locator"]
    track = catalog.tracks[locator]
    reader = catalog.native_reader(group)
    source_keys = [key for key, usable in zip(track["source_keys"], track["point_usable"]) if usable and key is not None]
    if not source_keys:
        return ([torch.zeros((0, 77), dtype=torch.float16) for _ in range(track["slots"])],
                [torch.zeros((0, 221), dtype=torch.float16) for _ in range(track["slots"])])
    first = track["source_keys"][0]
    if first is None:
        raise ValueError(f"{locator}: missing first native frame for fixed anchor")
    anchor = reader.read_frames(group, {first})[first]["ego_pose"]
    by_key = {}
    scene = catalog.manifest["scenes"][group]
    for frame in native_scene(reader, group, source_keys, [track["track_id"]], scene, include_rgb=False):
        points = frame["points"]
        box = frame["boxes"].get(track["track_id"])
        if points is None or box is None or frame["world_from_native"] is None:
            by_key[str(frame["key"])] = (np.empty((0, 77), np.float16), np.empty((0, 221), np.float16))
            continue
        center, size, yaw = box
        box7 = np.r_[center, size, yaw]
        relative = relative_xyz(points[:, :3], center, frame["world_from_native"], anchor)
        keep = np.linalg.norm(relative[:, :2], axis=1) <= 20.
        cropped = points[keep]
        if not len(cropped):
            by_key[str(frame["key"])] = (np.empty((0, 77), np.float16), np.empty((0, 221), np.float16))
            continue
        scaled = (points[:, :3].astype(np.float32) * .2).astype(np.float64)
        origin = np.floor(scaled / .01).astype(np.int64).min(0)
        stages = tiny.stages(cropped, grid_origin=origin)
        values = []
        for features, coords in stages:
            rel = relative_xyz(coords, center, frame["world_from_native"], anchor)
            keep_stage = np.linalg.norm(rel[:, :2], axis=1) <= 20.
            values.append(lidar_tokens(features[keep_stage], coords[keep_stage], box7,
                                       frame["world_from_native"], anchor))
        by_key[str(frame["key"])] = tuple(values)
    levels = [[], []]
    for key, usable in zip(track["source_keys"], track["point_usable"]):
        if key is None or not usable:
            values = (np.empty((0, 77), np.float16), np.empty((0, 221), np.float16))
        else:
            values = by_key.get(str(key), (np.empty((0, 77), np.float16), np.empty((0, 221), np.float16)))
        levels[0].append(torch.from_numpy(values[0]))
        levels[1].append(torch.from_numpy(values[1]))
    return tuple(levels)


def _logits(model, sample, device):
    batch = collate_e005([sample])
    with torch.inference_mode():
        # Let the checkpoint's calibrated profile choose ordinary/recompute,
        # matching the original E005 validation path.
        result = model(batch)[0, :len(sample["targets"])].cpu().numpy().astype(np.float32)
    if result.shape != (len(sample["targets"]), 4) or not np.isfinite(result).all():
        raise ValueError("Nonfinite or malformed diagnostic logits")
    return result


def _metrics(rows):
    return prediction_metrics(rows, 4)


def _save_predictions(path, condition_rows, panel):
    offsets = np.cumsum([0] + [row["slots"] for row in panel], dtype=np.int64)
    logits = np.concatenate([np.stack([condition_rows[c][i]["logits"] for c in CONDITIONS], axis=0)
                             for i in range(len(panel))], axis=1).astype(np.float32, copy=False)
    targets = np.concatenate([np.asarray(row["targets"], dtype=np.int64) for row in panel])
    gt_valid = np.concatenate([np.asarray(row["gt_valid"], dtype=bool) for row in panel])
    lidar_valid = np.concatenate([np.stack([condition_rows[c][i]["lidar_valid"] for c in CONDITIONS], axis=0)
                                  for i in range(len(panel))], axis=1)
    np.savez_compressed(path, conditions=np.asarray(CONDITIONS), locators=np.asarray([r["locator"] for r in panel]),
                        groups=np.asarray([r["group"] for r in panel]), offsets=offsets, targets=targets,
                        gt_valid=gt_valid, lidar_valid=lidar_valid, logits=logits)


def _synthetic_check():
    tokens = np.zeros((6, 77), np.float16)
    tokens[:, -5:-2] = np.asarray([[1, 0, 0], [3, 0, 0], [0, 3, 0], [-3, 0, 0], [0, -3, 0], [0, 0, 0]], np.float16) / 20
    wedge, random, eligible = _base_masks(tokens, 0., 17)
    assert eligible == 4 and int(wedge.sum()) == int(random.sum()) == 1
    assert not wedge[0] and not wedge[5]


def _mask_summary(plan):
    return {"direction_rad": plan["direction_rad"],
            "eligible": plan["eligible"],
            "wedge_removed": [[int(mask.sum()) for mask in level] for level in plan["wedge"]],
            "random_removed": [[int(mask.sum()) for mask in level] for level in plan["random"]]}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    _synthetic_check()
    started = monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    seed_run(args.panel_seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    prepared, dataset = _load_prepared(args)
    indices = _select_panel(dataset, args.panel_size, args.panel_seed)
    panel = _panel_record(dataset, indices)
    for row, index in zip(panel, indices):
        base = dataset.baseline[index]
        row["targets"] = base["targets"].numpy().tolist()
        row["gt_valid"] = base["gt_valid"].numpy().astype(bool).tolist()
    args.output.mkdir(parents=True)
    _write_json(args.output / "panel.json", {"selection": "seeded GT-only group panel", "seed": args.panel_seed,
                                                "tracks": [{k: v for k, v in row.items() if k not in ("targets", "gt_valid")} for row in panel]})

    config, transform = e002a_features.load_configuration(args.probe)
    if args.loki_root:
        config["loki_root"] = str(args.loki_root)
    catalog = e002a_features.FeatureCatalog(args.outputs_root, "loki", config, transform)
    raw_cache = {}
    mask_plans = {}
    with isolated_extraction():
        tiny = TinyUtoniaEncoder(args.utonia_code, args.tiny_checkpoint, device)
        for i, (row, index) in enumerate(zip(panel, indices), 1):
            raw_cache[row["locator"]] = _raw_levels(catalog, tiny, dataset.entries[index])
            mask_plans[row["locator"]] = _mask_plan(dataset[index], args.panel_seed)
            print(f"Raw crop {i}/{len(panel)} | {row['group']} | {tiny.calls} sweeps", flush=True)
    _write_json(args.output / "mask-plan.json", {locator: _mask_summary(plan)
                                                  for locator, plan in mask_plans.items()})

    models = {
        "loki": (args.loki_run, args.loki_epoch),
        "joint": (args.joint_run, args.joint_epoch),
    }
    output_summary = {"status": "complete", "experiment": "E006d", "conditions": list(CONDITIONS),
                      "models": {}, "panel": panel, "started_utc": started_utc,
                      "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      "raw_crop_sweeps": tiny.calls,
                      "cache": str(args.cache), "cache_manifest_sha256": checksum(args.cache / "manifest.json"),
                      "scope": "LOKI validation only; inference-only; no W&B; no held-out tests"}
    for name, (run, expected_epoch) in models.items():
        model, checkpoint, model_config, checkpoint_path = _load_model(run, device, expected_epoch)
        normalization = checkpoint.get("normalization")
        if not isinstance(normalization, dict) or "baseline" not in normalization or "bbox" not in normalization:
            raise ValueError(f"{run}: checkpoint has no complete E005 normalization")
        # The joint checkpoint uses merged source statistics; replay each model
        # under the exact statistics recorded with that checkpoint.
        dataset.baseline.statistics = normalization["baseline"]
        dataset.baseline.bbox_statistics = normalization["bbox"]
        condition_rows = {condition: [] for condition in CONDITIONS}
        baseline_reference = load_predictions(Path(run) / "loki-validation" / "predictions.npz")
        reference = {r["locator"]: r for r in baseline_reference}
        diagnostics = {condition: {"token_counts": [], "paired": []} for condition in CONDITIONS}
        max_delta = 0.
        for row, index in zip(panel, indices):
            base = dataset[index]
            masks = mask_plans[row["locator"]]
            plan = {"wedge": masks["wedge"], "random": masks["random"]}
            for condition in CONDITIONS:
                levels = raw_cache[row["locator"]] if condition == "raw-crop" else None
                sample = _condition_sample(base, condition, levels, plan)
                logits = _logits(model, sample, device)
                condition_rows[condition].append({"locator": row["locator"], "group": row["group"],
                                                  "targets": row["targets"], "gt_valid": row["gt_valid"],
                                                  "logits": logits,
                                                  "lidar_valid": sample["flags"][:, 2].numpy().astype(bool)})
                diagnostics[condition]["token_counts"].append(
                    [[int(len(a)), int(len(b))] for a, b in zip(*sample["lidar_levels"])])
            prior = reference[row["locator"]]
            current = condition_rows["full"][-1]
            if (prior["group"] != row["group"]
                    or not np.array_equal(prior["targets"], current["targets"])
                    or not np.array_equal(prior["gt_valid"], current["gt_valid"])
                    or not np.array_equal(prior["source_flags"]["lidar_valid"], current["lidar_valid"])):
                raise ValueError(f"{name}: cache replay labels or masks differ for {row['locator']}")
            delta = np.abs(current["logits"] - prior["logits"])
            max_delta = max(max_delta, float(delta.max(initial=0.)))
            if not np.allclose(condition_rows["full"][-1]["logits"], reference[row["locator"]]["logits"], rtol=1e-4, atol=1e-5):
                raise ValueError(f"{name}: full cache replay differs for {row['locator']} (max {delta.max():.3g})")
        model_summary = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": checksum(checkpoint_path),
                         "selected_epoch": int(checkpoint["epoch"]), "full_cache_max_abs_logit_delta": max_delta,
                         "metrics": {condition: _metrics(condition_rows[condition]) for condition in CONDITIONS},
                         "token_counts": {condition: diagnostics[condition]["token_counts"] for condition in CONDITIONS},
                         "paired": {}}
        full = condition_rows["full"]
        for condition in CONDITIONS:
            gaps = {0: [], 3: []}
            pairs = {name: {"current": 0, "new": 0, "rescued": 0}
                     for name in ("moving_as_crossing", "crossing_as_moving")}
            for current, base in zip(condition_rows[condition], full):
                gt = np.asarray(current["gt_valid"])
                target = np.asarray(current["targets"])
                pred, old = current["logits"].argmax(1), base["logits"].argmax(1)
                delta = current["logits"] - base["logits"]
                for key, truth, wrong in (("moving_as_crossing", 0, 3), ("crossing_as_moving", 3, 0)):
                    mask = gt & (target == truth)
                    if mask.any():
                        gaps[truth].append(float((delta[:, 3] - delta[:, 0])[mask].mean()))
                    error, previous = mask & (pred == wrong), mask & (old == wrong)
                    pairs[key]["current"] += int(error.sum())
                    pairs[key]["new"] += int((error & ~previous).sum())
                    pairs[key]["rescued"] += int((previous & ~error).sum())
            model_summary["paired"][condition] = {
                "mean_track_crossing_minus_moving_shift": {
                    CLASSES[truth]: float(np.mean(values)) if values else None for truth, values in gaps.items()},
                "paired_frame_errors": pairs}
        _save_predictions(args.output / f"predictions-{name}.npz", condition_rows, panel)
        output_summary["models"][name] = model_summary
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output_summary["elapsed_seconds"] = monotonic() - started
    _write_json(args.output / "summary.json", output_summary)
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "panel_tracks": len(panel), "models": list(models)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--loki-root", type=Path)
    parser.add_argument("--utonia-code", type=Path, default=DEFAULT_UTONIA_CODE)
    parser.add_argument("--tiny-checkpoint", type=Path, default=DEFAULT_TINY)
    parser.add_argument("--loki-run", type=Path, default=DEFAULT_LOKI_RUN)
    parser.add_argument("--joint-run", type=Path, default=DEFAULT_JOINT_RUN)
    parser.add_argument("--loki-epoch", type=int, default=14)
    parser.add_argument("--joint-epoch", type=int, default=30)
    parser.add_argument("--panel-size", type=int, default=12)
    parser.add_argument("--panel-seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
