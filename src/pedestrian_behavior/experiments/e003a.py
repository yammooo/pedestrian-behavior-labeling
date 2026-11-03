"""E003a structured branch/time masking over the frozen E002d inputs."""

from collections import Counter
import hashlib
import json
import numpy as np
import torch

from pedestrian_behavior.data.bbox import load_bbox
from pedestrian_behavior.data.features import features
from pedestrian_behavior.data.preparation import checksum, load_track
from pedestrian_behavior.inspection.pretrained import DINO_REVISION, DINO_PTH_SHA256


BRANCHES = ("geometry", "rgb", "lidar")
BLOCK_LENGTHS = (1, 5, 10, 20)
BLOCK_COUNTS = (0, 1, 2, 3)
BLOCK_COUNT_PROBABILITIES = (.1, .3, .3, .3)


def mask_seed(seed, source, epoch, locator):
    key = json.dumps([int(seed), source, int(epoch), locator], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _empty_gaps(natural, effective):
    observable = natural.any(axis=1)
    retained = effective.any(axis=1)
    gaps, start = [], None
    for slot, has_input in enumerate(retained.tolist() + [True]):
        if not has_input and start is None:
            start = slot
        elif has_input and start is not None:
            gaps.append({"start": start, "length": slot - start,
                         "newly_hidden": int(observable[start:slot].sum())})
            start = None
    return gaps


def _details(source, epoch, locator, natural, effective, *, augmented=False, attempts=(),
             accepted_attempt=None, fallback=False):
    changed = not np.array_equal(effective, natural)
    gaps = _empty_gaps(natural, effective)
    return {"source": source, "epoch": int(epoch), "locator": locator,
        "augmented": bool(augmented), "changed": bool(changed),
        "candidate_draws": len(attempts), "accepted_attempt": accepted_attempt,
        "fallback": bool(fallback), "attempts": list(attempts),
        "natural": {name: "".join("1" if value else "0" for value in natural[:, i])
                    for i, name in enumerate(BRANCHES)},
        "effective": {name: "".join("1" if value else "0" for value in effective[:, i])
                      for i, name in enumerate(BRANCHES)},
        "natural_branch_slots": {name: int(natural[:, i].sum()) for i, name in enumerate(BRANCHES)},
        "effective_branch_slots": {name: int(effective[:, i].sum()) for i, name in enumerate(BRANCHES)},
        "branch_retention_fraction": {name: int(effective[:, i].sum()) / int(natural[:, i].sum())
            if natural[:, i].any() else None for i, name in enumerate(BRANCHES)},
        "natural_observable_slots": int(natural.any(axis=1).sum()),
        "effective_observable_slots": int(effective.any(axis=1).sum()),
        "empty_gaps": gaps,
        "newly_hidden_observable_slots": sum(g["newly_hidden"] for g in gaps)}


def _rgb_identity(supply, source, locator):
    """Match the existing E002a RGB key without loading the 768D feature array."""
    feature_supply = supply.rgb_supply
    catalog = feature_supply.catalogs[source]
    track = catalog.tracks[locator]
    # shortcut: E002a cache hashes its producer module, so keep this exact key local; move it if that identity is versioned.
    return {**feature_supply.identity, **catalog.identity, "kind": "rgb-cls", "locator": locator,
        "archive_sha256": track["sha256"], "source": catalog.source_identity(track["group"]),
        "encoder": DINO_REVISION, "checkpoint": feature_supply.identity.get("rgb_checkpoint_sha256", DINO_PTH_SHA256),
        "crop": [3, 256, "bicubic-mean-padding"], "batch": 16,
        "precision": "fp16-cuda-autocast-float32-storage"}


def _rgb_validity(supply, source, locator, slots):
    """Read only frozen RGB availability; preserve E002a cache identities."""
    identity = _rgb_identity(supply, source, locator)
    valid = supply.rgb_supply.rgb_cache.valid(identity)
    if valid.shape != (slots,):
        raise ValueError("Invalid prepared RGB validity metadata")
    return valid.copy()


class MaskSampler:
    """Per-track counter-based RNG; it never consumes training/model randomness."""

    def __init__(self, seed=0):
        self.seed = seed

    def draw(self, source, epoch, locator, natural, *, force_augmented=False):
        natural = np.asarray(natural, dtype=bool)
        if natural.ndim != 2 or natural.shape[1] != 3 or not len(natural):
            raise ValueError("Masking needs a nonempty complete track")
        natural = natural.copy()
        length = len(natural)
        rng = np.random.default_rng(mask_seed(self.seed, source, epoch, locator))
        augmented = force_augmented or bool(rng.random() >= .5)
        attempts = []
        if not augmented:
            effective = natural.copy()
            attempt = None
            fallback = False
        else:
            effective = natural.copy()
            attempt = None
            fallback = True
            for number in range(1, 4):
                if rng.random() < .25:
                    block_length = length
                else:
                    choices = [value for value in BLOCK_LENGTHS if value < length]
                    block_length = int(rng.choice(choices)) if choices else length
                selected, subsets = np.zeros_like(natural), []
                for start in range(0, length, block_length):
                    count = int(rng.choice(BLOCK_COUNTS, p=BLOCK_COUNT_PROBABILITIES))
                    subset = sorted(rng.choice(3, size=count, replace=False).tolist())
                    selected[start:start + block_length, subset] = True
                    subsets.append("".join("GRL"[i] for i in subset) or "-")
                candidate = natural & selected
                gaps = _empty_gaps(natural, candidate)
                retained = int(candidate.any(axis=1).sum())
                minimum_retained = int(np.ceil(.25 * natural.any(axis=1).sum()))
                max_newly_hidden = min(10, length // 4)
                reasons = []
                if retained < minimum_retained:
                    reasons.append("retained-observable-slots")
                if any(gap["newly_hidden"] > max_newly_hidden for gap in gaps):
                    reasons.append("empty-gap-newly-hidden")
                attempts.append({"number": number, "block_length": block_length, "subsets": subsets,
                                 "rejected": reasons})
                if not reasons:
                    effective, attempt, fallback = candidate, number, False
                    break
        return effective, _details(source, epoch, locator, natural, effective,
            augmented=augmented, attempts=attempts, accepted_attempt=attempt, fallback=fallback)


class FeatureDataset:
    """Apply an E003a view before feature reads and physical point packing."""

    def __init__(self, base, source, arm, *, mode="natural", branch=None, epoch=None,
                 sampler=None, seed=0, fixed_masks=None):
        if mode not in ("natural", "training", "branch", "fixedmixed"):
            raise ValueError("Unknown E003a view")
        if mode == "branch" and branch not in BRANCHES:
            raise ValueError("Unknown diagnostic branch")
        self.base, self.source, self.arm = base, source, arm
        self.mode, self.branch, self.epoch = mode, branch, epoch
        self.sampler, self.seed = sampler or MaskSampler(seed), seed
        self.fixed_masks = fixed_masks
        self._natural, self._masks = {}, {}
        self._draw_epoch, self._records = None, {}

    def __len__(self):
        return len(self.base)

    @property
    def entries(self):
        return self.base.entries

    def _natural_flags(self, index):
        entry = self.base.baseline.entries[index]
        locator = entry["locator"]
        if locator in self._natural:
            return self._natural[locator]
        archive = self.base.baseline.collection / "tracks" / entry["archive"]
        metadata, arrays = load_track(archive)
        if metadata["track_locator"] != locator:
            raise ValueError("Frozen track identity mismatch")
        _, group_flags = features(arrays, self.base.baseline.configuration)
        bbox_archive = self.base.baseline.extension / "tracks" / entry["archive"]
        if checksum(archive) != entry["e001_sha256"] or checksum(bbox_archive) != entry["bbox_sha256"]:
            raise ValueError("Frozen geometry/bbox input changed")
        _, bbox = load_bbox(bbox_archive, metadata, entry["e001_sha256"])
        geometry = group_flags.any(axis=1) | bbox["bbox_valid"] | bbox["bbox_velocity_valid"]
        slots = len(geometry)
        rgb = (_rgb_validity(self.base.supply, self.source, locator, slots)
               if self.base.supply is not None and self.base.supply.rgb_supply is not None
               else np.zeros(slots, dtype=bool))
        points = (self.base.supply.points.counts[locator] > 0
                  if self.base.supply is not None and self.base.supply.points is not None
                  else np.zeros(slots, dtype=bool))
        natural = np.column_stack((geometry, rgb, points)).astype(bool, copy=False)
        self._natural[locator] = natural
        return natural

    def _effective(self, index):
        entry = self.base.baseline.entries[index]
        locator, natural = entry["locator"], self._natural_flags(index)
        if self.mode == "training":
            epoch = int(self.epoch() if callable(self.epoch) else self.epoch or 1)
            if epoch != self._draw_epoch:
                self._draw_epoch, self._records, self._masks = epoch, {}, {}
            key = (epoch, locator)
            if key not in self._masks:
                effective, details = self.sampler.draw(self.source, epoch, locator, natural)
                self._masks[key], self._records[locator] = effective, details
            return self._masks[key], self._records[locator]
        if self.mode == "branch":
            effective = np.zeros_like(natural)
            effective[:, BRANCHES.index(self.branch)] = natural[:, BRANCHES.index(self.branch)]
            details = _details(self.source, 0, locator, natural, effective)
            return effective, details
        if self.mode == "fixedmixed":
            key = (self.source, locator)
            drawn, details = self.sampler.draw(self.source, 0, locator, natural, force_augmented=True)
            if self.fixed_masks is not None and key in self.fixed_masks:
                effective = self.fixed_masks[key].copy()
                if not np.array_equal(drawn, effective):
                    raise ValueError("Fixedmixed sampler changed between model evaluations")
            else:
                effective = drawn
                if self.fixed_masks is not None:
                    self.fixed_masks[key] = effective.copy()
            self._records[locator] = details
            return effective, details
        effective = natural.copy()
        details = _details(self.source, 0, locator, natural, effective)
        return effective, details

    def cost(self, index):
        effective, _ = self._effective(index)
        if self.base.supply is None or self.base.supply.points is None:
            return 0
        locator = self.base.baseline.entries[index]["locator"]
        return int(self.base.supply.points.counts[locator][effective[:, 2]].sum())

    def __getitem__(self, index):
        effective, details = self._effective(index)
        sample = self.base.baseline[index]
        locator, slots = sample["locator"], len(sample["targets"])
        natural = self._natural_flags(index)
        geometry = sample.pop("inputs")
        natural_geometry = geometry[:, [8, 9, 10, 11, 18, 19]].bool().any(-1).numpy()
        if not np.array_equal(natural_geometry, natural[:, 0]):
            raise ValueError("Geometry availability metadata differs from loaded inputs")
        geometry = geometry.clone()
        geometry[~torch.from_numpy(effective[:, 0])] = 0.
        flags = [torch.from_numpy(effective[:, 0])]
        source_flags = {"geometry_valid": natural[:, 0].copy(),
                        "effective_geometry_valid": effective[:, 0].copy()}
        metadata = {"source_flags": source_flags}
        if self.base.supply is not None:
            metadata["frozen_feature_identity"] = self.base.supply.identity
        if self.base.supply is not None and self.base.supply.rgb_supply is not None:
            rgb = self.base.supply.rgb(self.source, locator)
            if len(rgb["valid"]) != slots or not np.array_equal(rgb["valid"], natural[:, 1]):
                raise ValueError("RGB availability metadata differs from loaded features")
            sample["rgb"] = torch.from_numpy(rgb["cls"])
            flags.append(torch.from_numpy(effective[:, 1]))
            source_flags.update(rgb_valid=natural[:, 1].copy(), effective_rgb_valid=effective[:, 1].copy())
        if self.base.supply is not None and self.base.supply.points is not None:
            counts = self.base.supply.points.counts[locator]
            if len(counts) != slots or not np.array_equal(counts > 0, natural[:, 2]):
                raise ValueError("LiDAR availability metadata differs from indexed counts")
            point_frames = [torch.from_numpy(value) for value in
                            self.base.supply.points.iter_track(locator, keep_mask=effective[:, 2])]
            if len(point_frames) != slots:
                raise ValueError("LiDAR complete-track length changed")
            effective_counts = np.where(effective[:, 2], counts, 0)
            sample["points"] = point_frames
            flags.append(torch.from_numpy(effective[:, 2]))
            source_flags.update(lidar_valid=natural[:, 2].copy(),
                                effective_lidar_valid=effective[:, 2].copy(),
                                point_tokens=counts.copy(), effective_point_tokens=effective_counts)
        sample.update(geometry=geometry, flags=torch.stack(flags, -1).float())
        details.update(self._posthoc_gt(details, effective, sample["targets"].numpy(), sample["gt_valid"].numpy()))
        metadata.update(source_flags=source_flags, e003a_mask=details)
        sample["prediction_metadata"] = metadata
        if self.mode == "training":
            self._records[locator] = details
        elif self.mode == "fixedmixed":
            self._records[locator] = details
        return sample

    @staticmethod
    def _posthoc_gt(details, effective, targets, gt_valid):
        natural = np.column_stack([np.fromiter((c == "1" for c in details["natural"][name]), bool)
                                   for name in BRANCHES])
        natural_input, effective_input = natural.any(axis=1), effective.any(axis=1)
        classes = []
        for label in range(4):
            rows = gt_valid & (targets == label)
            classes.append({"gt": int(rows.sum()), "natural_input_gt": int((rows & natural_input).sum()),
                            "effective_input_gt": int((rows & effective_input).sum())})
        return {"posthoc_gt": {"gt_slots": int(gt_valid.sum()),
                "natural_input_gt_slots": int((gt_valid & natural_input).sum()),
                "effective_input_gt_slots": int((gt_valid & effective_input).sum()),
                "class_coverage": classes}}

    def finish_epoch(self):
        if len(self._records) != len(self):
            raise ValueError("Training mask records do not cover every track exactly once")
        records = [self._records[key] for key in sorted(self._records)]
        summary = summarize_masks(records)
        return summary, records


def summarize_masks(records):
    rejected, tried_lengths, tried_subsets = Counter(), Counter(), Counter()
    accepted_lengths, accepted_subsets, empty_gaps, gap_hidden = Counter(), Counter(), Counter(), Counter()
    branch_natural, branch_effective = Counter(), Counter()
    totals, class_coverage = Counter(), [Counter() for _ in range(4)]
    for record in records:
        totals["tracks"] += 1
        totals["augmented"] += int(record["augmented"])
        totals["changed"] += int(record["changed"])
        totals["fallback"] += int(record["fallback"])
        totals["candidate_draws"] += record["candidate_draws"]
        totals["natural_observable_slots"] += record["natural_observable_slots"]
        totals["effective_observable_slots"] += record["effective_observable_slots"]
        totals["newly_hidden_observable_slots"] += record["newly_hidden_observable_slots"]
        totals["unchanged_augmented_accepted"] += int(record["augmented"] and not record["changed"] and not record["fallback"])
        totals["unchanged_fallback"] += int(record["fallback"] and not record["changed"])
        for attempt in record["attempts"]:
            tried_lengths[str(attempt["block_length"])] += 1
            for subset in attempt["subsets"]:
                tried_subsets[subset] += 1
            for reason in attempt["rejected"]:
                rejected[reason] += 1
            if attempt["number"] == record["accepted_attempt"]:
                accepted_lengths[str(attempt["block_length"])] += 1
                accepted_subsets.update(attempt["subsets"])
        for branch, count in record["natural_branch_slots"].items():
            branch_natural[branch] += count
            branch_effective[branch] += record["effective_branch_slots"][branch]
        for gap in record["empty_gaps"]:
            empty_gaps[str(gap["length"])] += 1
            gap_hidden[str(gap["newly_hidden"])] += 1
        gt = record.get("posthoc_gt", {})
        for key in ("gt_slots", "natural_input_gt_slots", "effective_input_gt_slots"):
            totals[key] += gt.get(key, 0)
        for index, coverage in enumerate(gt.get("class_coverage", ())):
            for key, value in coverage.items():
                class_coverage[index][key] += value
    clean = totals["tracks"] - totals["augmented"]
    return {**dict(totals), "clean_tracks": clean,
            "clean_fraction": clean / totals["tracks"],
            "changed_augmented_fraction": totals["changed"] / totals["augmented"] if totals["augmented"] else None,
            "fallback_fraction": totals["fallback"] / totals["augmented"] if totals["augmented"] else None,
            "effective_observable_fraction": totals["effective_observable_slots"] / totals["natural_observable_slots"]
                if totals["natural_observable_slots"] else None,
            "branch_retention_fraction": {branch: branch_effective[branch] / branch_natural[branch]
                if branch_natural[branch] else None for branch in BRANCHES},
            "rejection_reasons": dict(rejected), "attempted_block_lengths": dict(tried_lengths),
            "attempted_subsets": dict(tried_subsets), "accepted_block_lengths": dict(accepted_lengths),
            "accepted_subsets": dict(accepted_subsets), "natural_branch_slots": dict(branch_natural),
            "effective_branch_slots": dict(branch_effective), "empty_gap_lengths": dict(empty_gaps),
            "empty_gap_newly_hidden": dict(gap_hidden),
            "posthoc_gt_class_coverage": [dict(row) for row in class_coverage]}
