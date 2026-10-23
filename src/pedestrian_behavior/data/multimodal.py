"""Complete-track logical updates and shape-based physical packing."""

from concurrent.futures import ThreadPoolExecutor
import time

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_multimodal(samples):
    if not samples:
        raise ValueError("Empty physical microbatch")
    lengths = torch.tensor([len(s["targets"]) for s in samples], dtype=torch.int64)
    t = int(lengths.max())
    empty = torch.zeros(0, 68)
    for key in ("rgb", "points"):
        if any((key in s) != (key in samples[0]) for s in samples):
            raise ValueError("Mixed modality schemas in physical microbatch")
    result = {**{k: pad_sequence([s[k] for s in samples], batch_first=True, padding_value=value)
               for k, value in (("geometry", 0.), ("flags", 0.), ("targets", -100), ("gt_valid", False))},
            "lengths": lengths, "padding_mask": torch.arange(t)[None, :] >= lengths[:, None],
            "locators": [s["locator"] for s in samples], "archives": [s["archive"] for s in samples],
            "prediction_metadata": [s.get("prediction_metadata", {}) for s in samples]}
    if "rgb" in samples[0]:
        result["rgb"] = pad_sequence([s["rgb"] for s in samples], batch_first=True)
    if "points" in samples[0]:
        result["points"] = [f for s in samples for f in s["points"] + [empty] * (t - len(s["targets"]))]
    return result


def pack_tracks(indices, lengths, costs, profile):
    """Stable first-fit by length inside one logical update; unknown costs stay single."""
    groups = []
    for i in sorted(indices, key=lambda i: (-lengths[i], i)):
        if lengths[i] ** 2 > profile["temporal_cells"]:
            raise ValueError("Complete track exceeds calibrated temporal ceiling")
        cost = costs[i]
        if cost is None:
            groups.append(([i], None))
            continue
        for slot, (group, total) in enumerate(groups):
            if total is None:
                continue
            n, t = len(group) + 1, max(lengths[group[0]], lengths[i])
            if (n <= profile["max_tracks"] and n * t * t <= profile["temporal_cells"]
                    and total + cost <= profile["microbatch_tokens"]):
                group.append(i)
                groups[slot] = (group, total + cost)
                break
        else:
            groups.append(([i], cost))
    return [g for g, _ in groups]


class LogicalBatches:
    """One persistent generator; no feature payloads materialized for the whole update."""

    def __init__(self, dataset, profile, training=False, seed=0, max_updates=None, on_pack=None, *, prefetch=False):
        self.dataset, self.profile = dataset, profile
        self.training, self.max_updates = training, max_updates
        self.generator = torch.Generator().manual_seed(seed)
        self.on_pack, self.iteration = on_pack, 0
        self.prefetch = prefetch
        self.stats = {"prepared_seconds": 0., "wait_seconds": 0., "prepared_microbatches": 0}

    def __len__(self):
        n = (len(self.dataset) + 63) // 64
        return min(n, self.max_updates) if self.max_updates is not None else n

    def __iter__(self):
        self.iteration += 1
        order = torch.randperm(len(self.dataset), generator=self.generator).tolist() if self.training else list(range(len(self.dataset)))
        for update, start in enumerate(range(0, len(order), 64)):
            if self.max_updates is not None and update >= self.max_updates:
                break
            yield order[start:start + 64]

    def microbatches(self, indices):
        iterator = self._microbatches(indices)
        if not self.prefetch:
            yield from iterator
            return
        # Exactly one future beside the consumed batch; no whole-update materialization.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="prepared-input") as pool:
            pending = pool.submit(next, iterator, None)
            try:
                while True:
                    begin = time.perf_counter()
                    batch = pending.result()
                    self.stats["wait_seconds"] += time.perf_counter() - begin
                    if batch is None:
                        return
                    pending = pool.submit(next, iterator, None)
                    yield batch
                    del batch
            finally:
                pending.cancel()
                # Join before closing the generator or its cache resources.
                pool.shutdown(wait=True, cancel_futures=True)
                iterator.close()

    def _microbatches(self, indices):
        lengths = {i: self.dataset.entries[i]["slots"] for i in indices}
        costs = {i: self.dataset.cost(i) for i in indices}
        for group in pack_tracks(indices, lengths, costs, self.profile):
            begin = time.perf_counter()
            samples = [self.dataset[i] for i in group]
            batch = collate_multimodal(samples)
            if self.on_pack:
                self.on_pack({"pass": self.iteration, "logical_indices": indices, "physical_indices": group,
                              "locators": batch["locators"], "lengths": batch["lengths"].tolist(),
                              "point_counts": [len(f) for f in batch.get("points", ())],
                              "unknown_cost_indices": [i for i in indices if costs[i] is None]})
            self.stats["prepared_seconds"] += time.perf_counter() - begin
            self.stats["prepared_microbatches"] += 1
            yield batch
            del batch, samples

    def predictions(self):
        for indices in self:
            yield from self.microbatches(indices)


class PredictionBatches:
    def __init__(self, logical):
        self.logical = logical

    def __iter__(self):
        return self.logical.predictions()
