"""Complete-track logical updates and shape-based physical packing."""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_multimodal(samples):
    if not samples:
        raise ValueError("Empty physical microbatch")
    lengths = torch.tensor([len(s["targets"]) for s in samples], dtype=torch.int64)
    t = int(lengths.max())
    empty = torch.zeros(0, 68)
    return {**{k: pad_sequence([s[k] for s in samples], batch_first=True, padding_value=value)
               for k, value in (("geometry", 0.), ("rgb", 0.), ("flags", 0.), ("targets", -100), ("gt_valid", False))},
            "lengths": lengths, "padding_mask": torch.arange(t)[None, :] >= lengths[:, None],
            "points": [f for s in samples for f in s["points"] + [empty] * (t - len(s["targets"]))],
            "locators": [s["locator"] for s in samples], "archives": [s["archive"] for s in samples],
            "prediction_metadata": [s.get("prediction_metadata", {}) for s in samples]}


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

    def __init__(self, dataset, profile, training=False, seed=0, max_updates=None):
        self.dataset, self.profile = dataset, profile
        self.training, self.max_updates = training, max_updates
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        n = (len(self.dataset) + 63) // 64
        return min(n, self.max_updates) if self.max_updates is not None else n

    def __iter__(self):
        order = torch.randperm(len(self.dataset), generator=self.generator).tolist() if self.training else list(range(len(self.dataset)))
        for update, start in enumerate(range(0, len(order), 64)):
            if self.max_updates is not None and update >= self.max_updates:
                break
            yield order[start:start + 64]

    def microbatches(self, indices):
        lengths = {i: self.dataset.entries[i]["slots"] for i in indices}
        costs = {i: self.dataset.cost(i) for i in indices}
        for group in pack_tracks(indices, lengths, costs, self.profile):
            samples = [self.dataset[i] for i in group]
            batch = collate_multimodal(samples)
            yield batch

    def predictions(self):
        for indices in self:
            yield from self.microbatches(indices)


class PredictionBatches:
    def __init__(self, logical):
        self.logical = logical

    def __iter__(self):
        return self.logical.predictions()
