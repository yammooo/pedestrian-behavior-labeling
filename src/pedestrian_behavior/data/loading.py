"""Validated saved tracks to CPU samples and dynamically padded whole-track batches."""

from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .features import features, normalize
from .preparation import load_track


class TrackDataset(Dataset):
    def __init__(self, collection, entries, configuration, statistics, target_policy):
        self.collection, self.entries = Path(collection), entries
        self.configuration, self.statistics, self.target_policy = configuration, statistics, target_policy
        for entry in entries:
            if Path(entry["archive"]).name != entry["archive"] or not entry["archive"].endswith(".npz"):
                raise ValueError("Invalid track archive reference")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        archive = self.collection / "tracks" / entry["archive"]
        metadata, arrays = load_track(archive)
        if metadata["track_locator"] != entry["locator"]:
            raise ValueError("Setup/archive identity mismatch")
        targets, gt_valid = self.target_policy(metadata)
        if not gt_valid.any():
            raise ValueError("Eligible track unexpectedly has no GT")
        values, flags = features(arrays, self.configuration)
        return {"inputs": torch.from_numpy(normalize(values, flags, self.statistics)),
                "targets": torch.from_numpy(targets), "gt_valid": torch.from_numpy(gt_valid),
                "locator": metadata["track_locator"], "archive": str(archive)}


def collate_tracks(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([len(s["targets"]) for s in samples], dtype=torch.int64)
    return {"inputs": pad_sequence([s["inputs"] for s in samples], batch_first=True, padding_value=0.),
            "targets": pad_sequence([s["targets"] for s in samples], batch_first=True, padding_value=-100),
            "gt_valid": pad_sequence([s["gt_valid"] for s in samples], batch_first=True, padding_value=False),
            "lengths": lengths, "padding_mask": torch.arange(int(lengths.max()))[None, :] >= lengths[:, None],
            "locators": [s["locator"] for s in samples], "archives": [s["archive"] for s in samples]}


def track_batches(dataset, training=False, seed=0):
    return DataLoader(dataset, batch_size=64, shuffle=training, drop_last=False,
                      generator=torch.Generator().manual_seed(seed), collate_fn=collate_tracks)
