"""Diagnostic pooling of frozen point outputs; not a production cache format."""

import numpy as np


def pool_outputs(coordinates, features, voxel_m, preserve=None):
    """Mean features/XYZ on a native-origin grid; preserved rows remain individual tokens."""
    n = len(coordinates)
    if (coordinates.shape != (n, 3) or features.ndim != 2 or len(features) != n or
            not np.isfinite(coordinates).all() or not np.isfinite(features).all() or
            not np.isfinite(voxel_m) or voxel_m <= 0):
        raise ValueError("Invalid pooling inputs")
    if preserve is None:
        preserve = np.zeros(n, bool)
    if preserve.shape != (n,) or preserve.dtype != bool:
        raise ValueError("Invalid preservation mask")
    grid = np.floor(coordinates.astype(np.float64) / voxel_m)
    if np.any(np.abs(grid) >= 2**62):
        raise ValueError("Pooling grid exceeds integer range")
    keys = np.c_[np.ones(n, np.int64), grid.astype(np.int64)]
    keys[preserve] = np.c_[np.zeros(preserve.sum(), np.int64),
                           np.flatnonzero(preserve), np.zeros((preserve.sum(), 2), np.int64)]
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]] if n else np.zeros(0, np.int64)
    means = []
    for values in (coordinates, features):
        sums = np.add.reduceat(values[order].astype(np.float64), starts, axis=0) if n else np.zeros((0, values.shape[1]))
        means.append((sums / counts[:, None]).astype(np.float32))
    return *means, inverse, counts
