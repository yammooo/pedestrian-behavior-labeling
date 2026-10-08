"""Physical vectors, independent validity and frozen source-training statistics."""

import numpy as np

FEATURE_SETS = {
    "K": ("v_ped",),
    "K+T": ("v_ped", "delta_p_ped"),
    "K+T+R": ("v_ped", "delta_p_ped", "r", "v_rel"),
    "RAW": ("p_ped", "v_ped", "p_ego", "v_ego"),
}
SCALE_THRESHOLD = 1e-8


def features(arrays, configuration):
    """Return float64 numerical columns and one Boolean mask per vector."""
    p, v, e, ev = (arrays[k] for k in ("ped_position", "ped_velocity", "ego_position", "ego_velocity"))
    pm, vm, em, evm = (arrays[k] for k in ("ped_position_valid", "ped_velocity_valid", "ego_pose_valid", "ego_velocity_valid"))
    reference = p[np.flatnonzero(pm)[0]] if pm.any() else np.full(2, np.nan)
    vectors = {"v_ped": (v, vm), "delta_p_ped": (p-reference, pm),
               "r": (p-e, pm & em), "v_rel": (v-ev, vm & evm),
               "p_ped": (p, pm), "p_ego": (e, em), "v_ego": (ev, evm)}
    values, masks = zip(*(vectors[name] for name in FEATURE_SETS[configuration]))
    return np.concatenate(values, axis=1), np.column_stack(masks)


def fit_normalization(samples):
    """Merge float64 centered moments; samples contain numerical values and flags."""
    count = mean = m2 = None
    for values, flags in samples:
        valid = np.repeat(flags, 2, axis=1)
        if values.dtype != np.float64 or valid.shape != values.shape or not np.isfinite(values[valid]).all():
            raise ValueError("Invalid physical features")
        if count is None:
            count = np.zeros(values.shape[1], dtype=np.int64)
            mean = np.zeros(values.shape[1], dtype=np.float64)
            m2 = np.zeros(values.shape[1], dtype=np.float64)
        for j in range(values.shape[1]):
            column = values[valid[:, j], j]
            if not len(column):
                continue
            n, average = len(column), column.mean(dtype=np.float64)
            total = count[j] + n
            delta = average - mean[j]
            m2[j] += np.square(column-average).sum(dtype=np.float64) + delta**2 * count[j]*n/total
            mean[j] += delta*n/total
            count[j] = total
    if count is None:
        raise ValueError("No source-training tracks for normalization")
    std = np.sqrt(m2 / np.maximum(count, 1))
    guarded = [{"column": j, "reason": "entirely-missing" if count[j] == 0 else "near-zero",
                "std": float(std[j])} for j in range(len(count)) if count[j] == 0 or std[j] <= SCALE_THRESHOLD]
    scale = np.where(std <= SCALE_THRESHOLD, 1., std)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "count": count.tolist(),
            "std": std.tolist(), "threshold": SCALE_THRESHOLD, "guarded_columns": guarded}


def normalize(values, flags, statistics):
    mean, scale = (np.asarray(statistics[k], dtype=np.float64) for k in ("mean", "scale"))
    if (mean.shape != (values.shape[1],) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Invalid frozen normalization")
    valid = np.repeat(flags, 2, axis=1)
    numerical = np.where(valid, (values-mean)/scale, 0.)
    result = np.concatenate((numerical, flags), axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model inputs")
    return result
