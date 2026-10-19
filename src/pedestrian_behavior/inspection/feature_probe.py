"""Small, policy-free feature/crop/coordinate checks for frozen encoders."""

import numpy as np
from PIL import Image
import torch
from torch import nn

RGB_MEAN = np.array([.485, .456, .406], dtype=np.float32)
RGB_STD = np.array([.229, .224, .225], dtype=np.float32)


def square_crop(image, box):
    """Image-relative xyxy → centered 1.5x square; never shift at the border."""
    if image is None or box is None:
        return None, {"valid": False, "reason": "missing-image-or-box"}
    b = np.asarray(box, dtype=np.float64) * np.tile(image.size, 2)
    if b.shape != (4,) or not np.isfinite(b).all() or (b[2:] <= b[:2]).any():
        return None, {"valid": False, "reason": "invalid-box"}
    left, top = np.maximum(b[:2], 0)
    right, bottom = np.minimum(b[2:], image.size)
    if right <= left or bottom <= top:
        return None, {"valid": False, "reason": "box-outside-image"}
    side = 1.5 * max(b[2:]-b[:2])
    center = (b[:2]+b[2:])/2
    bounds = np.r_[center-side/2, center+side/2]
    # Sampling in output coordinates avoids allocating a huge out-of-image canvas.
    scale = side/256
    raw = np.asarray(image.convert("RGB"), dtype=np.float32)/255
    channels = []
    for c in range(3):
        channel = Image.fromarray(raw[:, :, c], mode="F")
        channel = channel.transform((256, 256), Image.Transform.AFFINE,
                                    (scale, 0, bounds[0], 0, scale, bounds[1]),
                                    resample=Image.Resampling.BICUBIC, fillcolor=float(RGB_MEAN[c]))
        channels.append(np.asarray(channel))
    pixels = np.stack(channels, axis=-1)
    area = np.prod(np.maximum(0, np.minimum(bounds[2:], image.size)-np.maximum(bounds[:2], 0)))
    return pixels.astype(np.float32), {"valid": True, "bounds_pixels": bounds.tolist(),
        "padding_fraction": float(1-area/(side*side)),
        "box_intersection_fraction": float((right-left)*(bottom-top)/np.prod(b[2:]-b[:2]))}


def normalized_crop(pixels):
    return torch.from_numpy(((pixels-RGB_MEAN)/RGB_STD).transpose(2, 0, 1).copy())


def crop_examples(tracks, valid_limit=3):
    """Show valid crops first, followed by one explicit missing observation."""
    valid = [i for i, t in enumerate(tracks) if t["crop"]["valid"]]
    missing = [i for i, t in enumerate(tracks) if not t["crop"]["valid"]]
    return valid[:valid_limit]+missing[:1]


def patch_maps(features, seed):
    """Per-crop PCA colors and cosine similarity; neither is segmentation."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3 or min(features.shape) < 3 or not np.isfinite(features).all():
        raise ValueError("Expected a finite spatial feature grid")
    if len(seed) != 2 or any(int(i) != i or not 0 <= i < n for i, n in zip(seed, features.shape[:2])):
        raise ValueError("Patch seed outside feature grid")
    values = features.reshape(-1, features.shape[-1])
    centered = values-values.mean(0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    basis = basis[:3]
    # SVD axes have arbitrary signs; fix them for reproducible local colors.
    signs = np.sign(basis[np.arange(3), np.abs(basis).argmax(1)])
    colors = centered @ (basis*signs[:, None]).T
    low, high = np.percentile(colors, [1, 99], axis=0)
    colors = np.divide(colors-low, high-low, out=np.full_like(colors, .5), where=high-low > 1e-8)
    unit = features/np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
    similarity = np.clip(unit @ unit[tuple(map(int, seed))], -1, 1)
    return np.clip(colors, 0, 1).reshape(*features.shape[:2], 3).astype(np.float32), similarity.astype(np.float32)


def dino_sheet(crop, colors, similarity, seed, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(crop)
    axes[0].plot((seed[1]+.5)*16, (seed[0]+.5)*16, "r+", markersize=14)
    axes[1].imshow(colors, interpolation="nearest", extent=(0, 256, 256, 0))
    heat = axes[2].imshow(similarity, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
    axes[2].plot(seed[1], seed[0], "r+", markersize=14)
    fig.colorbar(heat, ax=axes[2], fraction=.046)
    axes[3].imshow(crop)
    axes[3].imshow(similarity, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest",
                   extent=(0, 256, 256, 0), alpha=.5)
    for ax, title in zip(axes, ("Exact crop; box-centre patch", "Per-crop PCA (arbitrary colors)",
                                "Cosine similarity to marked patch", "Similarity overlay")):
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("16x16 patch features; qualitative only, no semantic or behavior labels", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix("."+suffix), dpi=130)
    plt.close(fig)


def box_membership(points, center, size, yaw):
    """Native ego/point-cloud XYZ and a native z-rotated centered cuboid."""
    center, size = np.asarray(center), np.asarray(size)
    if (center.shape != (3,) or size.shape != (3,) or not np.isfinite([*center, *size, yaw]).all()
            or (size <= 0).any()):
        raise ValueError("Invalid native cuboid")
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (np.abs((points-center) @ rotation) <= size/2+1e-6).all(axis=1)


def relative_xyz(points, center, world_from_native, anchor):
    """Offsets in the fixed initial ego-heading basis, with world-vertical z."""
    yaw = np.arctan2(anchor[1, 0], anchor[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    initial_from_world = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return ((points-center) @ world_from_native[:3, :3].T @ initial_from_world.T).astype(np.float32)


def voxel_sample(points, grid_size=.05):
    """Deterministic one-native-return per voxel, with exact native inverse map.

    Uses the author's FNV grid ordering; picks the first native return instead
    of the random return used by their validation loader. No averaging/TTA.
    """
    if points.ndim != 2 or points.shape[1] != 4 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected finite nonempty XYZ+strength")
    grid = np.floor(points[:, :3].astype(np.float64)/grid_size).astype(np.int64)
    grid -= grid.min(0)
    key = np.full(len(grid), np.uint64(14695981039346656037), dtype=np.uint64)
    for axis in range(3):
        key *= np.uint64(1099511628211)
        key ^= grid[:, axis].astype(np.uint64)
    order = np.argsort(key, kind="stable")
    _, first, inverse_sorted = np.unique(key[order], return_index=True, return_inverse=True)
    selected = order[first]
    inverse = np.empty(len(points), dtype=np.int64)
    inverse[order] = inverse_sorted
    # A hash collision must fail rather than associate unrelated coordinates.
    if not np.array_equal(grid, grid[selected][inverse]):
        raise ValueError("Voxel hash collision")
    return selected, inverse, grid[selected]


class ProbeReadout(nn.Module):
    """Temporary resource graph, not a behavior model or a training runner."""

    def __init__(self, width, seeds):
        super().__init__()
        self.input = nn.Linear(68, width)
        self.seeds = nn.Parameter(torch.randn(1, seeds, width)*.02)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))
        self.output = nn.Linear(width*seeds, 256)

    def forward(self, tokens):
        if len(tokens) == 0:
            return tokens.new_zeros(256)
        values = self.input(tokens).unsqueeze(0)
        query = self.seeds
        value = query+self.attention(query, values, values, need_weights=False)[0]
        value = value+self.ffn(self.norm(value))
        return self.output(value.flatten(1))[0]


def temporal_probe(temporal, values, lengths):
    """Grid-time positions and padding isolation for the temporary graph."""
    length = values.shape[1]
    slots = torch.arange(length, device=values.device)
    padding = slots[None] >= lengths[:, None]
    positions = slots[:, None]*.2
    scales = torch.exp(torch.arange(0, 256, 2, device=values.device)*(-np.log(10000)/256))
    pe = torch.empty(length, 256, device=values.device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(positions*scales), torch.cos(positions*scales)
    values = values.masked_fill(padding[:, :, None], 0)
    result = temporal(values+pe, src_key_padding_mask=padding)
    return result.masked_fill(padding[:, :, None], 0)


def crop_sheet(image, box, crop, info, output):
    from PIL import ImageDraw, ImageOps
    sheet = Image.new("RGB", (1024, 420), "#161616")
    draw = ImageDraw.Draw(sheet)
    if image is not None:
        fitted = ImageOps.contain(image, (740, 390))
        sheet.paste(fitted, (0, 25))
        if box is not None:
            b = np.asarray(box)*np.tile(fitted.size, 2)+[0, 25, 0, 25]
            draw.rectangle(tuple(b), outline="yellow", width=2)
        if info["valid"]:
            b = np.array(info["bounds_pixels"])*fitted.width/image.width+[0, 25, 0, 25]
            draw.rectangle(tuple(b), outline="cyan", width=2)
    if crop is not None:
        sheet.paste(Image.fromarray(np.uint8(np.clip(crop, 0, 1)*255)), (760, 35))
    draw.text((5, 5), "Native box (yellow), intended square crop (cyan)", fill="white")
    draw.text((760, 305), "Exact 256x256 input" if info["valid"] else info["reason"], fill="white")
    sheet.save(output)


def bev_sheet(points, labels, boxes, names, output):
    """Same native XYZ in raw/semantic BEV and a height-preserving oblique view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon
    colors = plt.get_cmap("tab20")(np.arange(len(names)))
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    finite = np.isfinite(points[:, :3]).all(axis=1)
    p, y = points[finite, :3], labels[finite]
    visible = (np.abs(p[:, :2]) <= 50).all(axis=1)
    p, y = p[visible], y[visible]
    axes[0].scatter(-p[:, 1], p[:, 0], s=.15, c=p[:, 2], cmap="viridis", vmin=-3, vmax=4)
    axes[1].scatter(-p[:, 1], p[:, 0], s=.2, c=colors[y])
    axes[2].scatter(-p[:, 1], p[:, 0]*.45+p[:, 2]*3, s=.2, c=colors[y])
    for ax, title in zip(axes, ("Raw height BEV", "Semantic predictions BEV", "Oblique: forward/height (not metric BEV)")):
        ax.set(title=title, xlabel="right (m)", ylabel="forward (m)")
        ax.set_aspect("equal")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_facecolor("#151515")
    for box in boxes:
        center, size, yaw = box
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])*np.array(size[:2])/2
        corners = corners @ np.array([[c, s], [-s, c]])+center[:2]
        for ax in axes[:2]:
            ax.add_patch(Polygon(np.c_[-corners[:, 1], corners[:, 0]], fill=False, edgecolor="white", linewidth=.8))
            ax.add_patch(plt.Circle((-center[1], center[0]), 15, fill=False, color="white", alpha=.2))
    fig.legend(handles=[Patch(color=colors[i], label=n) for i, n in enumerate(names)],
               loc="lower center", ncol=8, fontsize=8)
    fig.suptitle("Frozen nuScenes semantic head: qualitative transfer check, no segmentation GT")
    fig.tight_layout(rect=(0, .11, 1, .95))
    fig.savefig(output, dpi=150)
    plt.close(fig)
