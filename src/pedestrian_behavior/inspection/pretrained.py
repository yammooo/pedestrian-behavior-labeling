"""Pinned-author inference adapters, isolated from the behavior training stack."""

import importlib
from pathlib import Path
import runpy
import subprocess
import sys
import types

import numpy as np
import torch
from torch import nn

from pedestrian_behavior.data.preparation import checksum
from .feature_probe import voxel_sample

DITR_REVISION = "f66d1dadb82e97ded7750567a043e4783305b01c"
DINO_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
DITR_SHA256 = "93dbb9d3fda247adf56e1c1ebd419d6ada4ebe4dc141110c606926d3ad67b30d"
DITR_CONFIG_SHA256 = "7cce1838385d3f14830b5d61c88db98cbb62951cea39665311c61bca23f253e3"


def verify_code(path, revision):
    path = Path(path).resolve()
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision or subprocess.check_output(["git", "-C", str(path), "diff", "HEAD"]):
        raise ValueError(f"Unpinned/modified pretrained source: {path}")
    return path


def load_point_author(path):
    """Import only the author's selected PTv3 graph, not unrelated CUDA models.

    Their package initializer eagerly imports every backbone and optional custom
    operator. Namespace packages avoid those side effects; source stays intact.
    """
    for name in ("pointcept", "pointcept.models", "pointcept.utils", "pointcept.models.point_transformer_v3",
                 "pointcept.models.point_prompt_training"):
        package = types.ModuleType(name)
        package.__path__ = [str(path.joinpath(*name.split(".")))]
        sys.modules[name] = package
    prompt = sys.modules["pointcept.models.point_prompt_training"]
    prompt.PDNorm = importlib.import_module("pointcept.models.point_prompt_training.prompt_driven_normalization").PDNorm
    return importlib.import_module("pointcept.models.point_transformer_v3.point_transformer_v3m1_base").PointTransformerV3


class PointEncoder:
    def __init__(self, code, config, checkpoint, device):
        self.code = verify_code(code, DITR_REVISION)
        if checksum(checkpoint) != DITR_SHA256:
            raise ValueError("Unexpected D-DITR checkpoint checksum")
        if checksum(config) != DITR_CONFIG_SHA256:
            raise ValueError("Unexpected D-DITR configuration checksum")
        cfg = runpy.run_path(str(config))
        self.names = cfg["names"]
        model = cfg["model"]
        if model["type"] != "DefaultSegmentorV2" or model["num_classes"] != 16 or model["backbone_out_channels"] != 64:
            raise ValueError("Expected released nuScenes 64-channel segmentation checkpoint")
        backbone = dict(model["backbone"])
        backbone.pop("type")
        backbone["shuffle_orders"] = False
        point_cls = load_point_author(self.code)
        self.model = nn.Module()
        self.model.backbone = point_cls(**backbone)
        self.model.seg_head = nn.Linear(64, 16)
        # The verified author checkpoint includes NumPy scalar metric metadata.
        with torch.serialization.safe_globals([(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                                               np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType]):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).eval().to(device)
        self.device = device

    def __call__(self, native_points):
        # nuScenes raw LiDAR is x-right/y-forward/z-up, unlike our ego frame.
        # This is only an axis rotation; no invented roof-sensor translation.
        points = np.asarray(native_points, dtype=np.float32).copy()
        points[:, :2] = np.c_[-native_points[:, 1], native_points[:, 0]]
        selected, inverse, grid = voxel_sample(points)
        coord = torch.from_numpy(points[selected, :3]).to(self.device)
        data = {"coord": coord, "grid_coord": torch.from_numpy(grid).int().to(self.device),
                "feat": torch.from_numpy(points[selected]).to(self.device),
                "offset": torch.tensor([len(selected)], device=self.device)}
        from pointcept.models.utils.structure import Point
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            output = self.model.backbone(Point(data))
            if not torch.equal(output.coord, coord):
                raise ValueError("Decoded feature coordinates changed order")
            features = output.feat
            logits = self.model.seg_head(features)
        if features.shape != (len(selected), 64) or not torch.isfinite(features).all() or not torch.isfinite(logits).all():
            raise ValueError("Nonfinite/misaligned decoded point features")
        return features.float().cpu().numpy(), logits.float().cpu().numpy(), selected, inverse


class ImageEncoder:
    def __init__(self, code, checkpoint, device):
        code = verify_code(code, DINO_REVISION)
        self.model = torch.hub.load(str(code), "dinov3_vitb16", source="local", pretrained=True,
                                    weights=str(Path(checkpoint).resolve()))
        self.model.requires_grad_(False).eval().to(device)
        self.device = device

    def __call__(self, crops):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            result = self.model.forward_features(crops.to(self.device))["x_norm_clstoken"]
        if result.shape != (len(crops), 768) or not torch.isfinite(result).all():
            raise ValueError("Nonfinite/invalid DINO CLS features")
        return result.float().cpu().numpy()
