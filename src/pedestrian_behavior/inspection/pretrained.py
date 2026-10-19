"""Pinned-author inference adapters, isolated from the behavior training stack."""

import importlib
from pathlib import Path
import runpy
import re
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
DINO_HF_SHA256 = "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
DINO_PTH_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"


def image_state_from_hf(state, expected):
    """Invert the official Transformers DINOv3 ViT-B converter, not the model.

    Mapping reference: transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py.
    Recreate only deterministic buffers and the masked zero key bias; every
    learned parameter must be present. Strict loading follows conversion.
    """
    state = {k.removeprefix("model."): v for k, v in state.items()}
    result, consumed = {}, set()
    for key, initial in expected.items():
        if key == "rope_embed.periods" or key.endswith("qkv.bias_mask"):
            result[key] = initial
            continue
        renamed = key.replace("storage_tokens", "embeddings.register_tokens")
        renamed = {"cls_token": "embeddings.cls_token", "mask_token": "embeddings.mask_token"}.get(renamed, renamed)
        renamed = renamed.replace("patch_embed.proj", "embeddings.patch_embeddings")
        renamed = re.sub(r"blocks\.(\d+)\.", r"layer.\1.", renamed)
        renamed = renamed.replace("attn.proj", "attention.o_proj").replace("attn.", "attention.")
        renamed = re.sub(r"ls(\d+)\.gamma", r"layer_scale\1.lambda1", renamed)
        renamed = renamed.replace("mlp.fc1", "mlp.up_proj").replace("mlp.fc2", "mlp.down_proj")
        if ".qkv." in renamed:
            parts = []
            for name in ("q", "k", "v"):
                part_key = renamed.replace("qkv", name+"_proj")
                if name == "k" and key.endswith("bias"):
                    parts.append(torch.zeros_like(parts[0]))
                else:
                    parts.append(state[part_key])
                    consumed.add(part_key)
            result[key] = torch.cat(parts)
        else:
            value = state[renamed]
            result[key] = value.squeeze(1) if key == "mask_token" else value
            consumed.add(renamed)
        if result[key].shape != initial.shape:
            raise ValueError(f"Unexpected converted DINO shape: {key}")
    if set(state) != consumed:
        raise ValueError(f"Unexpected HF DINO parameters: {sorted(set(state)-consumed)}")
    return result


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
        # The top-level flag is not forwarded to author SerializedPooling layers.
        for module in self.model.backbone.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = False
        self.model.seg_head = nn.Linear(64, 16)
        # The verified author checkpoint includes NumPy scalar metric metadata.
        with torch.serialization.safe_globals([np._core.multiarray.scalar,
                                               (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
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
        # Author segmentation testing keeps sparse/linear weights and values FP32;
        # PTv3 itself casts FlashAttention QKV to FP16. Eval autocast causes mixed
        # sparse-convolution value/weight dtypes in spconv and is unsupported.
        with torch.inference_mode():
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
        # hubconf eagerly imports unrelated segmentation heads/torchmetrics.
        sys.path.insert(0, str(code))
        factory = importlib.import_module("dinov3.hub.backbones").dinov3_vitb16
        if Path(checkpoint).suffix == ".safetensors":
            if checksum(checkpoint) != DINO_HF_SHA256:
                raise ValueError("Unexpected official HF DINOv3 ViT-B/16 checkpoint checksum")
            from safetensors.torch import load_file
            self.model = factory(pretrained=False)
            state = image_state_from_hf(load_file(str(checkpoint)), self.model.state_dict())
            self.model.load_state_dict(state, strict=True)
        else:
            if checksum(checkpoint) != DINO_PTH_SHA256:
                raise ValueError("Unexpected official DINOv3 ViT-B/16 checkpoint checksum")
            self.model = factory(pretrained=True, weights=str(Path(checkpoint).resolve()))
        self.model.requires_grad_(False).eval().to(device)
        self.device = device

    def descriptors(self, crops, *, patches=False):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            output = self.model.forward_features(crops.to(self.device))
            result = output["x_norm_clstoken"]
        if result.shape != (len(crops), 768) or not torch.isfinite(result).all():
            raise ValueError("Nonfinite/invalid DINO CLS features")
        cls = result.float().cpu().numpy()
        if not patches:
            return cls
        tokens = output["x_norm_patchtokens"]
        if crops.shape[2:] != (256, 256) or tokens.shape != (len(crops), 256, 768) or not torch.isfinite(tokens).all():
            raise ValueError("Expected finite 16x16 DINO patch grid for exact 256x256 crops")
        return cls, tokens.float().cpu().numpy().reshape(-1, 16, 16, 768)

    def __call__(self, crops):
        return self.descriptors(crops)
