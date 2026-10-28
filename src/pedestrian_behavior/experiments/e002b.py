"""Frozen E002b model variants and model construction; no run lifecycle."""

from pedestrian_behavior.models.multimodal import ARM_FLAGS, MultimodalClassifier


ARMS = ("g", "g-rgb", "g-dditr")
VARIANTS = {
    "norm": {"geometry_width": 64, "rgb_width": 128, "spatial_width": 128,
             "temporal_width": 256, "temporal_ffn": 1024, "normalize_branches": True},
    "temporal-small": {"geometry_width": 64, "rgb_width": 128, "spatial_width": 128,
                        "temporal_width": 128, "temporal_ffn": 512, "normalize_branches": True},
    "branches-small": {"geometry_width": 32, "rgb_width": 64, "spatial_width": 64,
                       "temporal_width": 256, "temporal_ffn": 1024, "normalize_branches": True},
}


def model_spec(arm, variant, dropout=.1):
    if arm not in ARMS:
        raise ValueError("E002b supports only g, g-rgb and g-dditr arms")
    if variant not in VARIANTS:
        raise ValueError("Unknown E002b model variant")
    settings = VARIANTS[variant]
    flags = ARM_FLAGS[arm]
    widths = {"geometry_valid": settings["geometry_width"],
              "rgb_valid": settings["rgb_width"], "lidar_valid": settings["spatial_width"]}
    fusion_input = sum(widths[name] for name in flags) + len(flags)
    spec = {
        "variant": variant,
        "availability": list(flags),
        "geometry_mlp": [20, settings["geometry_width"], settings["geometry_width"]],
        "branch_layer_norm": {"enabled": True, "affine": True, "eps": 1e-5,
                              "mask_after_norm": True},
        "fusion": [fusion_input, settings["temporal_width"]],
        "temporal": {"width": settings["temporal_width"], "heads": 4, "layers": 2,
                     "ffn": settings["temporal_ffn"], "attention_bias": True,
                     "rope_base": 10000, "slot_seconds": .2, "bidirectional": True},
        "classifier": [settings["temporal_width"], 4],
        "activation": "GELU", "dropout": dropout, "layer_norm_eps": 1e-5,
        "trainable_precision": "float32", "attention_backend": "torch-SDPA-math",
    }
    if "rgb_valid" in flags:
        spec["rgb_projection"] = [768, settings["rgb_width"]]
    if "lidar_valid" in flags:
        width = settings["spatial_width"]
        spec.update(point_projection=[68, width],
                    spatial={"width": width, "heads": 4, "seeds": 1,
                             "layers": 1, "ffn": width * 4, "attention_bias": False})
    return spec


def build_model(arm, variant, profile, dropout=.1):
    """Construct one frozen E002b model; execution and fitting stay with callers."""
    model_spec(arm, variant, dropout)
    return MultimodalClassifier(arm, profile, temporal_dropout=dropout, **VARIANTS[variant])
