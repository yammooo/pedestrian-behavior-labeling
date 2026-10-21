# DINOv2: Learning Robust Visual Features without Supervision

Oquab et al., 2023 preprint / TMLR 2024. Reviewed 2026-10-13: [full paper](https://arxiv.org/pdf/2304.07193), pretraining and dense-transfer sections; [official models](https://github.com/facebookresearch/dinov2); [feature API](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py).

DINOv2 pretrains image ViTs by global/patch self-distillation on curated LVD-142M images and distills smaller backbones. It demonstrates frozen transfer to classification, segmentation and depth. The implementation exposes `forward_features()` with normalized CLS and **patch tokens**, and `get_intermediate_layers(..., reshape=True)` for spatial maps. ViT-S/14 produces 384-dimensional tokens. Official checkpoint links and extraction code exist. It consumes RGB images, not calibrated point clouds, trajectories or temporal state sequences. [Paper](https://arxiv.org/pdf/2304.07193); [API](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py).

This is a viable frozen RGB candidate: cache target/context features, retaining patch coordinates for a later matched query comparison. Camera calibration is unnecessary for native 2D crops. It becomes necessary when projecting LiDAR or 3D-only tracks into images.

Generic pretraining is not a behavior-recognition result or proof that tiny pedestrians retain relevant detail. A crop enlargement/resolution check is cheaper than encoder fine-tuning. Do not conflate missing target crops with missing scene RGB or certified out-of-FOV status.

LVD is not an auditable list of our exact recording IDs. No deliberate Waymo/LOKI pretraining was established, but zero overlap cannot be certified. Record the precise checkpoint and qualify external-pretraining access. Our scratch kinematic/small scene reference remains valuable for dataset-excluded claims.

[DINOv3 alternative](dinov3.md) · [Direction review](../offline-labeling-research-direction.md).

## Concrete crop interface (2026-10-16)

The [official model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) specifies patch size **14** and token widths **384/768/1024/1536** for Small/Base/Large/giant. At **224 × 224**, the image yields one CLS token and **256 patch tokens**, plus four register tokens for register checkpoints. Larger patch-compatible image sizes are supported; a fixed crop size is our batching/cost choice, not a requirement that native images already have that size. The [feature API](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py) exposes normalized CLS and patch features separately. E002a can start from one crop descriptor without claiming spatial-token reasoning; its [draft](../../../experiments/E002a-feature-feasibility/README.md#rgb-branch) owns the proposed crop and pooling policy.
