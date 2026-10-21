# BEVContrast: Self-Supervision in BEV Space for Automotive Lidar Point Clouds

Sautier et al., 3DV 2024. Initial candidate screen **2026-10-15**, using the [primary paper record](https://arxiv.org/abs/2310.17281), [official implementation/model listings](https://github.com/valeoai/BEVContrast) and [SemanticKITTI pretraining configuration](https://github.com/valeoai/BEVContrast/blob/main/cfgs/pretrain_sk_minkunet.yaml). This is not a full methods/results review or runtime verification.

The repository provides self-supervised outdoor LiDAR backbones: sparse SR-UNet checkpoints pretrained separately on nuScenes or SemanticKITTI, and a nuScenes VoxelNet checkpoint. BEV is the self-supervision space; the released models include spatial sparse-convolutional feature encoders. RGB projection is not a required inference dependency. Spatial features could be selected around our supplied pedestrian track and its surroundings before temporal labeling.

The SemanticKITTI recipe specifies MinkUNet18, four input channels and 128 output channels. The exact feature construction, intensity availability/normalization, voxelization and mapping to original points must be checked before using the checkpoint on LOKI or Waymo. Architecture-level point support is not proof of checkpoint compatibility. MinkowskiEngine/TorchSparse/SPConv variants require different environments; the repository warns that weights do not transfer easily between sparse libraries.

**Relevance:** a SemanticKITTI-pretrained checkpoint offers a candidate without documented deliberate Waymo/LOKI pretraining, unlike Utonia. This still requires a checkpoint-specific provenance audit; no recording-level disjointness is certified. nuScenes exposure also matters for a later auxiliary-source comparison. No behavior-state performance is established by segmentation results.

Compare with [Utonia](utonia.md), [ScaLR](scalr.md) and [TARL](tarl.md). Candidate only; the [direction review](../offline-labeling-research-direction.md) remains a recommendation, not an encoder selection.
