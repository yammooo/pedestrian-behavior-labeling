# Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations

Zhang et al., NeurIPS 2025. Focused encoder/provenance check, 2026-10-13: [v1 paper](https://arxiv.org/pdf/2510.23607v1), method/probe sections; [official API](https://github.com/Pointcept/Concerto); [model card](https://huggingface.co/Pointcept/Concerto).

Point self-distillation and image-to-point joint embedding prediction use calibrated pixel/point correspondences. The paper reports ScanNet linear-probe mIoU 77.32 versus Sonata 72.52 and their feature concatenation 75.91. It supplies spatial point encoders, not a temporal behavior head. **The current model card identifies default large/base/small/tiny weights as prerelease Utonia variants; original Concerto is `concerto_base_origin.pth`.** An outdoor checkpoint is separately listed. [Paper](https://arxiv.org/pdf/2510.23607v1); [model card](https://huggingface.co/Pointcept/Concerto).

This is direct precedent for complementary RGB/3D representation learning beyond concatenation. It does not make new behavior-specific supervision redundant, but it prevents claiming multimodal alignment alone as new.

Checkpoint **filenames and actual recipe** matter more than the repository name. The original paper's indoor provenance cannot be assigned to a default Utonia-derived weight. Missing-color performance and outdoor exposure must be checked per variant. Training alignment requires calibration; using pretrained point features may not, provided the chosen input contract is met.

Prefer explicit [Utonia](utonia.md) metadata for a practical XYZ-only candidate rather than relying on ambiguous family defaults. A known Waymo-pretrained variant belongs to a disclosed external-exposure setting, not the clean reference.

## Waymo-excluded checkpoint check (2026-10-16)

The [current official card](https://huggingface.co/Pointcept/Concerto) still identifies generic size-named weights as prerelease Utonia and separately lists `concerto_large_outdoor.pth`, without an exact checkpoint-specific outdoor dataset/split manifest. Generic names are therefore not a verified escape from Utonia exposure; missing provenance is not evidence of Waymo-free training. The original indoor `concerto_base_origin.pth` remains distinct, with substantial input/domain mismatch to our LiDAR. No verified Waymo-excluded outdoor Concerto checkpoint was established. Prefer a documented [nuScenes PTv3/D-DITR recipe](ptv3.md#concrete-outdoor-checkpoint-follow-up-2026-10-15) or [ScaLR's explicitly listed outdoor mixture](scalr.md) for the next bounded compatibility screen. No behavior-performance ordering is established.

[Direction review](../offline-labeling-research-direction.md).
