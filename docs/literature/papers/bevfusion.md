# BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation

Liu et al., ICRA 2023; reviewed 2026-10-13: [full paper](https://arxiv.org/pdf/2205.13542), §§III–IV; [MIT implementation/checkpoints](https://github.com/mit-han-lab/bevfusion). This is distinct from Liang et al.'s similarly named NeurIPS 2022 BEVFusion.

Camera features are lifted through depth hypotheses/calibration into a shared BEV grid; LiDAR features enter the same space. Fusion supports detection and map segmentation. The paper reports 1.3-point detection mAP/NDS and 13.6-point map-mIoU improvement over its comparison, with efficient BEV pooling. Official nuScenes detection/segmentation checkpoint links exist. The MIT repository was archived in January 2025 and documents an older Python/PyTorch/CUDA stack. [Paper](https://arxiv.org/pdf/2205.13542); [repo](https://github.com/mit-han-lab/bevfusion).

BEV preserves spatial road relationships better than a single global embedding, but its height compression can lose body articulation and small distant details. It is not target conditioned by default; our supplied 3D track could select local BEV tokens afterward.

Calibrated camera/LiDAR alignment and synchronized ego transforms are practical requirements. The absence of verified LOKI camera calibration blocks an honest paired BEVFusion adaptation today. A **LiDAR-only local BEV** does not have that camera requirement, and should not be confused with the full pretrained fusion method.

nuScenes-trained weights constitute external supervised representation exposure; Waymo leaderboard results do not specify the exposure of every downloadable checkpoint. Audit each chosen checkpoint and inherited RGB/LiDAR backbone. No installation or runtime test was performed.

Use BEV as a representation option, not a compulsory architecture. A small independent 3D scene branch plus native 2D RGB crops can test complementary evidence before sensor-level fusion. Modality robustness must be measured on natural missingness and controlled removal, not assumed from the model family name.

[Direction review](../offline-labeling-research-direction.md).
