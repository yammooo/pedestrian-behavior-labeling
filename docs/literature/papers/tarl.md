# Temporal Consistent 3D LiDAR Representation Learning for Semantic Perception in Autonomous Driving

Nunes et al., CVPR 2023; TARL. Initial **2026-10-15 repository-level screen** against the [official implementation](https://github.com/PRBonn/TARL), including data preparation, pretraining-weight link and downstream instructions. Full-paper methods/results and checkpoint execution remain unchecked.

TARL uses temporal LiDAR views, unsupervised ground removal and clustering to construct coarse object segments for self-supervised pretraining. The implementation prepares SemanticKITTI and links pretrained MinkUNet weights. Features are intended for downstream semantic/panoptic segmentation and detection. Temporal pretraining does not by itself make the backbone a complete-track behavior model; our dense temporal labeling head would still be needed.

The documented downstream command uses intensity. Exact pretrained input channels, intensity treatment, voxelization and LOKI support require inspection rather than substituting absent channels silently. MinkowskiEngine and the older downstream toolchain also need a bounded compatibility check in an isolated environment.

**Relevance:** another outdoor LiDAR-pretrained candidate with documented SemanticKITTI preparation, offering a possible alternative to deliberate Waymo exposure. A linked weight is not a verified provenance manifest or successful download. Ground-removal/clustering are described for pretraining; do not automatically impose them on behavior inference or remove road context that our task may need.

Screen [BEVContrast](bevcontrast.md) and [ScaLR](scalr.md) before expanding into multiple sparse-backbone implementations. No numerical comparison or encoder choice is established here. [Direction review](../offline-labeling-research-direction.md).
