# MultiMAE: Multi-Modal Multi-Task Masked Autoencoders

Bachmann et al., ECCV 2022. Reviewed 2026-10-13: [paper/supplement](https://arxiv.org/pdf/2204.01678), §3, transfer/availability experiments; [official code and models](https://github.com/EPFL-VILAB/MultiMAE).

RGB, depth and semantic-map patches enter a shared ViT; visible tokens are sampled across modalities and shallow task decoders reconstruct missing patches. Pretraining uses ImageNet with pseudo depth/segmentation from teacher models. It supports different input subsets and evaluates classification, segmentation and depth transfer. The authors supply pretrained models and input/output adapters. Its signals are **aligned 2D modalities**, not arbitrary raw LiDAR or full-track behavior labels. [Paper, §§3–4](https://arxiv.org/pdf/2204.01678); [repo](https://github.com/EPFL-VILAB/MultiMAE).

Masked multimodal reconstruction and lightweight modality/task adapters are established. A useful RQ2 control is a generic reconstruction objective using the same additional observations as semantic supervision. Better transfer from extra data alone would weaken a claim about heterogeneous semantics.

Do not equate feature reconstruction with recovering trustworthy missing measurements or behavior GT. Alignment, teacher exposure and pseudo-label errors become part of provenance. Training such an encoder from scratch is not a necessary December deliverable; a small auxiliary objective on an already validated representation would be more realistic.

[Direction review](../offline-labeling-research-direction.md).
