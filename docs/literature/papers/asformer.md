# ASFormer: Transformer for Action Segmentation

Yi, Wen and Jiang, **BMVC 2021**. Reviewed 2026-10-13: [full paper](https://arxiv.org/pdf/2110.08568), §§4–5 and Tables 4–7; [official implementation](https://github.com/ChinaYi/ASFormer). The earlier literature index incorrectly called its venue ICCV; this review corrects it.

ASFormer labels each timestep of an untrimmed video from already extracted temporal features. Dilated temporal convolutions, progressively enlarged attention windows and iterative cross-attention refinement encode local-to-global structure. It evaluates 50Salads, GTEA and Breakfast using frame accuracy, segmental F1 and edit score. On 50Salads, Table 4 gives segment F1@50 of 76.0 with its hierarchy versus 55.1 without it. This is not per-class frame F1. The repo provides training/inference code and benchmark data/features; these are not pedestrian-behavior pretrained weights. [Paper, §§4–5](https://arxiv.org/pdf/2110.08568).

Offline dense classification, future context and hierarchical temporal refinement are already established. ASFormer is a useful **adapted temporal baseline**, not proof that a Transformer must replace our BiLSTM.

For our tasks, replace the feature dimension/head and mask unlabeled GT independently from temporal context and padding. Annotation gaps must not be relabeled as actions or silently joined into scored segments. Smoothing/refinement can suppress real short states; first assess class/boundary effects with frame metrics. Any event/segment claim needs separately frozen definitions.

Its evidence supports economical local temporal bias under limited data, but comes from different activities/features. Keep the temporal head fixed while studying scene information. Adopt ASFormer-like refinement only if fragmentation is a diagnosed problem; a full decoder stack is unnecessary to establish whether RGB or LiDAR helps.

[Direction review](../offline-labeling-research-direction.md).
