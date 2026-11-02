# Towards Good Practices for Missing Modality Robust Action Recognition

Woo et al., **AAAI 2023**; ActionMAE. Initially reviewed 2026-10-13; expanded **2026-11-02** against the [full paper, v2](https://arxiv.org/html/2211.13916v2), §§2–4 and appendix, and [official implementation](https://github.com/sangminwoo/ActionMAE).

## What is masked and reconstructed

Per-frame ResNet features feed a temporal Transformer for each modality. Each produces one clip summary; ActionMAE masks these **modality summaries after temporal encoding**, rather than individual frame observations. Its Transformer encoder reads the remaining summaries plus a learned memory token. A decoder inserts dummy tokens at missing positions and reconstructs their features; the resulting summaries feed fusion and one clip-action prediction. At least one modality remains. [Paper, §§2.1, 3.1](https://arxiv.org/html/2211.13916v2#S3.SS1).

Training combines classification CE with MSE on synthetically hidden modality summaries. Missing summaries are reconstructed at inference too. Experiments use RGB/depth/infrared subsets on NTU60/120 and RGB/depth on NW-UCLA/UWA3D. Their selected space encoders use ImageNet initialization and are jointly trained; this is not our frozen-feature recipe. Reported comparisons improve missing-modality robustness. [Paper, §§3.2–4.5](https://arxiv.org/html/2211.13916v2#S3.SS2).

## Why it matters for us

**Two separate ideas:** [ModDrop](moddrop.md)-style masking can train the existing classifier directly; ActionMAE adds a reconstruction objective and learned reconstruction module. A masking-only experiment should not be described as an ActionMAE implementation.

The paper supports testing missing-modality behavior and fusion, but its clip summaries do not establish how to handle time-varying gaps within our densely labeled tracks. Its controlled modality-subset experiments also do not establish ROAD-Waymo↔LOKI generalization. Natural missingness in our [frozen populations](../../datasets/README.md#measured-rgb-and-lidar-availability) can correlate with scene and annotation selection.

**A possible later auxiliary objective:** reconstruct only features that genuinely existed and were deliberately hidden. Naturally missing LOKI RGB has no measured RGB-feature target; a reconstruction cannot be treated as a recovered observation or trustworthy GT.

Our LiDAR descriptor comes from a trainable spatial readout. Reconstructing it would require a defined target and potentially computing the hidden readout, reducing the computational savings of skipping masked LiDAR. That is an implementation question for our adaptation, not a result from this paper.

## Practical decision and boundaries

First compare ordinary structured observation masking with its matched unmasked control. If reconstruction is later justified, compare masking alone against masking plus reconstruction with the same visible observations, initialization, behavior labels and accounted compute. Report natural availability, fixed missingness cases, per-class F1 and CE; do not infer reliable recovery when decisive evidence is absent.

The official repository provides training code; its README lists PyTorch 1.11, CUDA 11.3 and Apex. Compatibility with our environment and a usable task checkpoint were not verified.

[Cross-paper overview](../README.md#observation-masking-precedents) · [Direction review](../offline-labeling-research-direction.md).
