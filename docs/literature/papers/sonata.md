# Sonata: Self-Supervised Learning of Reliable Point Representations

Wu et al., CVPR 2025. Reviewed 2026-10-13: [full paper/supplement](https://arxiv.org/pdf/2503.16429), Table 1, representation probes and limitations; [official inference/API](https://github.com/facebookresearch/sonata); [published weight repository](https://huggingface.co/facebook/sonata).

Sonata self-distills PTv3 point features while counteracting geometric shortcuts. It reports ScanNet linear-probe **mIoU** 72.5%, not pedestrian classification accuracy. Indoor pretraining uses approximately 140k real/synthetic scene clouds; outdoor experiments are a separate setup, with further multi-dataset supervised tuning. The default released encoder expects coordinates, color and normals. Its hierarchical point features can be restored to original points using pooling inverses and the grid-sampling inverse. Official weights/code exist; weights carry CC-BY-NC terms. [Paper](https://arxiv.org/pdf/2503.16429); [repo/API](https://github.com/facebookresearch/sonata).

This is a useful frozen spatial representation precedent, but **the default indoor checkpoint and outdoor headline scores are not interchangeable**. Color/normal availability and scan density differ from our raw LiDAR. Estimated normals and zero colors need evaluation, not an assumption of valid transfer. Projecting RGB colors would reintroduce LOKI's unresolved calibration issue.

A source-excluded indoor checkpoint avoids known deliberate Waymo training, but might be unsuitable for outdoor semantics. An outdoor checkpoint must expose its exact training dataset/split. Pretraining only, supervised fine-tuning and downstream decoder training are separate provenance stages.

The more directly applicable missing-channel candidate is [Utonia](utonia.md), with a stronger target-exposure limitation. Do not benchmark the whole family before establishing that a small local-scene branch helps. Source-scene probes and a few visualized feature maps would be the cheapest compatibility check; no such runtime pilot was performed here.

## Waymo-excluded checkpoint check (2026-10-16)

The [paper's outdoor semantic-segmentation section and Table 8](https://arxiv.org/html/2503.16429v1) explicitly describe **joint pretraining on nuScenes, Waymo and SemanticKITTI**. The subsequent supervised tuning is an additional stage; Waymo access is not confined to tuning. Those outdoor results therefore cannot support a Waymo-excluded initialization. The [official default training instructions](https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/sonata/README.md) instead list six indoor datasets and link the original default weights. That indoor checkpoint is a plausible no-deliberate-Waymo representation, but its color/normal contract and indoor-to-driving transfer remain limitations. No verified separate nuScenes-only outdoor Sonata checkpoint was established by this check.

For an outdoor alternative, [D-DITR's nuScenes student](ditr.md) and [ScaLR's multi-dataset outdoor model](scalr.md) are more directly matched candidates. Their listed point-training data exclude Waymo/LOKI; RGB-teacher web-data intersections remain unknown. No evidence establishes their pedestrian-labeling F1 relative to Utonia, and semantic-segmentation scores are not a behavior-transfer ranking. No weights downloaded or executed.

[Direction review](../offline-labeling-research-direction.md).
