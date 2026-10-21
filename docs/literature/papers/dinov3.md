# DINOv3

Siméoni et al., August 2025. Focused encoder check, 2026-10-13: [paper](https://arxiv.org/pdf/2508.10104), Gram-anchoring/dense features and model-family sections; [official implementation](https://github.com/facebookresearch/dinov3); [model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md).

DINOv3 uses self-distillation, patch objectives and Gram anchoring to preserve dense feature quality at scale. It releases smaller distilled image encoders as well as a large teacher. Natural-image models use **LVD-1689M**, curated from Instagram images; SAT-493M is a distinct satellite-pretrained family. Models expose spatial features and official checkpoint access, subject to the released license/access process. Dense transfer results do not establish pedestrian-state transfer. [Paper](https://arxiv.org/pdf/2508.10104); [model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md).

This is a relevant newer alternative to DINOv2, not a reason to run an encoder search. If checkpoint access and extraction are straightforward, choose one small natural-image model before behavior comparisons; otherwise DINOv2 is an adequate reproducible reference. Hold the chosen encoder fixed across pooling/query ablations.

No exact intersection with our recordings can be verified from the public dataset description. More pretraining data does not imply cleaner zero-shot provenance. Do not use the satellite family merely because it is newer, or transfer the large teacher to the 4080 without a resource check.

[Direction review](../offline-labeling-research-direction.md).

## Concrete crop interface (2026-10-16)

The [official model definitions](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py) specify **384** channels for ViT-S/16 and **768** for ViT-B/16, and expose normalized CLS and spatial patch tokens separately. At **256 × 256**, 16-pixel patches give **256 patch tokens**, independently of special tokens. Network input size can be chosen; this does not require resizing the full scene before taking a target crop. The [official natural-image transform](https://github.com/facebookresearch/dinov3#image-transforms) illustrates 256-pixel square input with ImageNet RGB mean/std. Aspect-preserving crop/letterbox is our proposed preprocessing choice, not the authors' demonstrated default. [E002a](../../../experiments/E002a-feature-feasibility/README.md#rgb-branch) proposes frozen ViT-B/16 as a bounded first choice, subject to checkpoint access and extraction feasibility; no behavior advantage over DINOv2 is established.
