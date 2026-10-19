# E002 — Multimodal offline reference

**Working protocol, updated 2026-10-19.** The bounded [E002a feature-feasibility probe](../E002a-feature-feasibility/README.md) is authorized; behavior training and full-cache preparation remain deferred until its native/resource evidence is reviewed. The choices below supersede the 16 October discussion proposals. No modality benefit or methodological novelty is assumed.

## Question and interpretation

Do pedestrian appearance and local 3D scene evidence improve offline behavior labeling beyond the existing geometry inputs, and do gains survive transfer and missing observations? Establish a useful, controlled multimodal reference before considering a new fusion/retrieval mechanism. Concatenation, pretrained encoders and bidirectional context are established components, not a methodological novelty claim; see the [literature recommendation](../../docs/literature/offline-labeling-research-direction.md).

The [five-seed review](../E001b-2d-bbox-contribution/results-review.md) motivates geometry as a pragmatic reference, not a universal winner. E001b geometry BiLSTM scores, mean ± sample SD in percentage points: ROAD→ROAD **52.18 ± 1.55**, ROAD→LOKI **45.57 ± 1.10**, LOKI→LOKI **62.03 ± 1.61**, LOKI→ROAD **47.53 ± 0.91**. Geometry is strongest in LOKI→ROAD; other inputs win elsewhere or are effectively tied. Selecting this representation follows inspection of both datasets' test results: disclose **test-informed development**, even with subsequent source-only checkpoint selection. Frozen encoders also retain their pretraining exposure.

## Retained contract

Reuse E001/E001b's frozen complete tracks, eligible population, 5 Hz grid, four-state targets, source-training normalization, split identities and native associations. Preserve 3D-only LOKI tracks, unlabeled context and internal observation gaps. Produce four logits for every non-padding slot; accepted GT alone contributes to track-averaged loss and metrics. Do not silently substitute the native centers/dimensions from E001c.

Existing [experiment partitions](../E001-kinematic-transfer/README.md#splits-and-access) stay frozen. They are not the official Waymo partitions: all acquired annotated ROAD clips used by E001 originate in official Waymo training. The [official-download verification](../../docs/datasets/road-waymo.md#official-download-verification-2026-10-16) owns this evidence; no split migration is proposed here.

## Geometry branch: exact E001b input

**20 values: 14 continuous + six flags**, using frozen source-training normalization. K+T+R comprises pedestrian XY velocity, displacement from first valid position, pedestrian-minus-ego XY position and **relative velocity `v_ped − v_ego`**, with one validity flag per vector. Bbox inputs are normalized horizontal center, **bottom vertical coordinate**, width, height, horizontal-center velocity and **bottom vertical velocity**, plus bbox/bbox-velocity validity. Retain the [E001b definitions](../E001b-2d-bbox-contribution/README.md#additional-observation-contract), source timestamps/immediate-neighbor derivatives, no interpolation and zero missing numerics after normalization. ROAD native Waymo FRONT only, no ROAD fallback; LOKI native boxes. Do not substitute E001c centers/sizes. Working geometry MLP: `20 → 64 → 64`, GELU; no additional hidden architecture search.

## RGB branch

Frozen **DINOv3 ViT-B/16 LVD-1689M**, 256² RGB input and its **768-channel normalized CLS descriptor**. One centered square crop of side `1.5 × max(native bbox width, height)`; preserve its intended location at image boundaries and pad with pretrained RGB mean. No image intersection of the target box means missing RGB; retain partial boxes and record crop padding/intersection. Numeric bbox geometry remains unclipped. Project CLS to256. No full-image branch, crop augmentation or patch retrieval initially. [Official model/interfaces/transforms](https://github.com/facebookresearch/dinov3); actual checkpoint access and native extraction are E002a gates, not assumed success. Web-pretraining overlap remains unknown.

## Point-cloud branch

Frozen **nuScenes-only D-DITR distilled + semantic-segmentation-fine-tuned**: strict checkpoint loading, XYZ+return strength input and **64-channel decoded spatial features before the 16-class semantic head**. Use the [screened recipe and exposure qualifications](../../docs/literature/papers/ditr.md), not injected DITR or the Waymo-containing multi-dataset student. Encoding occurs once per native scene/timestamp before pedestrian selection. Preserve voxel coordinates/native inverse mapping. [E002a](../E002a-feature-feasibility/README.md#encoder-contracts) owns exact pinned weights/code, strength/axis adapters, voxelization and repeatability measurements.

Use a **15 m horizontal cylinder**, no height cutoff, centered on the observed native target position. Keep all cropped spatial tokens initially; measure before choosing a cap or downsampling. Attach pedestrian-relative XYZ in E001's **fixed initial ego-heading axes** and a native rotated-cuboid membership flag: `feature64 + XYZ3 + inside_box1 =68`. Native box dimensions/heading select points; they are not extra geometry-MLP inputs. Geometric containment does not guarantee point ownership. Never interpolate across missing target observations.

Readout: shared learned **unconditioned** seed tokens, one cross-attention block with four heads and an FFN, followed by concatenation of seed outputs and projection to256. Geometry/RGB do not condition these seeds initially. Probe internal widths128/256 and one/four seeds for resources; later source-validation pilot compares one/four seeds at a fixed chosen width. No full point-to-point self-attention adapter. This PMA-style aggregation is established [Set Transformer machinery](https://proceedings.mlr.press/v97/lee19d.html), not a novel method claim.

Fixed mean/max is a cheap possible control, not the chosen default: it discards explicit feature-position associations, whereas the proposed readout retains them until task-trained aggregation. DINO CLS is itself learned compression, not a guarantee of retaining behavior-relevant scene relations. Full-track-conditioned spatial retrieval remains a conditional E003 hypothesis, not required for this reference.

## Fusion and temporal head

| Stage | Working shape |
|---|---|
| Geometry |20 →64|
| RGB |768 →256|
| Point readout |N×68 →256; internal width128/256 measured first|
| Concatenation |64+256+256+3 flags =579|
| Fusion |579 →256|
| Temporal |Two full-track bidirectional pre-norm Transformer layers, width256, four heads, FFN1024, dropout0.1|
| Output |Four logits at every non-padding timestep|

Use sinusoidal elapsed nominal-grid-time positions. Full noncausal attention within tracks, never across tracks. Mask padding; retain internal missing/unlabeled context. Both backbones remain frozen/eval; source behavior training updates geometry MLP, readout, projections/fusion, Transformer and classifier. Frozen extraction precision is separately recorded; trainable modules use FP32 deterministic PyTorch operations.

The three additional flags are `rgb_crop_valid`, `target_points_valid`, `context_points_valid` (local points outside the target box). Zero missing descriptors and mask projected branch embeddings so biases do not simulate observations. Partial point support remains usable. Corrupt/nonfinite input fails. Initial natural missingness and ordinary network dropout only; modality dropout is deferred to one possible bounded later comparison.

## Bounded comparison and safeguards

Four eventual availability-matched variants: **G+A**, **G+A+RGB**, **G+A+LiDAR**, **G+A+RGB+LiDAR**. G is exact E001b geometry; A supplies the same three new usability flags in every variant. Keep branch slots and fusion/temporal width fixed, with absent descriptors disabled; report trainable parameter counts. G+A is a new Transformer reference, not the historical BiLSTM. A pure-G Transformer is only essential if claiming a temporal-model improvement separately.

Train separately on each source, evaluate both datasets after development settings are frozen, retaining all eligible tracks and E001 loss/metrics/strata plus new modality support denominators. The point branch is LiDAR-only **at inference** but inherits visual distillation and nuScenes semantic supervision. Different DINO teachers across branches do not imply aligned representations or establish complementarity. With five seeds the four variants would cost40 attempts/80 cells; full execution remains unapproved.

Preserve source-only behavior normalization/checkpoint selection and explicitly disclose existing **test-informed representation development**. No target fitting or encoder updates. Frozen Utonia would not erase its Waymo exposure; the selected nuScenes recipe avoids deliberate Waymo/LOKI pretraining subject to checkpoint lineage. Single-observation preprocessing is not permission to fit target-corpus statistics.

## Deferred training pilot

After E002a evidence review, fix cache layout, internal spatial width and resource limits. Then the intended vanilla full-multimodal pilot is **four attempts per source**, seed0 and source train/validation only:

1. One readout seed at learning rates1e-4,3e-4,1e-3.
2. Four seeds at that source's selected rate.

Retain AdamW betas(.9,.999), epsilon1e-8, weight decay1e-4, clip1 and equal-track CE. Effective batch64 complete tracks via correctly track-weighted microbatch accumulation, including final partial batch. Budget100 epochs/patience15, three-epoch warmup then constant LR; retain strict-F1 patience and F1/CE/earlier-epoch selection semantics. No modality augmentation initially. Source-validation choice is then held fixed across that source's eventual modality variants. These are bounded working settings, not a proven robust optimum. No pilot training now.

## Before implementation and execution

Execute only [E002a](../E002a-feature-feasibility/README.md): verify actual encoder/native contracts, semantic BEV/box alignment and RGB crops; preserve point-coordinate correspondence; measure all-token readout/whole-track shapes and cache/storage estimates. Review evidence before full preparation or training implementation. Unsupported operations and logging/extraction failures remain visible; do not silently substitute checkpoints, truncate tracks or change native populations.

**Results:** no behavior runs. Native extraction/resource results belong to E002a; remaining decisions are cache representation/resource limits/internal spatial width after that report, plus verified RGB weight access. An empirical paper does not require a new architecture; any later mechanism claim needs a falsifiable matched comparison.
