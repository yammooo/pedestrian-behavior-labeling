# Point Transformer V3: Simpler, Faster, Stronger

Wu et al., CVPR 2024. Reviewed 2026-10-13: [full paper](https://arxiv.org/pdf/2312.10035), architecture/efficiency and outdoor experiments; [official code/model status](https://github.com/Pointcept/PointTransformerV3).

PTv3 organizes point neighborhoods through serialized spatial orders, uses hierarchical pooling and efficient attention, and supports point-level perception features. The paper reports roughly 3× speed and 10× memory improvements over PTv2 under its stated benchmark, and evaluates indoor/outdoor segmentation and detection. PTv3 is a **backbone**, not one universally pretrained model or a temporal encoder. The official repository warns that some old released weights are invalid after structural changes and that Waymo-trained weights cannot be released. [Paper](https://arxiv.org/pdf/2312.10035); [repo](https://github.com/Pointcept/PointTransformerV3).

Spatial point features can be queried by a pedestrian trajectory, but those semantics are not trained by default. Actual input feature channels, voxel scale, pooling inverses and checkpoint compatibility must be checked for the chosen model. XYZ support at architecture level does not establish that a specific color/normal-pretrained checkpoint works on XYZ-only LiDAR.

For frozen features, [Sonata](sonata.md), [Concerto](concerto.md) and [Utonia](utonia.md) are distinct candidates with different exposure/availability assumptions. Do not attribute their pretraining to PTv3 generally or select a checkpoint by its benchmark score without checking its data.

Scratch PTv3 fine-tuning is a large step relative to our current small behavior heads. Prefer a compact target/local-scene point or BEV reference first. Its published efficiency figures do not estimate our 4080 runtime, complete-track memory or cache size; those would need a separately authorized small feasibility pilot.

## Concrete outdoor checkpoint follow-up (2026-10-15)

Public Hugging Face file metadata and small saved configs were fetched at pinned revisions; no model tensors were downloaded or executed. Local evidence: ignored `outputs/inspection/literature/20261015-ptv3-checkpoints/`. The user is open to Waymo pretraining if a compatible original split can be reproduced; this does not change any existing split or result.

### First candidate: official nuScenes semantic-segmentation PTv3

- Repository: [`Pointcept/PointTransformerV3`](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/nuscenes-semseg-pt-v3m1-0-base), revision `5e25cfaf759289b03db1f90f143b3dc5172929d3`.
- File: `nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth`, **554,519,016 bytes**; published LFS SHA-256 `e2774d2fa1dd33e640a514afe0bd1e5af94eb08be19d08d1f9402f20dfd6db94`.
- The saved config trains `NuScenesDataset` on `train`, validates on `val`, and uses supervised 16-class segmentation. Its `test` entry also points to `val`; this release is not an official nuScenes test-set evaluation. A tuple naming nuScenes/SemanticKITTI/Waymo in disabled conditional-normalization settings does **not** establish multi-dataset training.
- Inputs are XYZ plus LiDAR return strength (`in_channels=4`, `feat_keys=('coord','strength')`), without camera color or normals. The backbone has encoder widths 32/64/128/256/512 and a spatial decoder with 64 output channels before the 16-class head. We could extract spatial features before that head and retain target/local-road context for the behavior model.
- Training/validation use 5 cm grid sampling; the saved test pipeline adds 2.5 cm presampling, 5 cm voxelization and multiple scale/flip votes. A cheaper frozen-feature pipeline would be a declared adaptation, not an exact benchmark reproduction.

This is a more direct outdoor LiDAR candidate than the original indoor Sonata checkpoint and has no deliberate Waymo training in its listed recipe. It carries **nuScenes semantic-label supervision**, not merely unlabeled pretraining. LOKI's inspected PLY reader already expects XYZ/intensity; cross-sensor strength units/distributions still need verification. Native intensity availability alone does not guarantee transfer.

**Compatibility gate:** the original repo warns about a PTv3 structural change; Pointcept documents released weights as targeting v1.5.1. Pin the corresponding implementation and verify an exact state-dictionary load before treating this as usable. File existence and a saved config do not settle compatibility. [Release history](https://github.com/Pointcept/Pointcept/releases); [maintainer repo](https://github.com/Pointcept/Pointcept).

### Second candidate: D-DITR on nuScenes

[D-DITR](ditr.md) supplies a PTv3 distilled from DINOv2 and an additional segmentation-fine-tuned version, both with LiDAR-only inference. The latter's saved config uses the same XYZ/strength input and 64-channel spatial output. Its **554,518,952-byte** `model_best.pth` is published under `rwthvision/ditr`, revision `a656b4385aa28060b30152676023903a923b4a17`, at `nuscenes/semseg-ptv3_weight=distill-ptv3_nuscenes_dino-L/model/model_best.pth` (LFS SHA-256 `93dbb9d3fda247adf56e1c1ebd419d6ada4ebe4dc141110c606926d3ad67b30d`). This is an alternative initialization, not a reason to benchmark all image teachers. Distilled and image-injected checkpoints must not be confused.

### Why simply adopting the original Waymo split is insufficient today

The **original supervised PTv3 Waymo run** exposes config/logs but no weight file in the verified model listing; its authors explicitly call it log-only. This statement does not apply to every later Waymo-pretrained PTv3 derivative.

[Utonia's actual pretraining configs](utonia.md#verified-pretraining-partitions-2026-10-15) include **both Waymo training and validation**. Moreover, all **798** annotated ROAD clips in our acquired train/validation release originate in **Waymo training**, including our E001 held-out subset. [Native split provenance](../../datasets/road-waymo.md#acquired-index-and-access-2026-10-02). Relabeling our current splits cannot create a Waymo-validation behavior test. A new protocol needs recording-disjoint behavior annotations relative to actual checkpoint exposure, or an explicitly exposure-qualified practical evaluation. No split changes, annotation acquisition, preprocessing or reruns were performed.

Other screened files: `concerto_large_outdoor.pth` exists (**830,815,983 bytes**), but its exact outdoor partitions/input handling remain less explicit; generic Concerto names can denote prerelease Utonia. Autoware publishes a PTv3 encoder ONNX (**112,786,044 bytes**) trained on internal T4Dataset, with no public evaluation metrics in its model card; deployment plugins, limited training provenance and no released PyTorch weights make it a weaker initial research choice. [Concerto card](https://huggingface.co/Pointcept/Concerto); [Autoware card](https://huggingface.co/AutowareFoundation/ptv3).

**Recommendation:** first verify the official nuScenes PTv3's code/weight/input compatibility; retain D-DITR as the most relevant image-distilled alternative. Both support an independent point branch without coloring our clouds. No behavior performance or RTX 4080 memory/throughput is established.

[Direction review](../offline-labeling-research-direction.md).
