# E002a — Native-feature feasibility

Authorized **2026-10-19**. Bounded gate for [E002](../E002-multimodal-reference/README.md), including semantic BEV inspection. No behavior training, held-out scoring, W&B runs or full-cache preparation. Existing E001-family archives, setup, normalization and results stay unchanged. This probe may complete partially if gated DINOv3 weights are unavailable; that state is explicit and cannot pass the RGB execution gate.

## Fixed scope and sample

Seed 0 selects eight training and two validation recording groups **per dataset**, then three native frames from each group's union of frozen selected frames. Add training-only crowded-group and longest-track context stress frames. Stress frames are excluded from random-sample extrapolation. Group selection and stress choices use identities, lengths and counts, never behavior classes or model errors. No test archives are opened. Reuse the accepted clip/scenario independence assumption.

Extract each scene once per selected timestamp and reuse features for all represented eligible pedestrians. Small pickle-free fixtures preserve coordinates, decoded features/logits, native indices/inverse mappings and separate identity references. Sampled source archives and frozen setup/collection manifests are hashed before/after. This is a bounded protection check, not a full filesystem audit.

## Encoder contracts

- **RGB:** frozen DINOv3 ViT-B/16 LVD-1689M; 768-channel normalized CLS. Centered square crop, side `1.5 × max(native bbox width, height)`, sampled to 256² with bicubic resampling. Mean-color out-of-image padding; never shift the crop. No native-box/image intersection means missing RGB. Log crop padding and intersection fractions. E001b native boxes only; no ROAD fallback.
- **Points:** frozen nuScenes D-DITR distilled + segmentation-fine-tuned checkpoint, strict state-dict loading, 64 decoded channels before the 16-class semantic head. Entire selected native scene scan, no temporal accumulation; Waymo retains all sensors/both returns and TOP ego-motion compensation.
- The author's [nuScenes loader](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/datasets/nuscenes.py) uses raw LiDAR XYZ and strength/255; the [Waymo preprocessing](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/datasets/preprocessing/waymo/preprocess_waymo.py) uses tanh(strength). Sampled LOKI PLY strengths are already [0,1] (2026-10-19 native audit); retain them and validate this bound on each probe frame. This aligns numeric ranges, not sensor reflectance physics.
- Our ego XYZ axes are forward/left/up. Rotate encoder input to nuScenes raw-LiDAR right/forward/up: `(-y,x,z)`; [official nuScenes KITTI conversion](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/kitti.py) documents the 90° difference. Do not invent a nuScenes roof-sensor translation. Different sensor origin/height remains a domain-shift limitation.
- Deterministic first native return per 5 cm voxel, author FNV ordering, exact inverse mapping; fixed serialization orders. These declared changes replace random validation sampling/order. No segmentation test-time augmentation or silent point truncation. Point extraction uses FP32 sparse/linear operations and the author's internal FP16 FlashAttention cast, matching its tester; eval autocast produces incompatible sparse value/weight dtypes. DINO uses CUDA FP16 autocast. Save float32 descriptors; test repeatability separately from deterministic FP32 trainable operations.

Point neighborhoods: **15 m horizontal radius in world-horizontal/fixed initial-heading axes**, no height cutoff. Tokens carry `[feature64, relativeXYZ3, inside_native_cuboid1]`. Native heading/full dimensions are used only for membership. A cuboid membership flag is geometric containment, not verified point ownership. No interpolation across missing native boxes/poses.

Pinned author revisions and hashes are enforced in [the adapter](../../src/pedestrian_behavior/inspection/pretrained.py). Weight: `nuscenes/semseg-ptv3_weight=distill-ptv3_nuscenes_dino-L/model/model_best.pth` at Hugging Face revision `a656b4385aa28060b30152676023903a923b4a17`; SHA256 `93dbb9d3fda247adf56e1c1ebd419d6ada4ebe4dc141110c606926d3ad67b30d`. Matching config SHA256 `7cce1838385d3f14830b5d61c88db98cbb62951cea39665311c61bca23f253e3`. [Pretraining/exposure qualifications](../../docs/literature/papers/ditr.md) remain applicable.

## Visual and numerical checks

Save raw-height BEV, semantic BEV and an oblique height-preserving point view with native target boxes and neighborhood circles. These are qualitative predictions from the existing nuScenes segmentation head, **not behavior outputs or measured segmentation accuracy**. Preserve a legend. Save full-image native-box/crop overlays beside the exact model crop, including naturally missing/partial observations. Inspect alignment, sparse/distant targets and gross semantic domain failures before interpreting features.

[CPU acceptance](../../tests/test_feature_probe.py): boundary/outside/missing crops, mean padding, rotated cuboids, fixed-heading transforms, point/strength mask correspondence, voxel inverse mapping, reordered feature-coordinate pairs, empty neighborhoods, finite gradients and development-only repeatable sampling. Native execution additionally checks finite outputs, 768/64 shapes, strict checkpoint loading and repeatability. Existing command behavior is covered by the full repository suite. Frozen extraction may use nondeterministic sparse kernels: recorded numerical repeatability is a measurement, not a cross-platform identity guarantee.

## Resource workload and estimates

Temporary benchmark graph only; no optimizer updates or trained behavior labeler. Shared unconditioned learned seeds, one spatial cross-attention block/four heads/FFN; internal widths 128/256 × one/four seeds; output256. Geometry64 + RGB256 + LiDAR256 + three flags =579 →256. Two bidirectional pre-norm temporal layers, width 256/four heads/FFN1024/dropout0.1, sinusoidal elapsed grid time, four-class readout. Profile FP32 forward/backward for whole-track microbatch shapes 1/2/4/8. Repeat the largest observed neighborhood at every slot of the longest eligible training track: this conservative shape test is **not measured training throughput or actual complete-track extraction**. Record OOMs without retry/truncation.

Estimate training/validation float32 payload and point-GPU extraction time per dataset using the random sample, scaling frame means by eligible scene-slot counts then averaging sampled group totals. Compare whole-scene sharing, per-frame neighborhood union sharing and duplicated pedestrian neighborhoods. Report RGB separately. Estimates exclude test preparation, index/inverse-map overhead, compression and native I/O; warm/cold kernel effects and small sample uncertainty limit precision. Cache representation, spatial width, any token limit and disk ceiling remain undecided until this report is reviewed.

## Environment and command

Official HF `model.safetensors` is also accepted: SHA256 `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` at model revision `5931719e67bbdb9737e363e781fb0c67687896bc`. Its parameter names are converted back using the [official Transformers mapping](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py), then strictly loaded into the pinned original backbone. Reconstruct only deterministic RoPE/bias-mask buffers and the masked zero key bias; learned weights must all be present. No alternate model implementation or Transformers dependency is introduced.

Use an isolated ignored `outputs/experiments/E002a/extractor-env`; never change E001 environments or add CUDA encoder dependencies to the main project. [Pinned direct dependencies](requirements.txt); save the fully resolved freeze/check output with execution evidence. CUDA 12.6 PyTorch wheels and matching prebuilt FlashAttention/torch-scatter avoid the host's older CUDA 11.7 compiler. Author PTv3 source is loaded without its eager imports of unrelated custom-op architectures; source files are not patched. The adapter verifies code revisions and weight/config hashes.

Code moves through Git; verify the remote revision. Native mount on aalto is `/run/media/user20/F47C60057C5FC152/projects`, not the old `/media` path. Native data remains read-only. Example from repository root (set paths to the pinned author clones/weights):

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/probe-e002a.py \
  --device cuda --output "$probe_root/native-probe" \
  --ditr-code "$probe_root/dependencies/ditr" \
  --ditr-config "$probe_root/weights/ditr-nuscenes-config.py" \
  --ditr-weights "$probe_root/weights/ditr-nuscenes.pth" \
  --dino-code "$probe_root/dependencies/dinov3" \
  --dino-weights /path/to/official-dinov3-vitb16.pth \
  --loki-root /run/media/user20/F47C60057C5FC152/projects/loki_data \
  --waymo-root /run/media/user20/F47C60057C5FC152/projects/waymo_v2
```

`--road-index` and `--outputs-root` override frozen-reference storage locations without rewriting manifests. Output directory must be fresh and under `outputs-root/experiments/E002a`. Omitting `--dino-weights` explicitly produces partial LiDAR/crop evidence; there is no alternate RGB checkpoint fallback. Results are local `probe.json`, `report.md`, `index.html`, `panels/`, `fixtures/` and `workloads.json`; failures retain traceback/status and propagate. No resume/retry framework.

## Execution evidence — 2026-10-19

The bounded LiDAR/crop attempt **native-probe-06** completed in 269 s on aalto at revision `f9ff228`, with explicit partial status because RGB weights were unavailable at execution. **75 native development frames**: 38 LOKI / 37 ROAD-Waymo, including separate training-only stress cases. First-frame repeated extraction was **identical (max error 0)** in both datasets. All **663 protected** sampled archives/setup/collection manifests retained their hashes. No behavior training, held-out inference, W&B runs or full caches.

| Measurement | LOKI | ROAD-Waymo |
|---|---:|---:|
| Native points/frame, median |231,936|174,021|
| Spatial tokens/frame, median |105,344|150,063|
| Encoder wall seconds/frame, median |0.196|0.232|
| Peak PyTorch allocated GiB during extraction |0.91|1.07|
| Represented eligible track/slots |474|418|
| Valid native RGB crops |156|120|
| Nonempty-neighborhood tokens, median / maximum |17,453 /66,991|15,434.5 /70,056|

Counts describe the bounded sample, not full-population coverage estimates. Some observed targets have no target-box returns; 29 LOKI /40 ROAD track/slots have zero target-box voxel support, including missing/unusable boxes. Preserve their context/other branches rather than filtering them out.

The inspected LOKI and ROAD panels show recognizable scene structure and plausible object/surface regions, with mixed road/other-flat/sidewalk/terrain predictions. This is a **qualitative observation on selected development images**, not verified segmentation correctness or evidence of behavior recoverability. Native image/crop overlays inspected so far match the declared square crop. Different sensor origin/height remains a domain-shift limitation; do not tune a coordinate translation against semantic appearance alone.

### Resource gate

All 16 conservative whole-track shapes (102 slots,70,056 tokens at every slot, widths128/256 × seeds1/4 × tracks1/2/4/8) reached CUDA OOM. This does **not** establish that every actual track fails. A separate **single-frame synthetic-zero workload at the measured maximum count** passed all 16 configurations with finite backward: one-frame/one-track peak about 0.30 GiB at width 128,0.50 GiB at 256; eight frames about 1.12/2.03 GiB. Its first-call timing includes warm-up and is not steady training throughput. The spatial adapter itself fits; retaining its activations across long tracks is the direct-graph bottleneck. Frame streaming/checkpointing or a declared token reduction need a decision, not silent truncation.

Float32 payload estimates from the random groups, **excluding test preparation, indexes/inverse-map overhead and native I/O**:

| Dataset / split | Full-scene GiB | Neighborhood-union GiB | Duplicated track-neighborhood GiB | RGB CLS GiB |
|---|---:|---:|---:|---:|
|LOKI training|821.78|362.82|1302.26|0.374|
|LOKI validation|85.88|36.49|84.59|0.026|
|ROAD training|1575.75|462.21|1490.07|0.299|
|ROAD validation|562.20|318.03|1313.70|0.161|

Even union sharing totals about **1,180 GiB / 1.15 TiB**; the laptop has about 511 GiB free after extraction environment setup. RGB CLS is approximately 0.86 GiB total and has no comparable storage concern. These small-sample group-scaled estimates can vary substantially; they are sufficient to reject assuming unbounded float32 point caches fit. Point-encoder wall-time extrapolation sums to about 5.9 hours; native I/O, transfer, indexes and caching can add time. Cache layout, budget, numeric precision and spatial token limits remain open; none was applied automatically.

### Artifacts and acceptance

Full evidence is under `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/native-probe-06/` on aalto. [Local report](../../outputs/experiments/E002a/native-probe-06/report.md), [panel gallery](../../outputs/experiments/E002a/native-probe-06/index.html), [numerical/provenance record](../../outputs/experiments/E002a/native-probe-06/probe.json), [whole-track workloads](../../outputs/experiments/E002a/native-probe-06/workloads.json) and `single-frame-workloads/` are ignored outputs. All 27 spatial fixtures, panels and numerical records have been collected locally; generated artifacts stay outside Git. Independent fixture checks reconstruct semantic counts exactly and verify finite float32 data, native selected-coordinate correspondence and pickle-free loading.

Earlier manual attempts are preserved alongside `execution-audit.json`: restricted NumPy checkpoint metadata (01), old sparse runtime NumPy SIGFPE (02/03), evaluation-autocast sparse dtype mismatch (04), and residual pooling-order shuffling (05). Fixed using specific safe scalar allowlisting, spconv-cu126 2.3.8 / cumm-cu126 0.7.11, author-style inference precision and disabling **all** serialization-order shuffling. No automatic retries or checkpoint substitutions.

CPU suite: 87 tests with six CUDA skips; CUDA suite: all 87; isolated extractor probe checks: all 9. The official conversion round-trip preserves all 188 full-backbone state tensors exactly (random initialized weights for this loading check, not pretrained inference). Main lock/dependency and isolated dependency checks pass. All 211 affected-owner links/anchors and `git diff --check` pass. [CPU](../../outputs/experiments/E002a/cpu-acceptance.log), [CUDA](../../outputs/experiments/E002a/cuda-acceptance.log), [isolated runtime](../../outputs/experiments/E002a/extractor-acceptance.log), [fixture acceptance](../../outputs/experiments/E002a/artifact-acceptance.json) and [independent conversion](../../outputs/experiments/E002a/dino-conversion-acceptance.json) retain the checks. DINOv3 access was granted after the point/crop probe. That historical attempt did not execute RGB; the [completion below](#rgb-checkpoint-completion--2026-10-19) records the subsequent original-checkpoint execution.

### RGB checkpoint completion — 2026-10-19

**native-probe-07** completes the fixed-sample probe using the original DINOv3 ViT-B/16 LVD-1689M `.pth`, with strict loading into the pinned original implementation. Matched the published `73cec8be` SHA prefix and pinned full SHA256 `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`; the file hash is unchanged after execution. Execution revision: `738d124edf41812e9ebb42030ecb5a9b857d8ac1`. The earlier partial attempt and failure evidence remain intact.

| Dataset | Native frames | Valid RGB crops / saved descriptors | RGB encoder wall seconds | Co-resident peak GiB | Incremental peak GiB | Repeat max error |
|---|---:|---:|---:|---:|---:|---:|
| LOKI |38|156|0.678|0.629|0.103|0|
| ROAD-Waymo |37|120|0.279|0.664|0.137|0|

All **276 descriptors** are finite float32 vectors of length 768, aligned with the exact native track/frame/crop identities in 60 pickle-free archives. Dataset-specific first-batch repeated extraction matches exactly on this hardware/runtime. Encoder times include input transfer and first-call overhead, exclude native I/O/crop preparation, and are not full-preparation throughput. Peaks are PyTorch allocated memory during RGB extraction with both encoders resident; the incremental column subtracts the pre-call live allocation. Point-extraction peaks in the new report also include resident DINO weights and should not be treated as an increase in point-model cost.

The sample matches **all 75 prior native frames**, with identical selected groups/archives and all 663 protected hashes unchanged. Both native point repeats also remain identical. Float32 cache payload estimates remain unchanged: RGB CLS approximately 0.86 GiB for both datasets' training/validation; point-neighborhood union approximately 1.15 TiB. All 16 conservative full-track readout shapes still reach OOM. Both encoder interfaces are now verified; this does **not** resolve the point-cache/activation budget or establish behavior performance. No training, held-out scoring, W&B run or full-cache preparation.

[Completed report](../../outputs/experiments/E002a/native-probe-07/report.md) · [Panels](../../outputs/experiments/E002a/native-probe-07/index.html) · [RGB acceptance](../../outputs/experiments/E002a/native-probe-07/rgb-acceptance.json) · [Provenance](../../outputs/experiments/E002a/native-probe-07/probe.json). RGB numerical fixtures and all panels are collected locally; the new duplicate spatial fixtures stay on aalto, while the prior 27 verified local spatial fixtures remain available in `native-probe-06`. Full new outputs remain at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/native-probe-07/`.

Validation after RGB measurement: CPU 87 tests (six CUDA skips), CUDA all 87, isolated extractor all 9; dependency/lock checks, owner links and `git diff --check` pass. These checks validate native extraction/correspondence and existing behavior, not annotation quality or cross-platform bitwise identity.

### DINO patch visual check — 2026-10-19

Authorized qualitative follow-up only; no behavior training or segmentation-head download. Reuse the completed probe's rendered development frames, displaying at most two valid crops plus one explicit missing example per frame. Native images/boxes must reproduce the earlier crop records. Preserve the original probe and frozen data; use a fresh output directory.

For each valid example, save exact float32 crop pixels, normalized CLS, normalized **16×16×768 patch tokens**, per-crop three-component PCA colors and cosine similarity to the patch at the centre of the image-visible native box. The seed is a geometric choice, not a verified pedestrian mask; it may contain background or occlusion. Fit PCA separately per crop, fix component signs and scale colors by its 1st/99th percentiles. Colors cannot be compared across crops. Maps are displayed at native patch resolution or enlarged overlays, never claimed as segmentation, pose or behavior predictions. Patch structure does not validate the CLS descriptor used by E002. [Official visualization examples](https://github.com/facebookresearch/dinov3#notebooks).

The original gallery selected the first three represented tracks regardless of crop validity. Three apparent zero-crop ROAD frames actually contain **3, 4 and 5** valid descriptors; `train_00028-1509232266190995` genuinely has zero. Its two native 3D tracks extend approximately 0.6/0.4 s beyond their last FRONT boxes; no ROAD pedestrian GT exists at that timestamp, and one identity has a native FRONT_RIGHT association. ROAD-origin identities do not imply FRONT boxes throughout their complete native context. The new gallery shows valid/represented counts and valid examples first. Prior numerical extraction is unaffected; original evidence remains intact.

From the isolated GPU checkout:

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/check-e002a-rgb.py \
  --probe "$probe_root/native-probe-07" \
  --output "$probe_root/dino-visual-check-new" --device cuda
```

The command reuses pinned checkpoint/native paths from `probe.json`; it does not decode LiDAR. Match the original crop order and batches of up to16 for FP16 comparison. Raw saved crop pixels retain bicubic overshoot; display clips to [0,1] without changing encoder inputs.

**Completed `dino-visual-check-03`** at revision `cff408200c9f87b0f05be1d737ac7d741fbfcb89` in **38.3 s** on aalto: **27 development frames, 45 valid patch examples and 20 explicit missing-crop examples**. All selected CLS descriptors match the previous extraction exactly (max absolute error0); the first repeated patch/CLS batch also matches exactly. All **725 protected files** remain unchanged: the prior 663 frozen references, probe record, original checkpoint and 60 RGB descriptor archives. [Gallery](../../outputs/experiments/E002a/dino-visual-check-03/index.html) · [numerical/provenance record](../../outputs/experiments/E002a/dino-visual-check-03/visual-check.json) · [independent fixture/link checks](../../outputs/experiments/E002a/dino-visual-check-03/acceptance.json). Panels have PNG/SVG exports; pickle-free fixtures retain exact inputs/features/maps/identities. Outputs are available locally and at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/dino-visual-check-03/`.

Purposive visual inspection shows distinguishable visual regions in some examples and broad context similarity in others. These views support extraction sanity only; they do not establish foreground segmentation, pose sensitivity, CLS behavior information or modality benefit. No segmentation head, behavior training, target-test access or W&B run.

The original [LiDAR-first gallery](../../outputs/experiments/E002a/native-probe-06/index.html) and [completed point/RGB gallery](../../outputs/experiments/E002a/native-probe-07/index.html) now also show the corrected crops alongside **visible original BEV panels**, with counts per frame and missing-crop context collapsed and labeled. Each keeps `index-before-crop-fix.html` as presentation evidence. Only HTML was refreshed; native/probe numerical files and encoder predictions are unchanged. The same presentation helper serves the DINO gallery; no inference rerun is needed.

### Crop-size visual comparison — 2026-10-19

Authorized diagnostic: compare centered square sides **1.5, 2 and 3 × max(native bbox width, height)** on the same 45 valid development examples. Every input remains 256×256, with unchanged bicubic sampling, mean padding, native boxes and no crop shifting. Larger crops add context while reducing target pixel size; no training setting or baseline cache is changed. Missing boxes stay missing. Show crops side by side, with padding fractions, and keep the three-scale DINO PCA/similarity comparison in an expandable panel. PCA colors are independently fitted per crop/scale, not matched semantic classes.

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/check-e002a-rgb.py \
  --probe "$probe_root/native-probe-07" \
  --output "$probe_root/dino-crop-scales" --device cuda --compare-crop-scales
```

The default command still uses only 1.5×. The optional comparison preserves the original 1.5× batch composition/checks and saves new 2×/3× float32 crops, CLS, patch features, maps and scale-specific crop/seed metadata in separate pickle-free fixtures. Original numerical evidence stays unchanged. Execution/acceptance results pending.

Retained earlier manual attempts: `dino-visual-check` failed the CLS comparison because single-crop FP16 inference changed batch composition; `dino-visual-check-02` reproduced all LOKI examples but failed on a serialized native-root string passed to the ROAD reader. Matching original batches and restoring a `Path` fixed those failures; regression checks cover both. No automatic retry or overwritten evidence. Full CPU **91 tests / six CUDA skips**, CUDA **91**, and final isolated extractor **13** checks pass; numerical fixtures reconstruct cosine maps and all four questioned ROAD counts, and documentation links/`git diff --check` pass. Logs: `outputs/experiments/E002a/dino-visual-{cpu,cuda,extractor}-acceptance.log`.
