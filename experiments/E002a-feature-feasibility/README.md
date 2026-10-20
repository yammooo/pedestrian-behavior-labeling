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

The default command still uses only 1.5×. The optional comparison preserves the original 1.5× batch composition/checks and saves new 2×/3× float32 crops, CLS, patch features, maps and scale-specific crop/seed metadata in separate pickle-free fixtures.

Completed at revision `792c9050653b74305758d9bb3853a9f96ece1b50` in **97.7 s** on aalto: **45 pedestrians × three crop sizes =135 feature fixtures** across the same 27 development frames. The repeated baseline patch batch matches exactly; all 725 protected files are unchanged. [Comparison gallery](../../outputs/experiments/E002a/dino-crop-scales/index.html) · [record](../../outputs/experiments/E002a/dino-crop-scales/visual-check.json) · [independent acceptance](../../outputs/experiments/E002a/dino-crop-scales/acceptance.json). PNG/SVG comparisons and pickle-free arrays are available locally and under the same ignored output path on aalto. CPU93 (six CUDA skips), CUDA93 and isolated extractor15 checks pass. Larger crops visibly trade target detail for scene context; this does not identify the best behavior-classification input. No training or held-out access.

The original native06/native07 and DINO gallery pages also link these side-by-side crop panels and expandable feature comparisons. Their numerical/provenance records retain the original 1.5× evidence; the 2×/3× data live only in `dino-crop-scales/`. The E002 working crop choice remains 1.5× pending a behavior-based comparison.

### Whole-track window videos — 2026-10-19

Authorized qualitative extension: **ten complete pedestrian tracks, five per dataset**, drawn only from the existing probe's development groups. Prefer tracks with the most paired native 2D/3D slots and distinct scenes, breaking ties by paired coverage and stable identity. This is purposive observation coverage, not a representative sample or behavior-based selection. Preserve every frozen 5 Hz slot through each full native track extent, including unlabeled/missing context; no interpolation, trimming or last-frame reuse.

Left: one centered **3× max-box-side RGB window**, native target box in white, thin 1.5×/2×/3× crop outlines in cyan/yellow/magenta. Display samples to640² for readability; model crops/defaults remain256²/1.5×. Right: the pinned frozen nuScenes D-DITR semantic head on each full native scan, then pedestrian-centered BEV of decoded 5 cm voxel points within **30 m horizontal radius**, no height cutoff. Draw **15/20/30 m radius circles**, the native target cuboid footprint and class legend. Use the same fixed initial ego-heading/world-horizontal coordinates as E002a; up is initial forward, screen-right is initial right. Scene inference uses the original axis/strength/pose contracts. The views are synchronized by saved native references and labeled grid/native times. Missing RGB or usable 3D observations produce explicit placeholders independently.

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/video-e002a-tracks.py \
  --probe "$probe_root/native-probe-07" \
  --output "$probe_root/track-window-videos" --device cuda
```

Outputs: ten H.264 MP4s at 1440×900/5fps, first/middle/last PNGs, `index.html`, per-video encoder logs and `videos.json` with track selection, per-frame identities/times/crop bounds/support, checkpoint/revision references and failure evidence. Geometry tests use known nested squares and metric offsets/radius boundaries; missing observations never inherit another timestep. No DINO feature extraction, training, held-out access or segmentation accuracy claim.

### Frozen point-feature PCA — 2026-10-20

Authorized comparison of **frozen nuScenes D-DITR versus original Utonia** on the same ten development tracks above. Qualitative feature inspection only; no behavior training, segmentation GT scoring, held-out access, new full cache or backbone selection. [Command](../../README.md#e002a-pretrained-feature-feasibility).

- Utonia inference source `de1f9deef4ee65e508d74347f8d0eb848d878abb`; original `utonia.pth` SHA256 `a0a2e5e234ce943082ae9ba26bcff74e285f16ed8d04fed5c36ffe8dc30ea353`, verified against official HF LFS metadata. Frozen strict author loading, all serialization shuffling disabled. Official [outdoor demo](https://github.com/Pointcept/Utonia/blob/de1f9deef4ee65e508d74347f8d0eb848d878abb/demo/7_pca_outdoor.py) recipe: 0.2 coordinate scale, 0.01 scaled grid (5 cm physical), ego origin retained, no scene centering, absent color/normals zero. The [actual NormalizeColor](https://github.com/Pointcept/Utonia/blob/de1f9deef4ee65e508d74347f8d0eb848d878abb/utonia/transform.py) divides RGB by 255; it does not subtract one. No LiDAR RGB coloring or normal estimation. D-DITR's existing XYZ/strength recipe is unchanged.
- Utonia colors use its official outdoor visualization's last three hierarchy levels (216+432+576=1224 channels), inverse-mapped to native rows. D-DITR uses its 64 decoded channels before the semantic head. These feature levels, supervision and input channels differ; this is a practical released-representation comparison, not an isolated architecture/pretraining ablation. Different grid axes/rounding are permitted by author recipes; both displays use the exact same native XYZ indices through their respective inverse maps.
- One PCA per encoder, fitted once on **16,384 rows**: four existing probe training scenes per dataset, first two selected probe frames per scene, 1024 deterministic native returns/frame within 30 m world-horizontal distance of an eligible track. No behavior label or error-based selection. Identical sampled native indices for both encoders. Three leading components, deterministic signs, fixed training-sample 1st/99th percentile channel bounds; no channel whitening or per-frame scaling. Colors do not correspond between encoders. Pooled calibration uses both datasets and cannot support strict zero-shot evaluation.
- Full-scene encoding, then the same target-centered 30 m world-horizontal neighborhood for both panels; initial-ego-heading BEV, no height cut, 15/20/30 m rings, native target footprint. Display maps features to all native returns; it does not recover detail lost in grid/hierarchical reduction. Reuse the complete frozen 5 Hz grids and RGB windows, with independent missing observations and no temporal interpolation/accumulation. Utonia's published pretraining includes Waymo training/validation; [exposure qualifications](../../docs/literature/papers/utonia.md#verified-pretraining-partitions-2026-10-15) remain applicable.
- Fresh output: `index.html`, ten 2048×850/5fps MP4s, first/middle/last PNGs, `comparison.json`, pickle-free `pca.npz`, calibration features/native indices and per-frame point offsets/PCA colors. Original probe/video records, checkpoints, setup and frozen source evidence are checksum-protected. Record revisions, settings, dependency/backend versions, hardware, repeat errors, encoder wall time/peak allocated CUDA memory and failures. Errors propagate with local evidence retained; no automatic retry.

Acceptance: known PCA axes/variance and fixed out-of-range clipping; training-only balanced calibration; gather equivalence to the official full hierarchy upcast; missing-modality values and coordinate scaling; identical rendered point positions, metric rings and missing panels; failure propagation/local preservation. Real execution checks finite/repeated features on both datasets, exact original grid identities/times, all 865 video frames, pickle-free artifact correspondence, and unchanged protected sources; evidence follows below.

Attempt history is retained: `point-pca-comparison` failed on an unused SciPy augmentation import before extraction; the adapter now imports only the pinned inference modules. `point-pca-comparison-02` was manually stopped and marked **invalid-preprocessing** after discovering incorrect -1 missing-color inputs. Its Utonia colors/variance are excluded. The actual pinned author normalization/scaling class bodies were then executed independently ([acceptance](../../outputs/experiments/E002a/point-pca-author-preprocessing-acceptance.json)); missing RGB is zero, white RGB is one, outdoor XYZ scale is 0.2. Only the corrected fresh `point-pca-comparison-03` is eligible for review. No automatic retries, existing result edits or native reprocessing.

**Corrected execution completed** at revision `18ce056ad6e4303015e86340bf4336bd5b0536f4` in **962.1 s** on the RTX 4080. Ten videos / **865 frames**: LOKI369 (316 valid RGB crops), ROAD-Waymo496 (382 valid RGB crops). Both encoders' repeated calibration features match exactly on each dataset (max absolute error0). All **721 protected files** remain unchanged. PCA bases/bounds reconstructed exactly from the saved 16,384 calibration feature rows per encoder. The three displayed components explain **60.12% D-DITR / 22.14% Utonia** variance; unequal projection coverage and differing feature stages preclude ranking encoder quality from these colors.

| Dataset | D-DITR median extraction s/frame | Utonia median extraction s/frame | D-DITR peak allocated GiB | Utonia peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| LOKI | 0.190 | 0.496 | 1.37 | 3.65 |
| ROAD-Waymo | 0.225 | 0.518 | 1.47 | 3.06 |

Times include grid preparation, inference, gathering/projection and CPU transfer; exclude native I/O and artifact encoding. Peaks include both resident encoders. Author precision: FP32 sparse/linear operations, internal FlashAttention FP16 for D-DITR / BF16 for Utonia, no outer autocast. These are practical probe costs, not an isolated architecture benchmark. Utonia's global PCA color distribution visibly differs across these two datasets; neither that observation nor a sharper-looking D-DITR map establishes behavior transfer quality. No backbone or E002 training setting was changed.

[Corrected gallery / ten MP4s](../../outputs/experiments/E002a/point-pca-comparison-03/index.html) · [Execution/provenance](../../outputs/experiments/E002a/point-pca-comparison-03/comparison.json) · [PCA reconstruction](../../outputs/experiments/E002a/point-pca-comparison-03/pca-reconstruction-acceptance.json) · [Artifact acceptance](../../outputs/experiments/E002a/point-pca-comparison-03/acceptance.json). Copies live locally and on `aalto` under the isolated review checkout's `outputs/experiments/E002a/point-pca-comparison-03/`. Final CPU102 (six CUDA skips), CUDA102 and isolated22 checks pass; [CPU](../../outputs/experiments/E002a/point-pca-cpu-acceptance.log), [CUDA](../../outputs/experiments/E002a/point-pca-cuda-acceptance.log), [isolated](../../outputs/experiments/E002a/point-pca-extractor-acceptance.log). Original E001-family archives/results and existing visual evidence are retained unchanged.

Independent local acceptance verifies all865 encoded frame counts/rates, exact original native keys/times/crop validity, **863 aligned finite float32 point/color fixtures**, 30 PNGs, and all16 calibration fixture hashes. Two ROAD-Waymo timesteps lack usable target 3D context; their missing panels are retained independently of RGB. [LOKI contact sheet](../../outputs/experiments/E002a/point-pca-comparison-03/loki-contact.png) · [ROAD-Waymo contact sheet](../../outputs/experiments/E002a/point-pca-comparison-03/road-waymo-contact.png).

Completed at revision `ee3a32557fc181d2b7181366f68d9075593a3698` in **365.1 s** on aalto: **ten videos / 865 frames / 173 s playback**, with five distinct scenes per dataset. Per-frame native key/timestamp lists exactly match each complete frozen track; `ffprobe` verifies encoded frame counts, dimensions and 5fps. All **666 protected files** are unchanged. CPU96 (six CUDA skips), CUDA96 and isolated extractor16 checks pass; native/encoded-frame and gallery checks are retained with the outputs. Spot-checked middle frames on both datasets show readable nested windows/rings and the linked target boxes. Semantic predictions remain qualitative, including the existing sensor/domain limitations.

[Video gallery](../../outputs/experiments/E002a/track-window-videos/index.html) · [per-frame record](../../outputs/experiments/E002a/track-window-videos/videos.json) · [native-reference check](../../outputs/experiments/E002a/track-window-videos/native-reference-acceptance.json) · [video/link acceptance](../../outputs/experiments/E002a/track-window-videos/acceptance.json). Available locally and at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/track-window-videos/`; logs live beside that folder as `track-video-{cpu,cuda,extractor}-acceptance.log` (CPU log local).

Retained earlier manual attempts: `dino-visual-check` failed the CLS comparison because single-crop FP16 inference changed batch composition; `dino-visual-check-02` reproduced all LOKI examples but failed on a serialized native-root string passed to the ROAD reader. Matching original batches and restoring a `Path` fixed those failures; regression checks cover both. No automatic retry or overwritten evidence. Full CPU **91 tests / six CUDA skips**, CUDA **91**, and final isolated extractor **13** checks pass; numerical fixtures reconstruct cosine maps and all four questioned ROAD counts, and documentation links/`git diff --check` pass. Logs: `outputs/experiments/E002a/dino-visual-{cpu,cuda,extractor}-acceptance.log`.

### Frozen semantic-head transfer check — 2026-10-20

Authorized bounded follow-up: evaluate the existing frozen nuScenes D-DITR **backbone and segmentation head**, without fitting either. Native `waymo_v2/{training,validation}/lidar_segmentation/` contains 798/202 Parquet files; schema and sampled nonzero values verified. Exactly **seven of the existing 37 Waymo development frames, from five clips**, match semantic-label timestamps. Use all seven, selected before inference; no new scene search, held-out access, behavior training or full-cache preparation.

The [official schema](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/segmentation.py) stores `[instance_id, semantic_class]`, TOP only, with sparse annotated timestamps/points and both returns. Keep the existing full multi-sensor scan as encoder input. Align labels through the same laser/return order and positive-finite range mask as native decoding; score only TOP returns with supported GT, at all ranges. Preserve native pixel indices, inverse voxel association and raw classes. Check the native strength sequence independently against concatenated range images. No interpolation or voxel-majority replacement of native GT.

**Taxonomy frozen before inference:** nine coarse shared classes: vehicle (Waymo car/truck/bus/other_vehicle; nuScenes car/truck/bus/construction_vehicle/trailer), pedestrian, bicycle, motorcycle, cone, drivable (Waymo road+lane_marker), sidewalk, manmade (Waymo buildings/signs/lights/poles) and vegetation (including Waymo trunks). Native IDs follow [Waymo's enum](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/segmentation.proto) and the [pinned nuScenes config](https://huggingface.co/rwthvision/ditr/blob/a656b4385aa28060b30152676023903a923b4a17/nuscenes/semseg-ptv3_weight=distill-ptv3_nuscenes_dino-L/config.py). Exclude Waymo undefined, riders, curb, other_ground and walkable, whose categories do not map cleanly. Predict native 16-class argmax first, then map; nuScenes barrier/other_flat/terrain become an explicit unmatched prediction column and remain errors on scored GT. Excluded GT contributes no false positives. Category boundaries remain approximate; this is not official Waymo mIoU.

Report pooled native-return confusion, per-class IoU/precision/recall/F1 and point/frame/group support, accuracy, supported-GT-class macro-IoU, per-frame scores and excluded/unlabeled counts. Absent classes remain explicit; no fabricated uncertainty intervals from seven correlated frames. Save raw/row-normalized PNG/SVG confusion, seven GT/prediction/error BEVs, pickle-free reconstruction fixtures, mapping/provenance/protected hashes/status/failures, `report.md` and `index.html` in a fresh ignored attempt. BEV bounds are display-only. Existing 5 cm deterministic sampling/axis/intensity/precision contracts remain unchanged; no TTA or new classifier. [Acceptance tests](../../tests/test_semantic_check.py) cover range masks/channels, sensor/return ordering, unmatched-prediction penalties, hand-computed metrics, empty support and aligned BEV pixels. [Command](../../README.md#e002a-pretrained-feature-feasibility).

This diagnoses the released head on a small Waymo development sample. It cannot establish behavior usefulness, frozen-feature linear separability, or LOKI transfer. No model winner or later experiment setting is selected from these scores.

**Completed `semantic-transfer-02`** at revision `85d51f0cce48438221a56104fd022d2510ac12e8` in **71.3 s** on aalto (summed encoder calls **2.47 s**, remainder native reads/checks/artifacts). Seven frames from five clips: six source-training frames and one source-validation frame. **994,982 scored native returns**, **78.30% accuracy**, **40.40% supported-class macro-IoU** (8/9 classes supported). These are mapped development diagnostics, not official benchmark or behavior results.

| Class | GT points | Supporting frames / clips | IoU % | Precision % | Recall % |
|---|---:|---:|---:|---:|---:|
| Vehicle |109,345|7 /5|61.73|89.67|66.45|
| Pedestrian |29,654|7 /5|22.05|97.29|22.19|
| Drivable |179,183|7 /5|73.22|88.97|80.53|
| Sidewalk |62,629|7 /5|15.20|57.46|17.12|
| Manmade |351,752|7 /5|76.58|83.51|90.22|
| Vegetation |261,951|7 /5|74.46|83.93|86.83|
| Bicycle |359|1 /1|0.00|0.00|0.00|
| Cone |109|2 /2|0.00|0.00|0.00|
| Motorcycle |0|0 /0|0.00|0.00|unknown|

The unsupported motorcycle class has 121 false-positive predictions; its recall is undefined and it is excluded from supported-class macro-IoU. Rare bicycle/cone scores have little support. Pedestrian recall ranges from 0–42.9% across these frames; most missed pedestrian points map to manmade (13,503) or vegetation (7,882). Sidewalk often maps to unmatched native categories (38,273 points). Broad scene structure transfers on this small sample, but the existing head is not a reliable pedestrian/sidewalk segmenter here. This does not imply the frozen features cannot support a learned behavior readout; it also supplies no evidence for LOKI semantic performance. No preprocessing or class mapping was tuned after scoring.

[Full table/report](../../outputs/experiments/E002a/semantic-transfer-02/report.md) · [GT/prediction/error gallery](../../outputs/experiments/E002a/semantic-transfer-02/index.html) · [Numerical/provenance record](../../outputs/experiments/E002a/semantic-transfer-02/semantic-check.json) · [Independent acceptance](../../outputs/experiments/E002a/semantic-transfer-02/acceptance.json) · [Confusion PNG](../../outputs/experiments/E002a/semantic-transfer-02/confusion.png) / [SVG](../../outputs/experiments/E002a/semantic-transfer-02/confusion.svg). Available locally and at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/semantic-transfer-02/`.

All **671 protected files** retained their hashes. Independent acceptance checks every **1,165,732 native TOP label/pixel association**, exact metric reconstruction from pickle-free arrays, and identical coordinates/voxel mappings/predictions for all four frames with retained original spatial fixtures (the other three have no prior spatial fixture). Seven BEV panels and raw/row-normalized confusion exports were checked. CPU107 (six CUDA skips), CUDA107 and isolated semantic5 checks pass; [CPU](../../outputs/experiments/E002a/semantic-cpu-tests.log), [CUDA](../../outputs/experiments/E002a/semantic-cuda-tests.log). The first `semantic-transfer` attempt failed before inference because a validation check wrongly rejected legal -1 **instance IDs**; semantic IDs remained nonnegative. The fix validates the semantic channel separately and is regression-tested; the failed evidence is retained. No automatic retry or existing result changes.

### Frozen-feature linear semantic probe — 2026-10-20

Authorized after the existing head's low pedestrian/sidewalk recall. Train **only a fresh affine `Linear(64,9)`** over the unchanged frozen/eval D-DITR representation. This is a supervised Waymo semantic diagnostic, not behavior-model training or strict zero-shot segmentation; no LOKI supervision, held-out access, W&B or full-population preparation. Earlier inspected development images/scores informed this probe; validation is exploratory, not an independent final estimate.

**Frozen before training:** retain the original probe's eight random training/two random validation clips, excluding extra stress-only groups. All ten have 29–40 semantic timestamps. Select six timestamps/clip, evenly spaced through sorted available native semantic timestamps (`linspace` integer positions): **48 training /12 validation frames**, no clip overlap. These are native semantic timestamps, not a new behavior-grid cache. Training: up to8192 uniformly sampled GT-supported native TOP returns/frame, deterministic NumPy seed0, fixed sample across epochs, no class balancing. Validation: every GT-supported TOP return at all ranges; no sampling. Preserve both returns, full scene encoder input and prior 5 cm/axis/intensity contracts. Same nine-class mapping/exclusions as the unadapted check.

Normalize the sampled **training** features by per-channel mean/std only (std floor1e-6); no validation fitting. New head585 parameters, FP32, AdamW LR0.001, weight decay0.0001, default betas(.9,.999)/epsilon1e-8, unweighted point CE, batch8192 with final partial batch retained, clip1. Seed0 and deterministic algorithms for head fitting; encoder extraction retains the documented sparse-kernel precision/repeatability qualifications. Budget50 epochs, patience10 on strict validation supported-class macro-IoU improvements; exact score ties prefer lower validation CE, then earlier epoch. No scheduler, hyperparameter sweep, backbone update, automatic retry or resume. Save atomic head-only `best.pt`/`last.pt` using shared checkpoint operations and reload the selected head.

Compare on identical validation points: (1) original 16-class argmax mapped with unmatched-prediction errors; (2) **taxonomy-only control**, summing native class probabilities per common class and choosing among the nine common classes without fitting; (3) new learned nine-class head. The control prevents attributing gains from discarding terrain/other_flat/barrier or merging vehicles to head learning. This remains head adaptation with Waymo labels, not a matched pretraining/method benchmark.

Save source references, class support, selected identities/pixels, compact train/full-validation float32 feature fixtures, original/control/new predictions, before/after backbone-state hashes, head checkpoints, epoch history, per-frame/pooled metrics, GT/prediction/error panels, confusion PNG/SVG and CE/IoU curves in a fresh ignored directory. No changes to E001-family data/results or previous probes. [Command](../../README.md#e002a-pretrained-feature-feasibility); [tests](../../tests/test_semantic_probe.py) check group separation/even sampling, taxonomy-only probability merging, source-only normalization, final partial batches and checkpoint reload/prediction equality. Native acceptance must reconstruct scores, verify cached labels against native pixels, confirm unchanged backbone/protected files and disclose unsupported/rare classes. Results are unknown until execution; success supports accessible common semantics, not behavior utility or LOKI transfer.
