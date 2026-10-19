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
- Deterministic first native return per 5 cm voxel, author FNV ordering, exact inverse mapping; fixed serialization orders. These declared changes replace random validation sampling/order. No segmentation test-time augmentation or silent point truncation. Frozen extraction uses CUDA FP16 autocast and saves float32 descriptors; test repeatability separately from deterministic FP32 trainable operations.

Point neighborhoods: **15 m horizontal radius in world-horizontal/fixed initial-heading axes**, no height cutoff. Tokens carry `[feature64, relativeXYZ3, inside_native_cuboid1]`. Native heading/full dimensions are used only for membership. A cuboid membership flag is geometric containment, not verified point ownership. No interpolation across missing native boxes/poses.

Pinned author revisions and hashes are enforced in [the adapter](../../src/pedestrian_behavior/inspection/pretrained.py). Weight: `nuscenes/semseg-ptv3_weight=distill-ptv3_nuscenes_dino-L/model/model_best.pth` at Hugging Face revision `a656b4385aa28060b30152676023903a923b4a17`; SHA256 `93dbb9d3fda247adf56e1c1ebd419d6ada4ebe4dc141110c606926d3ad67b30d`. Matching config SHA256 `7cce1838385d3f14830b5d61c88db98cbb62951cea39665311c61bca23f253e3`. [Pretraining/exposure qualifications](../../docs/literature/papers/ditr.md) remain applicable.

## Visual and numerical checks

Save raw-height BEV, semantic BEV and an oblique height-preserving point view with native target boxes and neighborhood circles. These are qualitative predictions from the existing nuScenes segmentation head, **not behavior outputs or measured segmentation accuracy**. Preserve a legend. Save full-image native-box/crop overlays beside the exact model crop, including naturally missing/partial observations. Inspect alignment, sparse/distant targets and gross semantic domain failures before interpreting features.

[CPU acceptance](../../tests/test_feature_probe.py): boundary/outside/missing crops, mean padding, rotated cuboids, fixed-heading transforms, point/strength mask correspondence, voxel inverse mapping, reordered feature-coordinate pairs, empty neighborhoods, finite gradients and development-only repeatable sampling. Native execution additionally checks finite outputs, 768/64 shapes, strict checkpoint loading and repeatability. Existing command behavior is covered by the full repository suite. Frozen extraction may use nondeterministic sparse kernels: recorded numerical repeatability is a measurement, not a cross-platform identity guarantee.

## Resource workload and estimates

Temporary benchmark graph only; no optimizer updates or trained behavior labeler. Shared unconditioned learned seeds, one spatial cross-attention block/four heads/FFN; internal widths 128/256 × one/four seeds; output256. Geometry64 + RGB256 + LiDAR256 + three flags =579 →256. Two bidirectional pre-norm temporal layers, width256/four heads/FFN1024/dropout0.1, sinusoidal elapsed grid time, four-class readout. Profile FP32 forward/backward for whole-track microbatch shapes 1/2/4/8. Repeat the largest observed neighborhood at every slot of the longest eligible training track: this conservative shape test is **not measured training throughput or actual complete-track extraction**. Record OOMs without retry/truncation.

Estimate training/validation float32 payload and point-GPU extraction time per dataset using the random sample, scaling frame means by eligible scene-slot counts then averaging sampled group totals. Compare whole-scene sharing, per-frame neighborhood union sharing and duplicated pedestrian neighborhoods. Report RGB separately. Estimates exclude test preparation, index/inverse-map overhead, compression and native I/O; warm/cold kernel effects and small sample uncertainty limit precision. Cache representation, spatial width, any token limit and disk ceiling remain undecided until this report is reviewed.

## Environment and command

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

## Execution evidence

Pending native execution. D-DITR weights and compatible CUDA extensions are available on aalto. DINOv3 official weights requested from user because model access is gated. Do not infer compatibility or useful segmentation from successful installation.
