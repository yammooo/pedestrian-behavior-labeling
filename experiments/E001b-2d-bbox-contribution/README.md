# E001b — 2D bounding-box contribution

Diagnostic extension of [E001](../E001-kinematic-transfer/README.md), agreed 2026-10-08 while the user runs E001. **Native Waymo FRONT geometry only, without ROAD fallback, accepted on 2026-10-09** after visual review. The original ROAD-preferred prototype remains historical evidence. The shared runner and five-seed launcher are implemented; full comparison runs have not started. Preparation is additive: the frozen E001 collections, setup, populations, splits and kinematics remain unchanged.

## Question and frozen comparison

Does annotated 2D bbox geometry add information for offline four-state labeling beyond pedestrian kinematics and ego interaction? Distinguish annotation availability from geometric contribution; the principal comparison is **geometry versus availability**.

| Configuration | Inputs | Dimension |
|---|---|---:|
| Baseline | Exact E001 **K+T+R** | 12 |
| Availability | Baseline + `bbox_valid`, `bbox_velocity_valid` | 14 |
| Geometry | Baseline + six normalized numeric bbox features + the same two flags | 20 |

The baseline includes `v_ped`, displacement from the first valid position, relative position and **relative velocity** (`v_ped − v_ego`), with the existing four vector-validity flags. Do not replace relative velocity with ego velocity or rename this input K+T+I. The user narrowed E001b to **BiLSTM only** on 2026-10-09: two new inputs × LOKI/ROAD-Waymo × seeds 0–4 require **20 additional training runs / 40 evaluation cells**. Reuse the matching E001 K+T+R BiLSTM seed runs at the accepted 75-epoch/8-patience budget if all shared settings match. Use descriptive configuration names rather than A/B/C, since E001 already uses A/B for model architectures.

Reuse E001's [models](../E001-kinematic-transfer/README.md#first-diagnostic-baseline), complete tracks and 5 Hz grid, accepted ontology/overrides, seed, source-only normalization, loss, AdamW, deterministic settings, training budget, checkpoint selection and [evaluation](../E001-kinematic-transfer/README.md#evaluation). Only input dimension changes. All eligible tracks remain, including 3D-only LOKI pedestrians. Test metrics cannot select settings/checkpoints; prior E001/native inspection is disclosed. No RGB encoder, LiDAR encoder, calibration reconstruction, extra dataset or native-data reprocessing.

## Additional observation contract

Given image-relative corners `(x_min/W, y_min/H, x_max/W, y_max/H)`, save columns in this order: horizontal center, bottom coordinate, width, height, horizontal center velocity and bottom-coordinate velocity. Speeds use the existing E001 immediate-neighbor derivative and selected source timestamps. LOKI uses **nominal seconds**, because physical timestamps are unknown; ROAD-Waymo uses the frozen frame timestamps, not reconstructed exposure times.

LOKI boxes use native `left, top, width, height` and dimensions from each selected PNG header. ROAD-Waymo uses **only native FRONT `CameraBoxComponent`** center/size geometry divided by FRONT calibration width/height, independently of ROAD behavior GT and 3D availability. A missing or invalid Waymo box stays missing: **no ROAD fallback**, and its behavior GT remains accepted. Unlabeled context can retain boxes. Do not use projected LiDAR boxes. Preserve native extents without clipping; count boxes outside image bounds. Record how many same-ID/time ROAD/native pairs are available and their maximum coordinate disagreement. Existing [ROAD association qualifications](../E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08) remain applicable.

`bbox_valid` requires finite coordinates and positive width/height. Nonfinite/degenerate boxes are masked and reported; an invalid native box is not replaced by a ROAD box. ROAD presence remains provenance, not a geometry input. Preserve `loki_2d`, `road_2d` and `waymo_2d` annotation-presence flags separately: presence alone is not verified numeric usability, visibility, occlusion or a behavior target.

`bbox_velocity_valid` requires a valid box at the current slot and an immediate valid neighbor from the **same annotation source**. Never bridge missing slots or source changes; never interpolate. Two neighbors use E001's unequal-time weighting, one neighbor uses its secant and a singleton has no velocity. Both new configurations receive the same short temporal preprocessing and validity flags.

Standardize each continuous column using valid **source-training context**, including unlabeled context, only. Reuse the existing population-standard-deviation/guard rule; missing numbers become zero after standardization. Do not standardize flags. B and C receive exactly the same flags. Source baseline and bbox statistics must both be supplied explicitly for opposite-dataset evaluation. Image-size normalization does not compensate for focal length or camera mounting differences; intrinsics are not required.

## Extension artifacts and data flow

Implementation: [native bbox readers](../../src/pedestrian_behavior/datasets/bbox.py) → [geometry and validation](../../src/pedestrian_behavior/data/bbox.py) → [E001b preparation and sample assembly](../../src/pedestrian_behavior/experiments/e001b.py). [Commands](../../README.md#e001b-bbox-extension) own execution.

Current ignored output: `outputs/experiments/E001b/bbox-native/<dataset>/`; the original ROAD-preferred `bbox/<dataset>/` stays intact as evidence and is rejected by the current loader's policy check. One pickle-free `tracks/<E001 archive name>.npz` for every eligible E001 track; no copied kinematic arrays. Each archive contains float64 `bbox_features[T,6]`, boolean usability/presence arrays, uint8 source codes, int64 `image_size[T,2]`, and scalar JSON metadata. Metadata records locator, exact selected frame keys/times, original track checksum and invalid-box issues. Missing numbers remain NaN in physical caches.

`manifest.json` records preparation status/failures, policy, native/source and original collection/setup checksums, per-track bbox/original checksums, source provenance, code/environment, times, source-training statistics and split-level coverage. Creation requires a fresh output directory and retains failures. Original E001 manifest/audit/setup and **all candidate archives**, including excluded candidates, are checksummed before/after preparation. Loading rejects stale, corrupted, incomplete or misaligned extensions and any population change.

`dataset_from_extension()` retains the existing sample/collation interface and appends 2 or 8 float32 columns to exact K+T+R inputs. Runtime uses saved caches; it does not access native data or fit statistics. E001's existing feature/loading/model/training semantics are unchanged; its attempt and scheduler operations are now shared with E001b. The preparation command still only prepares evidence. [Training commands](../../README.md#e001b-training-and-evaluation) use verified saved caches.

## Execution gate and acceptance

Before training, verify coordinates, dimensions, timestamps and same-frame native IDs; quantify full/valid/missing/velocity-valid support and 2D-without-3D context on frozen populations. Report supporting groups/tracks/context frames/accepted GT/class frames and class groups/tracks, plus never/partial/complete usable-box track cohorts. Coverage is observation evidence, not held-out predictive performance. No arbitrary coverage threshold is introduced; inspect documented limitations before authorizing training. Freeze additional settings before E001b test scores.

[Five acceptance scenarios](../../tests/test_e001b.py) cover hand-calculated image geometry/unequal-time derivatives; gaps/singletons/source changes/invalid dimensions; identical masks and missing-zero/source normalization; both native formats through cache reload and 14/20-column batches with unchanged kinematics/targets and checksum/corruption failures; and actual E001 setup with held-out-value independence and retained native-mismatch failure evidence. At implementation revision `858519e222572092bbfb7d77bfe80d90481e9748`, the full suite ran **43 tests**: CPU **41 passed / two CUDA skips**, remote CUDA **43 passed**. Logs remain under ignored `outputs/experiments/E001b/acceptance-{cpu,cuda}.log`. Dependencies are unchanged; no extra dependency was introduced.

### Real preparation and numerical verification (2026-10-08)

The original ROAD-preferred prototype caches are complete locally and on `aalto`, under `outputs/experiments/E001b/bbox/{loki,road-waymo}/`. The remote commands ran in `/home/user20/projects/pedestrian-behavior-labeling-e001b`, an isolated Git worktree at the implementation revision above with its own uv environment, writing only new E001b outputs in the original checkout. Low CPU/I/O priority was used; preparation allocated no CUDA tensors. E001 code/environment, saved kinematics, labels, populations and splits were not changed by this work. Before/after hashes matched for **13,368 LOKI / 9,576 ROAD-Waymo original collection/setup files**, including excluded candidates.

| Frozen eligible population | LOKI | ROAD-Waymo |
|---|---:|---:|
| Tracks | 12,364 | 6,608 |
| Context slots | 403,813 | 507,144 |
| Accepted GT frames | 391,569 | 254,156 |
| Numerically usable bbox slots | 183,815 | 257,573 |
| Accepted GT frames with usable bbox | 174,266 | 254,156 |
| Usable bbox-velocity slots | 182,202 | 256,976 |
| Tracks with no usable bbox | 4,139 | 0 |
| Partial / complete usable-box tracks | 7,471 / 754 | 5,967 / 641 |
| Bbox slots without usable 3D position | 9,549 | 41,670 |

Zero nonfinite/degenerate selected boxes were found. LOKI selected PNG headers all report **1920 × 1208**; ROAD FRONT calibration dimensions all report **1920 × 1280**, independently matched to RGB headers in three reviewed clips. LOKI has no selected out-of-bounds boxes; ROAD-Waymo has **12**, retained without clipping. ROAD supplies **254,263** selected boxes and native FRONT fallback supplies **3,310**. Preparation took **76.8 s locally / 71.0 s remotely for LOKI**, and **48.7 s remotely for ROAD-Waymo**; these are preparation times, not training throughput.

| Dataset / split | All accepted GT | GT with usable bbox | Bbox GT classes: moving / stopped / waiting / crossing |
|---|---:|---:|---|
| LOKI training | 262,071 | 118,451 | 68,524 / 9,303 / 15,834 / 24,790 |
| LOKI validation | 60,231 | 24,784 | 14,654 / 1,502 / 3,969 / 4,659 |
| LOKI test | 69,267 | 31,031 | 19,454 / 2,759 / 3,600 / 5,218 |
| ROAD-Waymo training | 167,280 | 167,280 | 86,095 / 39,059 / 11,336 / 30,790 |
| ROAD-Waymo validation | 45,388 | 45,388 | 24,018 / 11,232 / 3,317 / 6,821 |
| ROAD-Waymo test | 41,488 | 41,488 | 21,239 / 10,631 / 3,523 / 6,095 |

Full group/track/context/GT/class denominators for each coverage cohort remain in each manifest. These are label/observation-support audits, not model evaluation. Source-training numerical counts are **124,961** per box column / **123,936** per velocity column for LOKI, and **169,660 / 169,286** for ROAD-Waymo; neither source has guarded columns.

Independent local verification reloaded **all 18,972** eligible original/bbox pairs, checked every archive checksum and exact population/group assignments, directly reconstructed source-training means/stds, and checked all **439,178** usable bbox derivatives against a separate formula (maximum error **2.22e−16**). Eight 64-track test-split batches covering both source normalizations × both datasets × both new inputs were finite float32 with 14/20 columns; they were **not inferred or scored**. All **12,364** local/remote LOKI bbox archives are byte-identical. Evidence/reproduction: ignored `verification-local.json`, `verify-local.py`, `loki-remote-checksums.json`, `manifest-loki-remote.json` and preparation logs under `outputs/experiments/E001b/`. The remote LOKI manifest retains committed-revision provenance; its local precursor retains its earlier revision/code hashes.

### Geometry-source review and open gate (2026-10-09)

Across **254,263** selected same-ID/time ROAD/native FRONT pairs, **252,727** match within **1e−6** in each image-relative coordinate. The remaining **1,536 (0.604%)** span **1,231 tracks / 334 clips**; clipping explains none. Worst per-track coordinate differences have median **8.13 px** and maximum **109.67 px**; **85 tracks** exceed 20 px and **six** exceed 50 px. Pixel thresholds here describe severity only; they do not filter tracks or change inputs.

Three purposively chosen largest discrepancies in distinct clips were visually reviewed against the exact native RGB timestamp, with ROAD yellow and Waymo cyan overlays:

| Case | Selected identity and slot | Observation |
|---|---|---|
| 1 | `train_00216`, `25aa091d-1547-45d0-b9bb-45c4d40b619a`, slot 49 | ROAD box lies below the visible high-visibility-clothing pedestrian and encloses road surface. Native Waymo box encloses the person: clear ROAD geometry misplacement at this frame. |
| 2 | `train_00175`, `8189d569-f441-4e60-ae42-abc945b0f64a`, slot 0 | Equal-size boxes have shifted centers among adjacent pedestrians. Native Waymo follows the person more closely; one frame does not establish an identity error or its cause. |
| 3 | `train_00340`, `d3e7eb65-8538-4f77-b96e-066b5a482f04`, slot 53 | Dark/noisy scene with vertically shifted boxes; independent actor extent is inconclusive. |

Evidence: ignored `native-box-verification.json` and `box-review/case-{1,2,3}.png`, retained on both machines. JSON preserves coordinates/timestamps, affected identities, sampled image-size checks, reviewer findings and image hashes. This purposive review is not a population visual error rate; the difference count alone does not establish which source is correct in every pair. The cause of ROAD shifts remains **unknown**.

**The original ROAD-preferred geometry cache is not accepted for E001b training.** Its `bbox_valid` certifies numeric usability/source association, not visually correct actor enclosure. The user accepted **native Waymo FRONT only, with no ROAD fallback**, on 2026-10-09. All 254,263 ROAD-selected boxes on the frozen eligible 5 Hz population have native same-ID/time FRONT boxes; no ROAD-only selected boxes were found. This does not establish coverage outside that population. Original caches/review evidence stay intact; revised inputs use a fresh directory. This decision changes only E001b 2D features, not E001's accepted kinematic association policy.

### Native-only cache acceptance (2026-10-09)

At revision `a3a67c20cdc1ce40a36dc49266fc9ba29f3fb8ac`, revised preparation uses `bbox-native/` with the accepted policy. LOKI geometry is unchanged; ROAD uses source code 3 for every selected box and never source code 2. Box velocities and source-training statistics are recomputed, since former ROAD/Waymo source switches no longer interrupt native velocities. Checks cover missing/invalid Waymo boxes with ROAD GT retained, unlabeled native context, differing source geometry, and rejection of old-policy manifests. The initial increment deferred training; the later shared-runner increment is recorded below.

**Both native-only caches are complete and verified locally and on `aalto`.** LOKI preparation took 65.0 s locally. All 12,364 archives are byte-identical to the prototype and verified after transfer; population, class/box coverage and source-training statistics are unchanged. Independent reconstruction checked 182,202 bbox derivatives (maximum error 2.22e−16), training moments and two finite, unscored 64-track availability/geometry batches. All 13,368 protected E001 collection/setup hashes match.

The initial native-only revision passed **49 CPU tests / three CUDA skips** and **six E001b tests** in the isolated `aalto` cu118 environment, without CUDA allocation. GPU tests were not rerun alongside active training. Both original and active seed-sweep checkout revisions, source/script/dependency-file hashes and working diffs match before/after. Evidence: ignored `native-only/acceptance-cpu.log`, `verification-local.json`, `verify-local.py`, `verification-loki-aalto.json` and `active-training-{before,after,verification}.json`; remote acceptance log is `native-only/acceptance-aalto.log`.

The initial ROAD attempt stopped before any scene/track was saved because the recorded `/media/user20/F47C60057C5FC152` path was unavailable. All 9,576 protected ROAD E001 files matched afterward. Its failure evidence remains on both machines under `native-only/failed-road-unmounted/` (and a remote `.log`). Subsequent inspection, following the user's `AlreadyMounted` output, verified the native partition at **`/run/media/user20/F47C60057C5FC152/projects`**. New E001b preparation accepts `--waymo-root` at that mount, retains frozen camera-box checksum validation and records resolved native paths; E001 references remain unchanged. A relocation regression test verifies reload and checksum failure evidence. [Commands](../../README.md#e001b-bbox-extension) own mount restoration and the exact preparation invocation. At revision `be9e158441da2f6453c907cd602dc04d3082fe34`, fresh ROAD preparation finished in **41.0 s**, with all **9,576** protected E001 hashes unchanged.

ROAD retains **6,608 tracks / 507,144 context slots / 254,156 accepted GT frames**. All non-velocity coverage/class/group/track cohorts match the prototype. All **257,573** usable boxes now come from native FRONT, including unlabeled context; **zero** ROAD geometry boxes, invalid native boxes or native boxes outside image bounds are selected. Geometry changes above 1e−6 affect **1,536 slots**. There are **257,122** velocity-valid slots (**146 added, none removed**); source-training counts are **169,660** per box column and **169,382** per velocity column, with no guarded columns. Original prototype statistics/12 out-of-bounds ROAD boxes remain historical evidence above.

Independent native Parquet comparison checked every **507,144** ROAD context slot and all **257,573** selected boxes across **514** eligible clips. Maximum geometry error is **2.64e−16**; native camera-box and calibration hashes match the frozen/prototype evidence. Independent local reconstruction checked **all 18,972** original/new archive pairs, frozen populations/splits and source moments, plus **439,324** valid derivatives (maximum error **3.33e−16**). Eight finite float32, unscored 64-track batches cover both source normalizations × both datasets × both new configurations. No inference, training or test metric calculation was performed.

The final path-override revision passed **50 CPU tests / three CUDA skips**, **seven E001b tests** in the remote cu118 environment, and local links/`git diff --check`. Both E001 checkout revisions, source/script/dependency-file hashes and working diffs matched before/after ROAD preparation. Evidence on both machines: `native-only/acceptance-root-override-{cpu,aalto}.log`, `preparation-road-native-aalto.log`, updated `verification-local.json`, `verification-native-road-aalto.json` and `road-training-{before,after,verification}.json`. The local one-off `native-only/verify-local.py` retains the independent normalization/derivative/transfer-batch reconstruction. GPU tests were not repeated while E001 training was active.

### Shared runner and launcher (2026-10-09)

Implemented through shared `experiments/run.py` and `sweep.py`, with thin E001/E001b wrappers. E001b fixes BiLSTM, uses 14/20 inputs and retains source-only baseline/bbox statistics in config/provenance/checkpoints; both dataset/cache manifests and policies are referenced by hashes. Old-policy/stale caches fail validation before external logging. Run names are `E001b-<source>-<availability|geometry>-BiLSTM-seed<N>-<timestamp>`, with `-smoke` for restricted validation-only checks. Output is ignored `outputs/experiments/E001b/runs/`; W&B stays in `yammo-unipd/pedestrian-behaviour-labeling` with the existing tidy namespaces and no uploaded model/prediction artifacts.

**Diagnostics frozen before E001b test results:** preserve all ten E001 axes, and add `frame-bbox` and `bbox-velocity` (`missing`, `valid`) plus `track-bbox` (`never`, `partial`, `complete`). They use numerically verified native boxes over full resampled context; presence flags remain separate provenance. Save their per-slot membership, validity/source arrays and bbox archive/hash references with unpadded predictions. Shared slice logic recomputes equal-track weights, reports supported classes/denominators and leaves empty slices unscored. No GT/condition metadata enters model inputs.

At implementation revision `dee884b682495d6b74ff5aaef155b2d6fe2ef05d`, acceptance uses the combined [expected/actual report](../../scripts/check-e001-data.py), [E001b runner tests](../../tests/test_e001b_training.py) and shared scheduling checks. The CPU suite ran **56 tests: 52 passes / four CUDA skips**; the isolated remote CUDA suite passed **all 56**. Both expected/actual reports passed **40 scenarios** (four CPU skips, none on CUDA). Three captured pre-refactor E001 attempts (K MLP, K BiLSTM and K+T+R BiLSTM, source LOKI, seed 3, two epochs) matched exactly after extraction: best/last model tensors, selected epochs, source normalization, changing-weight history except runtime, saved prediction arrays/conditions and both test metrics. Synthetic E001b checks cover both sources and inputs, checkpoint prediction equality, source statistics for transfer, saved-metric regeneration, numeric-box/empty strata, 20-run matrix/bounded scheduling, and cache/W&B failures with retained local evidence. Evidence lives under ignored `outputs/experiments/E001b/training-acceptance/`: `acceptance-{cpu,cuda}.log`, `report-{cpu,cuda}/index.html` and `regression-{before,after}.json`. Restricted real GPU smoke verification is recorded separately below.

Two separately labeled native ROAD GPU smokes completed at that revision in the isolated `pedestrian-behavior-labeling-e001b` checkout, each with one epoch, two training updates (128 tracks) and one source-validation batch (64 tracks). **No held-out tests were evaluated.** Both best/last checkpoints reload, reproduce saved logits exactly on CUDA, and retain source-only normalization. Pickle-free predictions regenerate primary/all 13-axis metrics exactly; each smoke exports 20 PNG and 20 SVG files. Actual W&B readback verifies two `train/loss_step` records and one epoch record containing `train/loss_epoch`, `val/loss` and `val/macro_f1`.

| Validation-only smoke | W&B | Parameters | Runtime | Peak CUDA allocated |
|---|---|---:|---:|---:|
| Availability | [zu6tar46](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/zu6tar46) | 30,468 | 18.55 s | 110,745,088 bytes |
| Geometry | [pgsuv43w](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/pgsuv43w) | 30,852 | 23.51 s | 110,903,296 bytes |

These are implementation checks, not comparison scores or concurrency benchmarks. Local/remote run directories are `runs/acceptance-dee884b-{availability,geometry}-gpu-smoke/`. `training-acceptance/verification-gpu-smokes.json` records checkpoint/metric equality, exposure, plots and W&B checks. Upload audits confirm only agreed metrics/tables/plots: W&B implements metric tables as `run_table` artifacts and can materialize a `wandb-history` Parquet archive of the same logged fields. Its audited 81 columns contain scalar metrics and media references, not logits/targets/full predictions. No model/prediction artifacts were uploaded; checkpoints/predictions remain local. History readback ignores absent/null values in sparse rows.

All **61,857** original collection/setup/native-bbox files matched before/after; original and active E001 checkout revisions, source/script/dependency hashes and working diffs also match. Evidence is `training-acceptance/frozen-{before,after,verification}.json` and local `active-training-{before,after,verification}.json`. CUDA logs/reports, smoke outputs and numerical/upload audits are retained locally and on `aalto`; regression evidence remains local. Both uv environments passed lock/package checks. Plot inspection subsequently added markers for single-epoch CE and margins for empty-slice counts; exports regenerated from saved results are in local `training-acceptance/plot-verification/`, preserving the original smoke evidence. The final CPU suite again passed **52 tests / four CUDA skips** (`acceptance-cpu-final.log`); the plot-only correction does not change CUDA computation. Owner links/anchors and `git diff --check` passed.

Full comparisons remain user-controlled: **20 additional training attempts / 40 test cells**, 75 epochs, patience 8, seeds 0–4. Reuse the 10 matching K+T+R BiLSTM source/seed baseline attempts only when shared settings/populations/statistics agree. Prior E001 test inspection and native source review remain disclosed; this increment does not select settings from E001b held-out results.

## Interpretation and current evidence

Availability above baseline may reflect annotation-selection shortcuts. Geometry above availability supports an additional geometric contribution under this observation/model contract. Within-domain improvement with transfer degradation suggests camera/annotation/dataset dependence. Little geometry gain does not establish that RGB appearance or scene context is useless. Missing 2D annotations must not be interpreted as evidence of a particular behavior.

The native-only source policy is accepted; current cache verification is recorded above. Independent visual accuracy of every native box remains unknown. No full E001b comparison runs or held-out test results. Synthetic acceptance and clearly labeled source-validation smokes are implementation checks. Checkpoints/predictions for later valuable runs still need separate backup; external storage is TBD.
