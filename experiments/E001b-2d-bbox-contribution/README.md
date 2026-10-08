# E001b — 2D bounding-box contribution

Planned diagnostic extension of [E001](../E001-kinematic-transfer/README.md), agreed 2026-10-08 while the user runs E001. No E001b training or test scores. Preparation is additive: the frozen E001 collections, setup, populations, splits and kinematics remain unchanged.

## Question and frozen comparison

Does annotated 2D bbox geometry add information for offline four-state labeling beyond pedestrian kinematics and ego interaction? Distinguish annotation availability from geometric contribution; the principal comparison is **geometry versus availability**.

| Configuration | Inputs | Dimension |
|---|---|---:|
| Baseline | Exact E001 **K+T+R** | 12 |
| Availability | Baseline + `bbox_valid`, `bbox_velocity_valid` | 14 |
| Geometry | Baseline + six normalized numeric bbox features + the same two flags | 20 |

The baseline includes `v_ped`, displacement from the first valid position, relative position and **relative velocity** (`v_ped − v_ego`), with the existing four vector-validity flags. Do not replace relative velocity with ego velocity or rename this input K+T+I. Reuse the four E001 K+T+R runs if all shared settings match. Two new inputs × MLP/BiLSTM × LOKI/ROAD-Waymo require **8 additional training runs / 16 evaluation cells**. Use descriptive configuration names rather than A/B/C, since E001 already uses A/B for model architectures.

Reuse E001's [models](../E001-kinematic-transfer/README.md#first-diagnostic-baseline), complete tracks and 5 Hz grid, accepted ontology/overrides, seed, source-only normalization, loss, AdamW, deterministic settings, training budget, checkpoint selection and [evaluation](../E001-kinematic-transfer/README.md#evaluation). Only input dimension changes. All eligible tracks remain, including 3D-only LOKI pedestrians. Test metrics cannot select settings/checkpoints; prior E001/native inspection is disclosed. No RGB encoder, LiDAR encoder, calibration reconstruction, extra dataset or native-data reprocessing.

## Additional observation contract

Given image-relative corners `(x_min/W, y_min/H, x_max/W, y_max/H)`, save columns in this order: horizontal center, bottom coordinate, width, height, horizontal center velocity and bottom-coordinate velocity. Speeds use the existing E001 immediate-neighbor derivative and selected source timestamps. LOKI uses **nominal seconds**, because physical timestamps are unknown; ROAD-Waymo uses the frozen frame timestamps, not reconstructed exposure times.

LOKI boxes use native `left, top, width, height` and dimensions from each selected PNG header. ROAD-Waymo uses the frozen export's normalized ROAD xyxy box when present; only absent ROAD boxes fall back to native FRONT `CameraBoxComponent` center/size geometry divided by FRONT calibration width/height. Do not use projected LiDAR boxes. Preserve native extents without clipping; count boxes outside image bounds. Record how many same-ID/time ROAD/native pairs are available and their maximum coordinate disagreement. Existing [ROAD association qualifications](../E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08) remain applicable.

`bbox_valid` requires finite coordinates and positive width/height. Nonfinite/degenerate boxes are masked and reported; an invalid present ROAD box is not silently replaced. Preserve `loki_2d`, `road_2d` and `waymo_2d` annotation-presence flags separately: presence alone is not verified numeric usability, visibility, occlusion or a behavior target.

`bbox_velocity_valid` requires a valid box at the current slot and an immediate valid neighbor from the **same annotation source**. Never bridge missing slots or source changes; never interpolate. Two neighbors use E001's unequal-time weighting, one neighbor uses its secant and a singleton has no velocity. This short temporal preprocessing is available to both architectures and to both new configurations.

Standardize each continuous column using valid **source-training context**, including unlabeled context, only. Reuse the existing population-standard-deviation/guard rule; missing numbers become zero after standardization. Do not standardize flags. B and C receive exactly the same flags. Source baseline and bbox statistics must both be supplied explicitly for opposite-dataset evaluation. Image-size normalization does not compensate for focal length or camera mounting differences; intrinsics are not required.

## Extension artifacts and data flow

Implementation: [native bbox readers](../../src/pedestrian_behavior/datasets/bbox.py) → [geometry and validation](../../src/pedestrian_behavior/data/bbox.py) → [E001b preparation and sample assembly](../../src/pedestrian_behavior/experiments/e001b.py). [Commands](../../README.md#e001b-bbox-extension) own execution.

Ignored output: `outputs/experiments/E001b/bbox/<dataset>/`. One pickle-free `tracks/<E001 archive name>.npz` for every eligible E001 track; no copied kinematic arrays. Each archive contains float64 `bbox_features[T,6]`, boolean usability/presence arrays, uint8 source codes, int64 `image_size[T,2]`, and scalar JSON metadata. Metadata records locator, exact selected frame keys/times, original track checksum and invalid-box issues. Missing numbers remain NaN in physical caches.

`manifest.json` records preparation status/failures, policy, native/source and original collection/setup checksums, per-track bbox/original checksums, source provenance, code/environment, times, source-training statistics and split-level coverage. Creation requires a fresh output directory and retains failures. Original E001 manifest/audit/setup and **all candidate archives**, including excluded candidates, are checksummed before/after preparation. Loading rejects stale, corrupted, incomplete or misaligned extensions and any population change.

`dataset_from_extension()` retains the existing sample/collation interface and appends 2 or 8 float32 columns to exact K+T+R inputs. Runtime uses saved caches; it does not access native data or fit statistics. E001's existing feature, loader and runner code is unchanged. The E001b training command and final plotting/logging wiring are a later increment; this command only prepares evidence.

## Execution gate and acceptance

Before training, verify coordinates, dimensions, timestamps and same-frame native IDs; quantify full/valid/missing/velocity-valid support and 2D-without-3D context on frozen populations. Report supporting groups/tracks/context frames/accepted GT/class frames and class groups/tracks, plus never/partial/complete usable-box track cohorts. Coverage is observation evidence, not held-out predictive performance. No arbitrary coverage threshold is introduced; inspect documented limitations before authorizing training. Freeze additional settings before E001b test scores.

[Five acceptance scenarios](../../tests/test_e001b.py) cover hand-calculated image geometry/unequal-time derivatives; gaps/singletons/source changes/invalid dimensions; identical masks and missing-zero/source normalization; both native formats through cache reload and 14/20-column batches with unchanged kinematics/targets and checksum/corruption failures; and actual E001 setup with held-out-value independence and retained native-mismatch failure evidence. Full CPU/CUDA regression checks and real collection evidence are recorded below when available.

## Interpretation and current evidence

Availability above baseline may reflect annotation-selection shortcuts. Geometry above availability supports an additional geometric contribution under this observation/model contract. Within-domain improvement with transfer degradation suggests camera/annotation/dataset dependence. Little geometry gain does not establish that RGB appearance or scene context is useless. Missing 2D annotations must not be interpreted as evidence of a particular behavior.

Real preparation/coverage verification: in progress. Independent visual accuracy of every native box remains unknown. No E001b training, metrics, W&B runs or comparison results. Checkpoints/predictions for later valuable runs still need separate backup; external storage is TBD.
