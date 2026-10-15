# E001c — Bounding-box inputs

Preparation authorized **2026-10-15**. Rebuild every numeric input from native observations, retaining E001's eligible tracks, native associations, selected frames/times, accepted targets/overrides and splits. Existing E001/E001b caches/results remain immutable. This increment prepares caches only; no training or held-out scoring.

## Inputs and bounded comparison

| Configuration | Numeric inputs | With four flags |
|---|---|---:|
| `box3d` | XYZ center, three native dimensions, XYZ center velocity | 13 |
| `box3d-2d` | Above + image-relative center UV, width/height and all four 2D velocities | 21 |
| `box3d-2d-size-velocity` | Above + velocities of the three 3D dimensions | 24 |

All configurations receive `box3d_valid`, `box3d_velocity_valid`, `box2d_valid`, `box2d_velocity_valid`. Thus `box3d` already knows 2D availability; numeric geometry additions have a matched availability control. No heading, ego-state input columns or encoders. Proposed training: existing BiLSTM, both sources, seeds 0–4, max 75 epochs/patience 8: 30 attempts/60 cells. Reuse [E001 training/evaluation](../E001-kinematic-transfer/README.md#loss-and-training), including source-only normalization/selection. Training integration/execution is outside this increment. Disclose existing target-test-informed development; earlier results are references, not matched center-only ablations.

## Native and numeric contract

- XYZ extends E001's full ego/world transforms to z: subtract initial ego xyz and remove initial ego yaw, retaining world vertical. Dimensions retain native order: LOKI `dim_x/dim_y/dim_z`, Waymo `size.x/size.y/size.z`; no sorting, rotation or swaps. Column names length/width/height refer to these native axes, not independently verified anatomical axes. No heading input.
- Retain LOKI native `pos_x/pos_y/pos_z` without height correction. A purposive cloud sample supports center anchoring, but the [official release description](https://usa.honda-ri.com/loki) does not explicitly define center/base or dimension axes. Independent annotation accuracy remains unknown. Waymo explicitly specifies centers in [label.proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto).
- Re-read camera boxes/dimensions: native LOKI, native Waymo FRONT only, **no ROAD fallback**, independent of GT. Use geometric center UV, width/height divided by W/H; no clipping. E001b used bottom-v instead.
- Velocities use E001's immediate-neighbor unequal-time weighted centered rule or immediate one-sided secants, per second. No interpolation, farther-neighbor search, gap bridging or native velocity substitution. LOKI time remains nominal; ROAD time is physical.
- Audit center/size usability before collapsing masks. Fail with retained evidence if they differ instead of discarding usable information. Center and size velocities then share the same neighbors. Retain individual masks and native source-presence flags as provenance. Invalid 2D boxes are masked/reported.
- Fit population mean/std on valid **source-training context**, including unlabeled context, never validation/test. Physical missing entries are NaN; float32 model inputs become zero after standardization, with four unstandardized flags. Save all 20-column moments; configurations take their declared prefix and identical flags. Use existing guarded scales for entirely missing/near-constant columns.

## Data flow and artifacts

[Command](../../scripts/prepare-e001c.py) → [policy/assembly](../../src/pedestrian_behavior/experiments/e001c.py) → [native readers](../../src/pedestrian_behavior/datasets/boxes.py) → [shared numeric/cache operations](../../src/pedestrian_behavior/data/boxes.py). No prior numeric cache constructs inputs. Old E001 xy is checked against recomputation only; metadata supplies fixed associations, alignment, targets and splits.

Fresh ignored outputs: `outputs/experiments/E001c/box-cache/{loki,road-waymo}/`. Existing directories are rejected. Pickle-free `tracks/<frozen-name>.npz` contains float64 `values[T,20]`, four Boolean input flags, individual center/size validity, native `native_3d/loki_2d/road_2d/waymo_2d` presence, int64 `image_size[T,2]` and `targets[T]`, Boolean `gt_valid[T]`, and JSON identity/timestamp/extent/original-archive-hash references. Exact columns live in the manifest policy. GT never enters `box_inputs()`.

Manifest: frozen setup/collection hashes/entries/splits, native component/annotation/pose hashes and resolved paths, source-training moments, context/GT/class coverage by split/flag, invalid-box counts, mask audit, command/revision/code hashes, versions, times and failure/status. Replacement is atomic; incomplete/failed outputs are unusable. Verify native sources against frozen hashes and every original collection archive/manifest/audit/setup unchanged. [Runnable commands](../../README.md#e001c-bbox-cache).

## Acceptance and evidence

[Tests](../../tests/test_e001c.py): centers/XYZ transforms, size velocities, irregular timestamps, gaps/singletons, invalid sizes, unequal-mask refusal, source moments/missing zeros, identical flags across 13/21/24 inputs, both native formats, frozen targets/populations, no ROAD fallback, relocated components, persistence/corruption/failure evidence. Full regression protects E001/E001b after shared normalization gains explicit per-column validity.

Real rebuild/verification: **pending**. Record counts, paths, revision and checks here after completion. Cache coverage is not a model result; no E001c training has run.
