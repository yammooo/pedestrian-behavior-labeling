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

### Native rebuild (2026-10-15)

Both caches completed on `aalto` at preparation revision `850b9fd88e6b7f82552d90671b220c1b79925d69`, from isolated checkout `/home/user20/projects/pedestrian-behavior-labeling-e001c`. Outputs are under `/home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001c/box-cache/`; the local copy uses the same relative output root. No native numeric inputs were reused from older caches.

| Dataset | Tracks | Context slots | Accepted GT | Usable 3D | 3D velocity | Usable 2D | 2D velocity |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOKI | 12,364 | 403,813 | 391,569 | 391,569 | 391,399 | 183,815 | 182,202 |
| ROAD-Waymo | 6,608 | 507,144 | 254,156 | 459,626 | 459,600 | 257,573 | 257,122 |

All native center/size usability masks coincide: the four proposed flags suffice. Both datasets have zero invalid selected native 2D boxes. Across **all splits**, no track changes 3D dimensions above 1e−8 m. Maximum absolute 3D size velocity is **0 LOKI / 1.49e−10 m/s ROAD**; ROAD residuals are numerical noise. Source scales are guarded at one. The size-velocity configuration therefore adds no meaningful observed information; retain requested cache columns, but revisit this redundant training comparison before launching it. Any difference between 21- and 24-input models would not establish a useful size-velocity signal.

Preparation took **155.0 s LOKI / 111.3 s ROAD**. Every **22,944** protected E001 collection/setup file and **18,974** E001b cache file matched before/after; all three existing remote checkout revisions/diffs matched. Final CPU regression: **63 tests, four CUDA skips**. Final CUDA regression: **all 63 passed** at `a69a808b6dba3e2280949a9d16dbcc61b564845c`; source preparation code is unchanged from the cache revision. Checked **471 local links/anchors** and `git diff --check`. Cache coverage is not a model result; no E001c training, inference or W&B logging occurred.

Independent reconstruction checked **all 18,972 tracks / 910,957 context slots**: archive hashes, frozen entries/splits/targets, all ten velocity components, original planar velocities, E001b box conversion to geometric center, source-training moments and every configuration under **both** source normalizations. Maximum derivative reconstruction error is **zero** using integer source-time differences. The first read-only audit accidentally cast absolute Waymo timestamps to float seconds before differencing; correcting that audit recovered exact equality without changing cache code/data (`audit-note.txt`).

Both caches are also retained locally: checksum verification covered **every new/original archive** and both collection/setup hashes, with exact file membership. Ignored evidence on both machines: `outputs/experiments/E001c/verification/` contains preparation logs, CPU/CUDA test logs, `protected-before.json`, `protected-verification.json`, `reconstruction.json`, `verification-local.json` and `audit-note.txt`. The two cache manifests contain split/class coverage, moments, frozen/native references and all per-archive hashes. LOKI native vertical-anchor/axis semantics remain qualified above.
