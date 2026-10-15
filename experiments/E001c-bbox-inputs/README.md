# E001c — Bounding-box inputs

Preparation and thin training integration authorized **2026-10-15**. Numeric inputs are rebuilt from native observations, retaining E001's eligible tracks, native associations, selected frames/times, accepted targets/overrides and splits. Existing E001/E001b caches/results remain immutable. All **30 five-seed comparisons / 60 test cells** completed on 2026-10-15; [the joint review](../E001b-2d-bbox-contribution/results-review.md#e001c-native-box-inputs) records results. Separate implementation acceptance is retained below.

## Inputs and bounded comparison

Final comparison agreed **2026-10-15**: test the contribution of 2D size velocities, retaining added 3D size velocities as a redundant-input control.

All three configurations include 3D center XYZ, native length/width/height and center velocity XYZ; image-relative 2D center UV, width/height and center velocity UV. These are **15 common numeric columns**: dimensions themselves and center velocities are always included.

| Configuration | Additional dimension velocities | With four flags |
|---|---|---:|
| `boxes` | None | 19 |
| `boxes-2d-size-vel` | 2D width/height | 21 |
| `boxes-2d-3d-size-vel` | 2D width/height + 3D length/width/height | 24 |

All configurations receive identical `box3d_valid`, `box3d_velocity_valid`, `box2d_valid`, `box2d_velocity_valid` flags. The main comparison is `boxes-2d-size-vel` versus `boxes`; the third configuration adds effectively zero-valued inputs, not useful observed 3D size dynamics (see native evidence below). No fourth configuration with only 3D size velocities. No heading, ego-state input columns or encoders.

Use the existing BiLSTM, both sources, seeds 0–4, max 75 epochs/patience 8: **30 training attempts / 60 evaluation cells**. Reuse [E001 training/evaluation](../E001-kinematic-transfer/README.md#loss-and-training), including source-only normalization/selection and all 13 E001/E001b observation diagnostics. W&B names include the final configuration, for example `E001c-road-waymo-boxes-2d-size-vel-BiLSTM-seed0-<UTC>`. [Training commands](../../README.md#e001c-training-and-evaluation) use the shared runner and bounded scheduler. Full comparisons are complete. Disclose existing target-test-informed development; earlier results are references, not matched center-only ablations.

## Native and numeric contract

- XYZ extends E001's full ego/world transforms to z: subtract initial ego xyz and remove initial ego yaw, retaining world vertical. Dimensions retain native order: LOKI `dim_x/dim_y/dim_z`, Waymo `size.x/size.y/size.z`; no sorting, rotation or swaps. Column names length/width/height refer to these native axes, not independently verified anatomical axes. No heading input.
- Retain LOKI native `pos_x/pos_y/pos_z` without height correction. A purposive cloud sample supports center anchoring, but the [official release description](https://usa.honda-ri.com/loki) does not explicitly define center/base or dimension axes. Independent annotation accuracy remains unknown. Waymo explicitly specifies centers in [label.proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto).
- Re-read camera boxes/dimensions: native LOKI, native Waymo FRONT only, **no ROAD fallback**, independent of GT. Use geometric center UV, width/height divided by W/H; no clipping. E001b used bottom-v instead.
- Velocities use E001's immediate-neighbor unequal-time weighted centered rule or immediate one-sided secants, per second. No interpolation, farther-neighbor search, gap bridging or native velocity substitution. LOKI time remains nominal; ROAD time is physical.
- Audit center/size usability before collapsing masks. Fail with retained evidence if they differ instead of discarding usable information. Center and size velocities then share the same neighbors. Retain individual masks and native source-presence flags as provenance. Invalid 2D boxes are masked/reported.
- Fit population mean/std on valid **source-training context**, including unlabeled context, never validation/test. Physical missing entries are NaN; float32 model inputs become zero after standardization, with four unstandardized flags. Save all 20-column moments; configurations take their declared prefix and identical flags. Use existing guarded scales for entirely missing/near-constant columns.

## Data flow and artifacts

[Command](../../scripts/prepare-e001c.py) → [policy/assembly](../../src/pedestrian_behavior/experiments/e001c.py) → [native readers](../../src/pedestrian_behavior/datasets/boxes.py) → [shared numeric/cache operations](../../src/pedestrian_behavior/data/boxes.py). No prior numeric cache constructs inputs. Old E001 xy is checked against recomputation only; metadata supplies fixed associations, alignment, targets and splits.

The completed cache contains every required column; final configurations select the first **15/17/20 numeric columns**, respectively, plus the same four flags. No native recomputation is needed. Frozen cache manifests retain preparation-era names (`box3d`, `box3d-2d`, `box3d-2d-size-velocity`) and 13/21/24-input selections. The helper retains these for preparation compatibility; training commands accept only the final names and 19/21/24 inputs. Each attempt records its actual numeric columns/flags and model dimension separately from the unchanged cache policy.

Fresh ignored outputs: `outputs/experiments/E001c/box-cache/{loki,road-waymo}/`. Existing directories are rejected. Pickle-free `tracks/<frozen-name>.npz` contains float64 `values[T,20]`, four Boolean input flags, individual center/size validity, native `native_3d/loki_2d/road_2d/waymo_2d` presence, int64 `image_size[T,2]` and `targets[T]`, Boolean `gt_valid[T]`, and JSON identity/timestamp/extent/original-archive-hash references. Exact columns live in the manifest policy. GT never enters `box_inputs()`.

Manifest: frozen setup/collection hashes/entries/splits, native component/annotation/pose hashes and resolved paths, source-training moments, context/GT/class coverage by split/flag, invalid-box counts, mask audit, command/revision/code hashes, versions, times and failure/status. Replacement is atomic; incomplete/failed outputs are unusable. Verify native sources against frozen hashes and every original collection archive/manifest/audit/setup unchanged. [Runnable commands](../../README.md#e001c-bbox-cache).

## Five-seed results (2026-10-15)

The user ran `scripts/train-e001c-seeds.py --jobs 8` in the isolated E001c checkout at revision `25a27632736bbea280b8621fef03197bcd7a4cb2`. Sweep `outputs/experiments/E001c/runs/seeds-20261015-114954-843530/` completed all 30 attempts without recorded failures. Selected epochs span **10–48**, completed epochs **18–56**; every attempt stopped on patience, none on the 75-epoch cap. Concurrent runtime is not a controlled comparison with the earlier four-/six-job sweeps.

The [joint E001/E001b/E001c review](../E001b-2d-bbox-contribution/results-review.md) owns mean/sample-SD tables, same-seed differences, class trade-offs and all PNG/SVG comparisons. Basic `boxes` averages below the earlier E001b geometry reference in all four directions; ROAD→LOKI is lower in every seed. Adding 2D size velocities shows no systematic benefit. The third configuration's differences cannot establish useful 3D size dynamics because those additional inputs are zero/numerical noise and input-layer initialization changes. E001c replaces several earlier features at once; these results do not isolate center choice, native dimensions or removal of ego-relative features.

[Metadata/selection checks](../../outputs/experiments/E001b/analysis/20261015-seed-review/verification.json) retain identical frozen populations/splits/settings and source-only normalization. The [prediction audit](../../outputs/experiments/E001b/analysis/20261015-seed-review/prediction-verification.json) exactly reconstructs all 60 new primary cells, verifies matching identities/times/GT/all condition memberships, and reconstructs all 13 axes for 12 seed-0 cells. All original 200 prediction hashes and 700 metadata hashes remain unchanged. Combined numerical evidence is under ignored `outputs/experiments/E001b/analysis/20261015-seed-review/` on both machines; full checkpoints/predictions remain on Aalto. The later box-input design follows prior test inspection: distinguish source-only training/selection from target-informed representation development.

## Acceptance and evidence

[Cache/selection tests](../../tests/test_e001c.py): centers/XYZ transforms, size velocities, irregular timestamps, gaps/singletons, invalid sizes, unequal-mask refusal, source moments/missing zeros, identical flags across preparation-era and final selections, both native formats, frozen targets/populations, no ROAD fallback, relocated components, persistence/corruption/failure evidence. [Training integration tests](../../tests/test_e001c_training.py) cover all six source/configuration combinations, explicit source normalization in both evaluation directions, exact checkpoint/prediction replay, saved primary/13-axis metric reconstruction, W&B content/failure propagation, archive/GT/population checks, 30-run commands and shared child scheduling. Existing regression protects E001/E001b.

### Training integration (2026-10-15)

[Single attempt](../../scripts/train-e001c.py) and [five-seed launcher](../../scripts/train-e001c-seeds.py) reuse the existing BiLSTM, shared training/checkpoint/evaluation/W&B lifecycle and bounded process scheduler. No new model architecture or training algorithm. `BoxTrackDataset` checks original/cache hashes, identity/alignment and cached versus frozen targets; full cache policy/population validation precedes external logging. Evaluation adds the same three verified-box axes as E001b after inference, preserving the ten E001 axes and native source flags. Checkpoints retain the source's full frozen box statistics; no unused K+T+R columns enter the model.

Outputs: `outputs/experiments/E001c/runs/` with the existing history/steps/status/provenance/checkpoints, both test prediction/metric archives and PNG/SVG plots. Restricted ROAD CUDA smokes evaluate source validation only. The isolated `aalto` checkout has its own uv environment and ignored E001/E001c links to existing outputs; the [manual launcher command](../../README.md#e001c-training-and-evaluation) remains reproducible. At this implementation acceptance no full comparisons had launched; subsequent execution/results are recorded above.

At implementation revision `973eb31f58ec100910c14829fe0ad4347dda2e75`, CPU regression passed **69 tests, five CUDA skips**; CUDA passed **all 69**. Twelve real validation samples covered both source normalizations × both datasets × all three configurations without inference/scoring. Synthetic full-attempt tests cover all six source/configuration combinations and both test directions; these are implementation fixtures, not scientific comparison results.

Three separately labeled real ROAD GPU smokes each completed **one epoch, two training updates (128 tracks), one source-validation batch (64 tracks), no held-out tests**:

| Configuration | Parameters | Runtime (s) | W&B validation-only smoke |
|---|---:|---:|---|
| `boxes` | 30,788 | 17.61 | [wo8i5sw0](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/wo8i5sw0) |
| `boxes-2d-size-vel` | 30,916 | 17.13 | [qziyj40o](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/qziyj40o) |
| `boxes-2d-3d-size-vel` | 31,108 | 17.98 | [4tqbofog](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/4tqbofog) |

All best checkpoints reproduce saved CUDA logits exactly and retain source statistics. Saved predictions regenerate primary/all 13-axis metrics; each smoke exports **20 PNG + 20 SVG** plots. Actual W&B readback confirms two `train/loss_step` records, one epoch record containing `train/loss_epoch`, `val/loss`, `val/macro_f1`, all 13 stratum plots and no uploaded `.pt`/`.npz` files. The shared uploader remains unchanged; W&B's table/media artifacts are numerical summaries and plots, not model/full-prediction artifacts.

All **80,831** files under protected E001 collection/setup, E001b native-box and E001c box-cache roots matched before/after. The original three remote checkout revision/diff snapshots also match. Checked 482 local links/anchors and `git diff --check`; inspected exported loss/box-stratum plots. Evidence on both machines: ignored `outputs/experiments/E001c/training-acceptance/` contains CPU/CUDA logs, per-smoke logs, `frozen-before.json`, `frozen-verification.json`, `real-sample-validation.json` and `verification-gpu-smokes.json`. Smoke outputs use `runs/acceptance-973eb31-<configuration>-gpu-smoke/`. These checks do not provide comparison scores or a concurrency benchmark. Separate backup storage for valuable future checkpoints/predictions remains TBD.

### Native rebuild (2026-10-15)

Both caches completed on `aalto` at preparation revision `850b9fd88e6b7f82552d90671b220c1b79925d69`, from isolated checkout `/home/user20/projects/pedestrian-behavior-labeling-e001c`. Outputs are under `/home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001c/box-cache/`; the local copy uses the same relative output root. No native numeric inputs were reused from older caches.

| Dataset | Tracks | Context slots | Accepted GT | Usable 3D | 3D velocity | Usable 2D | 2D velocity |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOKI | 12,364 | 403,813 | 391,569 | 391,569 | 391,399 | 183,815 | 182,202 |
| ROAD-Waymo | 6,608 | 507,144 | 254,156 | 459,626 | 459,600 | 257,573 | 257,122 |

All native center/size usability masks coincide: the four proposed flags suffice. Both datasets have zero invalid selected native 2D boxes. Across **all splits**, no track changes 3D dimensions above 1e−8 m. Maximum absolute 3D size velocity is **0 LOKI / 1.49e−10 m/s ROAD**; ROAD residuals are numerical noise. Source scales are guarded at one. The final comparison retains the third configuration as an explicitly redundant-input control by user agreement. Any difference between 21- and 24-input models would not establish a useful 3D size-velocity signal; input-layer parameter count/initialization also changes.

Preparation took **155.0 s LOKI / 111.3 s ROAD**. Every **22,944** protected E001 collection/setup file and **18,974** E001b cache file matched before/after; all three existing remote checkout revisions/diffs matched. Final CPU regression: **63 tests, four CUDA skips**. Final CUDA regression: **all 63 passed** at `a69a808b6dba3e2280949a9d16dbcc61b564845c`; source preparation code is unchanged from the cache revision. Checked **471 local links/anchors** and `git diff --check`. Cache coverage is not a model result; no E001c training, inference or W&B logging occurred.

Independent reconstruction checked **all 18,972 tracks / 910,957 context slots**: archive hashes, frozen entries/splits/targets, all ten velocity components, original planar velocities, E001b box conversion to geometric center, source-training moments and every configuration under **both** source normalizations. Maximum derivative reconstruction error is **zero** using integer source-time differences. The first read-only audit accidentally cast absolute Waymo timestamps to float seconds before differencing; correcting that audit recovered exact equality without changing cache code/data (`audit-note.txt`).

Both caches are also retained locally: checksum verification covered **every new/original archive** and both collection/setup hashes, with exact file membership. Ignored evidence on both machines: `outputs/experiments/E001c/verification/` contains preparation logs, CPU/CUDA test logs, `protected-before.json`, `protected-verification.json`, `reconstruction.json`, `verification-local.json` and `audit-note.txt`. The two cache manifests contain split/class coverage, moments, frozen/native references and all per-archive hashes. LOKI native vertical-anchor/axis semantics remain qualified above.
