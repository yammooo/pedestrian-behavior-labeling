# ROAD-Waymo

Status: Acquired on aalto; ROAD-authoritative same-frame 3D index inspected and rechecked

Last updated: 2026-10-06

## Candidate role

Initial real behaviour dataset with verified partial multimodal correspondence in the acquired train/validation subset. [E001](../../experiments/E001-kinematic-transfer/README.md) uses it as source and target in separate comparisons with LOKI, with an accepted experiment-specific projection. Linkage and similar class names do not establish universal semantic equivalence. [Gate 1](README.md#outstanding-dataset-checks) owns remaining native-data acceptance checks.

## Documented annotation form

The [official repository](https://github.com/salmank255/Road-waymo-dataset) describes frontal videos with agent, action, location, frame-box and tube annotations. The acquired `road_waymo_trainval_v1.0.json` and exported vocabulary preserve `tube_uid`, native class-ID lists and complete ROAD box records. Box coordinates are normalized `[x1,y1,x2,y2]`; original ROAD geometry is retained even when it differs from Waymo's camera box.

Observed pedestrian action strings are `Mov`, `MovAway`, `MovTow`, `PushObj`, `Stop`, `Wait2X`, `Xing`, `XingFmLft`, and `XingFmRht`. Decode original IDs using `road_label_definitions.json` (the exporter uses `all_*_labels`). Action `Xing` and location `xing` are distinct. Definitions, overlaps and boundaries remain native evidence; [E001](../../experiments/E001-kinematic-transfer/README.md#accepted-four-state-projection) owns its accepted projection/conflict policy and loss. Frame-level AV actions remain in the original ROAD JSON and are not pedestrian targets.

The [ROAD paper, Table 12 and §3.1](https://arxiv.org/pdf/2102.11585#page=19) defines native `Mov` as travel across the lane/traffic direction, rather than all locomotion. `MovTow`/`MovAway` describe an agent's movement toward/away from the AV; `Stop` describes a stationary agent considered ready to resume motion. `Wait2X` adds pavement position and road-facing orientation to stationary behavior. No numeric speed cutoff is supplied in those definitions. [ROAD-Waymo §III](https://arxiv.org/html/2411.01683v3#S3) states that it carries forward ROAD's annotation strategy. E001 deliberately collapses direction-specific movement labels into a broader output MOVING; do not confuse that output with native `Mov`, or agent actions with AV-action labels. Video/ego-relative image motion alone does not establish pedestrian world motion.

The [ROAD-Waymo paper, v3](https://arxiv.org/abs/2411.01683v3) describes action/location/event understanding and compatibility with UK ROAD for real-country domain adaptation (ROAD++). This is paper-level motivation, not evidence that acquired 3D associations or ROAD/LOKI taxonomies are accepted. Existing population figures below remain explicitly sourced to paper v1 or the inspected release.

## Waymo correspondence and selection

Original Waymo is the intended source of sensor/3D observations. Its [labeling specifications](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md) describe camera and 3D labeling. Compatible release, timestamps, associations, ego data, and map coverage must be checked rather than inferred from the ROAD tube ID.

Waymo has separate Perception and Motion datasets; [the official repository](https://github.com/waymo-research/waymo-open-dataset) describes maps with the Motion dataset. Do not assume those map assets correspond to every ROAD-Waymo segment.

The frontal annotation population differs from LOKI's 3D-first population. Additional Waymo 3D pedestrians are not automatically ROAD behaviour-labeled. Matching must measure retained/excluded populations and ambiguous associations.

## Acquired index and access (2026-10-02)

On SSH host `aalto`:

- ROAD-Waymo: `/media/user20/F47C60057C5FC152/projects/road_waymo`
- Waymo Perception v2: `/media/user20/F47C60057C5FC152/projects/waymo_v2`
- Handoff to read first: `/home/user20/road_waymo_mapping/MERGED_PEDESTRIANS.md`
- Verified index: `/home/user20/road_waymo_mapping/merged_pedestrians_20261002/`

Use `pedestrians.csv.gz` for annotations and `scene_manifest.json` for original modality paths. `report.json`, `export_audit.json`, `clip_summary.json`, and `road_label_definitions.json` contain provenance, coverage and native vocabularies. The manifest covers **798 distinct scenes/clips**: ROAD's 600 train and 198 validation clips, all drawn from Waymo **training**. ROAD's 202 test videos have no downloaded behaviour annotation JSON and contribute no rows. ROAD and Waymo split names are not interchangeable; preserve both split fields.

Source evidence lives beside the index: `BOX_MAPPING_RESULTS.md`, `PEDESTRIAN_COVERAGE.md`, `join_boxes.py`, `audit_pedestrian_coverage.py`, and `merge_pedestrians.py`. The older coverage report used a strict Waymo-pedestrian class filter; its 426,077 pairs and 6,525 tracks are historical audit counts, superseded for training by the ROAD-authoritative policy below.

### Identity and supervision policy

The inspected join is:

```text
ROAD clip + one-based frame -> Waymo scene + frame_timestamp_micros + FRONT (1)
ROAD tube_uid -> native camera_object_id
official camera_to_lidar_box_association -> laser_object_id
native lidar_box(scene, timestamp, laser_object_id) -> same-frame geometry
```

All exported pedestrian timestamps use `timestamp_basis=automatic_verified`. Other classes in the general box audit include human-confirmed alignment; that caveat does not apply to this pedestrian export. The join uses native IDs and published associations, not nearest-box or IoU assignment. Missing identities and boxes remain missing; no interpolation or inferred association supplies supervision.

[Accepted policy](../archive/2026-10-02-road-waymo-linkage.md): ROAD defines the pedestrian class. Filter/train using `merged_agent_label=Ped`, not `waymo_3d_type`. Apply `has_3d_box` as the validity mask for exported same-frame 3D geometry; a nonempty `official_laser_object_id` alone is insufficient. E001's additional native-ID observations need a separate validity/provenance mask and do not change that CSV field. CSV booleans are strings: compare to `"True"`, not Python truthiness of the field.

Preserve `road_annotation_json`, original label IDs, native `waymo_3d_type`/`waymo_3d_label`, `semantic_disagreement` and all original geometry. **419 paired rows across 15 camera tracks in 10 clips** have Waymo camera type Pedestrian and associated LiDAR type Cyclist. ROAD's class takes precedence, but these boxes are not resized to person-only geometry. [E001's qualified acceptance](../../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08) reviews all 15 plus two native-context-only Cyclist tracks; several visible bicycle actors support differing geometry/conventions, while small/occluded cases remain semantically inconclusive.

### Track locators and native payload

The [shared schema](README.md#reader-and-prepared-track-schema) uses `{"clip":"train_00383","tube_uid":"7fc2c418-f760-496a-927c-717d8df6ad06"}` as a canonical JSON track locator. Frame keys are integer Waymo microsecond timestamps, so no duplicate source-time array is saved. Scene provenance retains Waymo split/segment/component addresses and officially associated LiDAR IDs; source/export and numeric component checksums support reproduction. Unknown release information remains unknown.

Prepared native semantics retain the original ROAD annotation fields/IDs (excluding its box, retrievable in the source), decoded action/location lists and **all duplicate annotation IDs**. `native_metadata` separately preserves export `has_3d_box`, association status, export 3D type/label and semantic disagreement, plus the observed native LiDAR type. Export pairing remains distinct from extended native 3D availability. Native extensions add context, without ROAD behavior GT. No target correction changes the native payload.

[Official v2 box definitions](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/box.py) place native LiDAR labels in the frame vehicle coordinates; preparation uses `[VehiclePoseComponent].world_from_vehicle.transform`. [Official frame timing](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto) clarifies that frame start time and vehicle-pose instant differ; the pose defines the label frame. Derived velocities use frame timestamps and should not be interpreted as exact sensor-exposure-time measurements. Camera-synchronized boxes remain a distinct component and are not substituted.

### Prepared inventory (2026-10-07)

The complete reader/preparation audit retains all **9,573 ROAD candidates** across **562 clips with pedestrian annotations**, with **624,170** 5 Hz slots and an initial ego anchor for every candidate. Full native union observations total **1,211,822**: **712,630** unique ROAD annotation frames, **721,403** native FRONT observations and **922,321** native linked 3D observations (overlapping streams). The **10 identical repeated ROAD annotations** retain their original IDs. No conflicting duplicate/association or preparation failure was encountered; this structural check does not replace visual association acceptance.

After selection, **460,867** slots have usable 3D position and **460,841** usable pedestrian velocity; **6,629 tracks** retain at least one usable position. Five of the previously observed 6,634 native-3D tracks lose their sparse 3D evidence during declared scene-frame selection; all candidates remain saved. **292 grid slots** have no scene frame within 75 ms (**0.047%**), consistent with previously observed native timing gaps. LOKI has zero such missing selections. These are preparation counts, before target projection/eligibility/splits.

Evidence: ignored `outputs/experiments/E001/reader-preparation/road-waymo/{manifest,audit}.json`, originally generated in the Git checkout on `aalto` and copied locally for independent reload. The [eight frozen native cases](../../experiments/E001-kinematic-transfer/reader-cases.json) retain both Moving/Stop conflicts without target overrides.

### Measured population

The export and this inspection's independent full-CSV recheck agree:

| Measure | Count |
|---|---:|
| ROAD pedestrian annotation rows | 712,640 |
| Same-frame 3D boxes (`has_3d_box=True`) | 426,496 (59.85%) |
| Official association, native box missing at that frame | 98,024 |
| No official association | 187,980 |
| ROAD camera ID absent from downloaded camera annotations | 140 |
| ROAD tracks `(road_clip_id, road_tube_uid)` | 9,573 |
| Tracks with at least one paired frame | 6,540 |
| Tracks paired at every ROAD-labeled frame | 4,606 |
| Tracks partly paired | 1,934 |
| Tracks with no paired frame | 3,033 |
| Clips with ROAD pedestrian labels | 562 |
| Clips with paired pedestrian observations | 512 |

There are **712,630 distinct frame/object observations**, of which **426,491** are paired. Ten extra annotation rows repeat an observation; the recheck found identical normalized boxes and action/location labels for those repeats. Keep raw rows intact; E001 deduplicates identical observations while retaining all annotation IDs and rejects conflicting repeats. Annotation-row identity is `(road_clip_id, road_frame_1based, road_annotation_id)`; pedestrian identity is clip-scoped `road_tube_uid`, not the annotation ID or an unscoped LiDAR ID.

| ROAD split | All pedestrian rows | Paired rows |
|---|---:|---:|
| train | 543,372 | 320,048 |
| val | 169,268 | 106,448 |

| Native pedestrian action | Rows carrying label | Paired rows |
|---|---:|---:|
| Mov | 74,717 | 42,865 |
| MovAway | 146,822 | 85,180 |
| MovTow | 138,535 | 88,920 |
| PushObj | 5,322 | 3,424 |
| Stop | 190,902 | 102,167 |
| Wait2X | 46,706 | 30,135 |
| Xing | 41,572 | 31,855 |
| XingFmLft | 35,471 | 22,497 |
| XingFmRht | 38,109 | 23,753 |

Labels can overlap; these are observation counts, not unique tracks or action episodes. Paired-only filtering changes class coverage (approximately 53.52% for Stop versus 76.63% for Xing). Keep all ROAD observations for applicable 2D behaviour supervision, and mask absent 3D rather than silently selecting only complete tracks.

### Geometry, modalities and validation limits

`waymo_box3d_center_size_heading_json` is native `[cx,cy,cz,sx,sy,sz,heading]` in the vehicle frame, with metres and radians. This is a moving ego frame, not a world trajectory; derive velocity only after appropriate ego compensation or verification of native speed attributes. Original optional attributes, including point counts and speed, are in `waymo_3d_attributes_json`; null values remain missing.

The manifest references camera images, native LiDAR, calibration, vehicle/LiDAR poses, native and camera-synchronized boxes and other components in place. A read-only native check on 2026-10-08 found camera-calibration Parquet files for all **562** pedestrian clips on `aalto`, each with one FRONT row and finite `f_u`, `f_v`, `c_u`, `c_v`, `k1`, `k2`, `k3`, `p1`, `p2`. These supply camera intrinsics/distortion; the component also stores camera-to-vehicle extrinsics and image dimensions ([official schema](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/context.py)). Files live under `/media/user20/F47C60057C5FC152/projects/waymo_v2/training/camera_calibration/`; this availability check does not validate RGB projection. File existence does not establish per-frame content or annotation coverage. Sampled `lidar` Parquet stores range-image returns, not PLY vertices; point-cloud visualization needs decoding and calibration/pose handling. Native and camera-synchronized boxes have distinct timing/geometry provenance. FRONT behaviour supervision does not label every pedestrian visible to other cameras or LiDAR. HD map coverage is still unverified.

The [full structural recheck](../archive/2026-10-02-road-waymo-linkage.md#structural-recheck) supports export consistency, not a measured visual matching error rate.



Remaining work: document acquisition/release history, preserve runnable mapping/checker code in version control, characterize action episodes and physical-ID overlap, investigate timing irregularities below, audit semantic disagreements and varied associations visually, and declare the training acceptance rule. The merger currently hardcodes Waymo `training` paths; this matches the inspected subset but must be changed before processing another split. The combined native reader/preparation is now implemented; no model has run.

## Track extents and gaps (2026-10-05)

Full annotation-only index scan, with identity `(road_clip_id, road_tube_uid)`: context spans the first to last ROAD pedestrian observation, including empty intermediate frames. Ten identical repeated observations are counted once, reconciling to 712,630 observations and 426,491 paired observations. Extent durations use exact verified Waymo endpoint timestamps, independently of gallery playback.

| Population | Tracks | Mean s | Median s | P90 s | P95 s | P99 s | Max s |
|---|---:|---:|---:|---:|---:|---:|---:|
| All ROAD pedestrians | 9,573 | 7.99 | 6.20 | 19.10 | 19.70 | 19.80 | 19.803223 |
| At least one paired 3D observation, full ROAD extent | 6,540 | 8.18 | 6.50 | 19.10 | 19.70 | 19.80 | 19.803223 |

3,709 tracks (38.74%) have internal empty ROAD frames, accounting for 7.92% of all first-to-last frame positions. The longest empty run has 181 native frame positions; its adjacent observed boundaries are 18.199996 s apart. Paired 3D is present at 55.11% of full-span positions across all tracks, or 78.84% among tracks with any 3D. These denominators include empty ROAD frames, unlike paired coverage over labeled rows. No missing observation or behavior target was filled.

Median timestamp interval per ROAD index step is 0.099993 s. However, 110 observed intervals exceed 0.15 s per index step (maximum 0.400036 s); causes remain unverified. Do not assume every ROAD index step is exactly 0.1 s or blindly take alternate frames to produce 5 Hz. Conservative capacity `ceil(duration × 5) + 1` has median **33**, P95 **100**, maximum **101** positions. Small timestamp deviations can add a capacity position. This is a capacity estimate, distinct from [E001's selected-frame grid](../../experiments/E001-kinematic-transfer/README.md#first-baseline-temporal-representation).

Whole tracks containing `Wait2X` have median 13.70 s, longer than the overall median; action cohorts overlap and are not action-episode durations. Durations are bounded by downloaded train/validation clips; no test supervision, additional native LiDAR-only tracks or ID stitching is included.

Local report: `outputs/inspection/track-statistics/README.md`; `road-waymo/tracks.csv` records every pedestrian, and `road-waymo/summary.json` includes quantiles, split/cohort/action window coverage, masks, source CSV checksum and timing outliers. Remote results: `/home/user20/road-waymo-gallery-20261005/track-statistics-road-waymo-final/`. [Reproduction commands](../../README.md#inspect-track-lengths-and-gaps). The 101-position capacity applies to ROAD-only extents under that rule, not the later native-ID union or final eligible 5 Hz samples.

## Native identity and context audit (2026-10-06)

Read-only scans on `aalto` used the acquired CSV/manifest and their native `lidar_box` and FRONT `camera_box` components in place. Counts below precede E001 resampling, pose-validity and accepted-label filtering; they are not final eligible populations. The [versioned association audit](../../scripts/audit-road-associations.py), run at verified revision `055df5f08e84c4695a78b29e637810e0bc4f1941` on 2026-10-08, reproduces the candidate/link/observation/extension counts below and checks every saved source/component hash, native extent and exported native pair/type. Historical whole-manifest/prefix/file-count and union-before/after summaries remain evidence from the original scan rather than outputs of that narrower executable recheck. [Command](../../README.md#prepare-and-inspect-complete-tracks) and [E001 acceptance](../../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08).

| Identity / grouping check | Observed result |
|---|---:|
| ROAD tubes with exactly one official LiDAR ID | 6,809 |
| ROAD tubes without an official LiDAR ID | 2,764 |
| Tubes with multiple official LiDAR IDs | 0 |
| Scene-scoped LiDAR IDs shared by different ROAD tubes | 0 |
| Pedestrian clips / native segments | 562 / 562 |
| Manifest clips / unique Waymo segments | 798 / 798 |

No repeated leading segment-name prefix was found. These checks support scene-scoped joins; they do not prove independent recordings, geography or different physical people. The acquired ROAD train/validation versus Waymo training split discrepancy above remains provenance, not a justification for mixing native split names.

E001 assumes every clip is independent by explicit user instruction (2026-10-08). This is a protocol assumption, not a remaining independence audit/approval gate.

| Native context check | Observed result |
|---|---:|
| LiDAR-box files scanned for linked scenes / missing files | 516 / 0 |
| ROAD tubes with any native 3D observation on the unique linked ID | 6,634 |
| Tubes gaining 3D only outside the export's same-frame pairs | 94 |
| Tubes with 3D before ROAD start / after ROAD end / either | 2,123 / 4,589 / 5,326 |
| Additional native 3D observation timestamps beyond export pairs | 495,830 |
| Export pairs absent from native 3D | 0 |
| FRONT camera-box files scanned | 562 |
| ROAD tubes found by native FRONT ID | 9,570 of 9,573 |
| Tubes with FRONT observations before ROAD start / after ROAD end / either | 120 / 154 / 253 |
| Additional FRONT observation timestamps absent from ROAD | 8,913 |

All 495,830 additional native 3D timestamps lack a corresponding ROAD observation; they supply context without behavior GT. Counts for before/after overlap. The ROAD+3D union's maximum extent is **19.824513 s**; this scan excludes the additional FRONT extension, so it is not the final all-modality maximum. The three ROAD IDs absent from native FRONT are retained as original ROAD evidence, not replaced by inferred camera identities.

The complete 2026-10-08 native type scan found **921,036 Pedestrian / 1,285 Cyclist** observations on linked IDs, with Cyclist geometry on **17 tracks**; two have Cyclist observations only in native context and were absent from the export's disagreement flags. No other native 3D type was found. A purposive **33-track / 236-snapshot** RGB/BEV/crop review covers all 17, frozen cases and deterministic small/crowded/large-velocity cases. No wrong actor link was demonstrated in those views; tiny, dark, occluded and out-of-FRONT context cannot all be independently confirmed. Visible bicycle and scooter actors mean native class differences/large speeds cannot automatically be treated as erroneous associations. E001 accepts the official links with these qualifications; no population visual error rate is claimed. Ignored evidence: `outputs/experiments/E001/association-review/complete/`, including `audit.json`, `index.html` and a snapshot `SHA256SUMS`; [versioned per-case record](../../experiments/E001-kinematic-transfer/road-association-review.json). Original acquisition/release revisions remain unknown; no native annotations, geometry or saved archives were changed.

The export contains **6,521** tracks with at least one same-frame 3D box and at least one native action (before the accepted projection/grid). In this candidate population, **16** have one native behavior observation, **151** have at most five, and **27** have one same-frame 3D observation. This does not require input and label to coincide or establish post-resampling eligibility.

### Native action overlaps

An independent full-CSV recheck on 2026-10-06 counted action sets per `(road_clip_id, road_tube_uid, frame_timestamp_micros)`, deduplicating repeated observations and checking identical action sets. Of **712,630** observations, **2,034** have no action, **703,087** have one, **7,466** have two and **43** have three. Thus **7,509 (1.05%)** have simultaneous action labels; this is frame-level overlap, not merely a track changing behavior over time. The scan read `pedestrians.csv.gz` in the acquired index with remote Conda `pedestrian-behavior` and modified no source data.

On 712,630 deduplicated ROAD observations, mapping the eight motion/stop/wait/crossing strings into four states **before applying E001's crossing priority** yields 708,439 unique mapped states, 2,147 mapped-state conflicts, 10 unmapped-only observations and 2,034 with no native action. Of the unique-state observations, 5,269 also carry an unmapped co-label. These are native projection diagnostics, not final accepted-GT counts.

| Exact simultaneous action set | Observations | Tracks |
|---|---:|---:|
| `MovTow + PushObj` | 2,168 | 26 |
| `MovAway + Xing` | 1,222 | 21 |
| `MovAway + PushObj` | 733 | 13 |
| `PushObj + Xing` | 699 | 14 |
| `Mov + PushObj` | 681 | 12 |
| `PushObj + XingFmLft` | 438 | 5 |
| `MovTow + Xing` | 427 | 13 |
| `PushObj + Wait2X` | 219 | 3 |
| `PushObj + Stop` | 210 | 3 |
| `Mov + XingFmRht` | 152 | 2 |
| `Mov + XingFmLft` | 139 | 1 |
| `PushObj + XingFmRht` | 121 | 2 |
| `Mov + Xing` | 105 | 3 |
| `MovTow + Stop` | 49 | 1 |
| `Mov + MovTow` | 47 | 1 |
| `Mov + MovAway` | 46 | 1 |
| `MovTow + PushObj + Xing` | 43 | 1 |
| `MovAway + Stop` | 10 | 1 |

This table is exhaustive for multi-action sets in the acquired train/validation export. Observation rows are disjoint; tracks may occur in multiple rows. Projected conflicts are **2,088 MOVING+CROSSING** and **59 MOVING+STOPPED**. No Stop+Wait2X or waiting+crossing set was observed in this subset. E001 keeps crossing for the former and now requires audited corrections for the latter rather than permanent exclusion; native action lists remain unchanged.

### Movement/stop conflict inspection (2026-10-06)

Both conflict tracks have same-frame native 3D boxes throughout and native type Pedestrian, without a semantic-disagreement flag. A read-only scan transformed box centers with `[VehiclePoseComponent].world_from_vehicle.transform`, then measured horizontal endpoint displacement over verified timestamps. Measurements use exported same-frame boxes, not independently reconstructed sensor trajectories.

| Clip / native tube ID | ROAD frames / action set | Duration s | World displacement m | Net speed m/s | Ego displacement m | Proposed correction |
|---|---|---:|---:|---:|---:|---|
| `train_00383` / `7fc2c418-f760-496a-927c-717d8df6ad06` | 81–129 / `MovTow + Stop` (49 observations) | 4.799600 | 6.208154 | 1.293473 | 28.653447 | MOVING |
| `train_00425` / `e1dce432-b655-4627-bea9-8f7e0a5f17be` | 1–10 / `MovAway + Stop` (10 observations) | 0.899799 | 0.073747 | 0.081960 | 5.730265 | STOPPED |

Median successive-box horizontal speed is 1.299458 and 0.081960 m/s respectively. The first conflict spans the whole observed 4.8 s track, rather than a one-frame transition. Three native FRONT frames per track were inspected (81/105/129 and 1/5/10); dark images, small boxes, occlusion and endpoint truncation limit independent visual confirmation. These measurements support different corrections and argue against a universal movement/stop priority. They do not prove annotation error, source interpolation policy or exact per-frame physical speed. Do not use an automatic input-speed threshold to redefine benchmark GT.

On 2026-10-08, the user reviewed the saved 5 Hz RGB/LiDAR examples and confirmed the two corrections above. A native CSV recheck verified their exact frame ranges, action sets and timestamps, covering all **59** known movement/stop conflicts. E001 owns the accepted [versioned overrides](../../experiments/E001-kinematic-transfer/behavior-overrides.json). Review artifacts: `outputs/experiments/E001/conflict-review/`; native annotations and prepared tracks remain unchanged. [E001 association acceptance](../../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08) is now recorded separately with visual limitations.

## RGB + LiDAR BEV inspection gallery (2026-10-02)

The repository now provides a small native [inspection reader](../../src/pedestrian_behavior/datasets/road_waymo.py) and [gallery command](../../README.md#inspect-road-waymo). They consume the unchanged index and manifest, support Waymo-root relocation, and stream native component arrays. They do not define a final training adapter or rebuild the association stage.

Each 1564×604 MP4 contains **all original FRONT frames**, including frames without
the selected track's ROAD annotation. Playback is 5 FPS to match the LOKI gallery;
Waymo's approximately 10 Hz recordings therefore play at roughly half speed.
RGB keeps its aspect ratio inside the 960×604 panel. The 604×604 BEV shows both
returns from all available LiDAR sensors in a fixed 40 m view, with +x up and +y
right. Points and yaw-rotated boxes use the native same-frame vehicle coordinates.
Only the selected pedestrian's available ROAD 2D box and native 3D footprint are
yellow. Action and location labels remain separate; original Cyclist boxes remain
unchanged and class disagreements are marked.

Only the BEV **view center** is interpolated through missing boxes: known positions
are transformed to world coordinates, interpolated by timestamp, then returned to
the current vehicle frame. The nearest known world position is held at either end.
Tracks without any 3D box use a marked ego-centered view. Missing 2D/3D boxes and
point-cloud components are marked separately; absent boxes are never drawn.
An available cloud with no points inside the focused view is marked explicitly.
Identical repeated ROAD observations become one video observation while retaining
all annotation IDs; conflicting repeats raise an error.

LiDAR decoding follows Waymo's [range-image geometry](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/range_image_utils.py)
and [v2 conversion conventions](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/utils/lidar_utils.py),
including per-pixel TOP sensor ego compensation. No sweep accumulation, inferred
associations, camera-synchronized-box substitution or person segmentation is used.
Component streams must have ordered timestamps; malformed/misaligned data fails
explicitly. This is an inspection reader, not a frozen training-loader contract.

[Two real gallery runs](../archive/2026-10-02-road-waymo-linkage.md#visual-samples-and-artifacts) checked waiting/crossing and Cyclist-disagreement examples. This small sample does not establish population-wide association quality.



## Native action gallery (2026-10-05)

Generated ten examples for each of the nine native pedestrian actions on `aalto`, using the unchanged inspection renderer and index with seed 0. All 90 selected clip-scoped tracks are distinct. Selection requires the action somewhere in the track; it does not require complete 3D pairing or a single-action sequence. This is a browsing sample, not a measured association/semantic acceptance audit.

Local artifacts: `outputs/road_waymo/index.html` (all classes), per-class `index.html` pages, MP4s and action-frame previews. Players start at the first frame carrying the selected action and retain full-scene context. `samples.json` records identities, action start frames and paired coverage; class logs/run JSONs, `SHA256SUMS`, `validation.json` and the artifact README preserve reproduction and verification. Remote workspace: `/home/user20/road-waymo-gallery-20261005/`. Original datasets/index were read in place.

## Pedestrian population and sequence scale (2026-10-01)

The [official release README](https://github.com/salmank255/Road-waymo-dataset) reports **1,000 videos**, approximately **20 s** each, and 198k annotated frames. Its 54k agent tracks include all classes.

[Paper v1, Tables 11–12](https://arxiv.org/html/2411.01683v1) reports **11,759 pedestrian agent tubes**, including **2,186 test tubes**, and 867,407 pedestrian boxes. Subtraction gives **9,573 non-test tubes** (train + validation, not training alone). The table totals 52,362 agent tubes, whereas the headline says 54k; verify the acquired release rather than treating all published totals as identical.

| Pedestrian action | All action tubes | Test action tubes |
|---|---:|---:|
| Stop | 3,368 | 641 |
| Move away | 2,331 | 461 |
| Move towards | 2,162 | 403 |
| Move | 1,604 | 334 |
| Wait to cross | 516 | 106 |
| Crossing | 675 | 60 |
| Cross from right | 571 | 81 |
| Cross from left | 531 | 88 |

Action tubes are label-specific sequences; do not sum them as unique pedestrians or equate them to LOKI's distinct tracks per action. The **516 waiting tubes** are especially relevant to supervision scale. The acquired 9,573 train/validation pedestrian tracks match the paper's non-test total; current paired counts above do not include test supervision. The extent scan above measures gaps and whole-track lengths; action episodes remain uncharacterized.
