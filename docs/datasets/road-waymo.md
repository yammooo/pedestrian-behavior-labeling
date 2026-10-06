# ROAD-Waymo

Status: Acquired on aalto; ROAD-authoritative same-frame 3D index inspected and rechecked

Last updated: 2026-10-05

## Candidate role

Initial real behaviour dataset with verified partial multimodal correspondence in the acquired train/validation subset. The proposed kinematic diagnostic uses it as both source and target in separate comparisons with LOKI. Neither linkage nor similar class names establishes LOKI compatibility; the tentative four-state projection requires manual audit. [Gate 1](README.md#outstanding-dataset-checks) owns remaining acceptance checks.

## Documented annotation form

The [official repository](https://github.com/salmank255/Road-waymo-dataset) describes frontal videos with agent, action, location, frame-box and tube annotations. The acquired `road_waymo_trainval_v1.0.json` and exported vocabulary preserve `tube_uid`, native class-ID lists and complete ROAD box records. Box coordinates are normalized `[x1,y1,x2,y2]`; original ROAD geometry is retained even when it differs from Waymo's camera box.

Observed pedestrian action strings are `Mov`, `MovAway`, `MovTow`, `PushObj`, `Stop`, `Wait2X`, `Xing`, `XingFmLft`, and `XingFmRht`. Decode original IDs using `road_label_definitions.json` (the exporter uses `all_*_labels`). Action `Xing` and location `xing` are distinct. Definitions, overlap, boundaries and correspondence to LOKI remain to be audited; no cross-dataset mapping or loss is accepted. Frame-level AV actions remain in the original ROAD JSON and are not pedestrian targets.

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

[Accepted policy](../archive/2026-10-02-road-waymo-linkage.md): ROAD defines the pedestrian class. Filter/train using `merged_agent_label=Ped`, not `waymo_3d_type`. Apply `has_3d_box` as the validity mask for 3D supervision; a nonempty `official_laser_object_id` alone is insufficient. CSV booleans are strings: compare to `"True"`, not Python truthiness of the field.

Preserve `road_annotation_json`, original label IDs, native `waymo_3d_type`/`waymo_3d_label`, `semantic_disagreement` and all original geometry. **419 paired rows across 15 camera tracks in 10 clips** have Waymo camera type Pedestrian and associated LiDAR type Cyclist. ROAD's class takes precedence, but these boxes are not resized to person-only geometry. Whether each disagreement reflects a labeling convention or an association error needs visual review.

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

There are **712,630 distinct frame/object observations**, of which **426,491** are paired. Ten extra annotation rows repeat an observation; the recheck found identical normalized boxes and action/location labels for those repeats. Keep raw rows intact; define duplicate handling explicitly when producing one sample per track/time. Annotation-row identity is `(road_clip_id, road_frame_1based, road_annotation_id)`; pedestrian identity is clip-scoped `road_tube_uid`, not the annotation ID or an unscoped LiDAR ID.

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

The manifest references camera images, native LiDAR, calibration, vehicle/LiDAR poses, native and camera-synchronized boxes and other components in place. File existence does not establish per-frame content or annotation coverage. Sampled `lidar` Parquet stores range-image returns, not PLY vertices; point-cloud visualization needs decoding and calibration/pose handling. Native and camera-synchronized boxes have distinct timing/geometry provenance. FRONT behaviour supervision does not label every pedestrian visible to other cameras or LiDAR. HD map coverage is still unverified.

The [full structural recheck](../archive/2026-10-02-road-waymo-linkage.md#structural-recheck) supports export consistency, not a measured visual matching error rate.



Remaining work: document acquisition/release history, preserve runnable mapping/checker code in version control, characterize action episodes and physical-ID overlap, investigate timing irregularities below, audit semantic disagreements and varied associations visually, and declare the training acceptance rule. The merger currently hardcodes Waymo `training` paths; this matches the inspected subset but must be changed before processing another split. No model or data adapter is implemented by this inspection.

## Track extents and gaps (2026-10-05)

Full annotation-only index scan, with identity `(road_clip_id, road_tube_uid)`: context spans the first to last ROAD pedestrian observation, including empty intermediate frames. Ten identical repeated observations are counted once, reconciling to 712,630 observations and 426,491 paired observations. Extent durations use exact verified Waymo endpoint timestamps, independently of gallery playback.

| Population | Tracks | Mean s | Median s | P90 s | P95 s | P99 s | Max s |
|---|---:|---:|---:|---:|---:|---:|---:|
| All ROAD pedestrians | 9,573 | 7.99 | 6.20 | 19.10 | 19.70 | 19.80 | 19.803223 |
| At least one paired 3D observation, full ROAD extent | 6,540 | 8.18 | 6.50 | 19.10 | 19.70 | 19.80 | 19.803223 |

3,709 tracks (38.74%) have internal empty ROAD frames, accounting for 7.92% of all first-to-last frame positions. The longest empty run has 181 native frame positions; its adjacent observed boundaries are 18.199996 s apart. Paired 3D is present at 55.11% of full-span positions across all tracks, or 78.84% among tracks with any 3D. These denominators include empty ROAD frames, unlike paired coverage over labeled rows. No missing observation or behavior target was filled.

Median timestamp interval per ROAD index step is 0.099993 s. However, 110 observed intervals exceed 0.15 s per index step (maximum 0.400036 s); causes remain unverified. Do not assume every ROAD index step is exactly 0.1 s or blindly take alternate frames to produce 5 Hz. Conservative capacity `ceil(duration × 5) + 1` has median **33**, P95 **100**, maximum **101** positions. Small timestamp deviations can add a capacity position; actual resampling/grid/tolerance are unset.

Whole tracks containing `Wait2X` have median 13.70 s, longer than the overall median; action cohorts overlap and are not action-episode durations. Durations are bounded by downloaded train/validation clips; no test supervision, additional native LiDAR-only tracks or ID stitching is included.

Local report: `outputs/inspection/track-statistics/README.md`; `road-waymo/tracks.csv` records every pedestrian, and `road-waymo/summary.json` includes quantiles, split/cohort/action window coverage, masks, source CSV checksum and timing outliers. Remote results: `/home/user20/road-waymo-gallery-20261005/track-statistics-road-waymo-final/`. [Reproduction commands](../../README.md#inspect-track-lengths-and-gaps). Full context at 5 Hz fits the inspected population in 101 positions under the stated rule; no architecture/window policy is selected.

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
