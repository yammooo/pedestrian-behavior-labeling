# ROAD-Waymo

Status: Acquired on aalto; ROAD-authoritative same-frame 3D index inspected and rechecked

Last updated: 2026-10-02

## Candidate role

Baseline real behaviour source with verified partial multimodal correspondence in the acquired train/validation subset. Neither this linkage nor similar class names establishes LOKI compatibility. [Gate 1](../DATASET_INSPECTION_PLAN.md#gate-1--road-waymo--waymo-linkage) owns remaining acceptance checks.

## Documented annotation form

The [official repository](https://github.com/salmank255/Road-waymo-dataset) describes frontal videos with agent, action, location, frame-box and tube annotations. The acquired `road_waymo_trainval_v1.0.json` and exported vocabulary preserve `tube_uid`, native class-ID lists and complete ROAD box records. Box coordinates are normalized `[x1,y1,x2,y2]`; original ROAD geometry is retained even when it differs from Waymo's camera box.

Observed pedestrian action strings are `Mov`, `MovAway`, `MovTow`, `PushObj`, `Stop`, `Wait2X`, `Xing`, `XingFmLft`, and `XingFmRht`. Decode original IDs using `road_label_definitions.json` (the exporter uses `all_*_labels`). Action `Xing` and location `xing` are distinct. Definitions, overlap, boundaries and correspondence to LOKI remain to be audited; no cross-dataset mapping or loss is accepted. Frame-level AV actions remain in the original ROAD JSON and are not pedestrian targets.

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

Use `pedestrians.csv.gz` for annotations and `scene_manifest.json` for original modality paths. `report.json`, `export_audit.json`, `clip_summary.json`, and `road_label_definitions.json` contain provenance, coverage and native vocabularies. The manifest covers **798 distinct scenes/clips**: ROAD's 600 train and 198 validation clips, all drawn from Waymo **training**. ROAD's 202 test videos have no downloaded behaviour annotation JSON and contribute no rows. ROAD and Waymo split names are not interchangeable.

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

[Accepted policy](../DECISIONS/0003-road-authoritative-pedestrian-index.md): ROAD defines the pedestrian class. Filter/train using `merged_agent_label=Ped`, not `waymo_3d_type`. Apply `has_3d_box` as the validity mask for 3D supervision; a nonempty `official_laser_object_id` alone is insufficient. CSV booleans are strings: compare to `"True"`, not Python truthiness of the field.

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

The supplied independent export audit reports a full pass. On 2026-10-02 this repository inspection additionally rechecked all CSV rows, output hashes, annotation-key uniqueness, masks, geometry validity, track totals, duplicates, split/action counts and existence of all manifest paths. One paired observation was checked directly against original association and LiDAR-box Parquet rows. The remote merge synthetic self-check also passed. These checks support structural consistency, not a measured visual matching error rate; no new image/point-cloud visual audit was performed here.

Ignored inspection evidence: `outputs/inspection/road_waymo_handoff/` (remote document/script snapshots, `recheck.py`, `recheck.json`). Source index hashes in the remote report identify the inspected artifact; the CSV SHA256 is `1b1fa986ce901c803946e3b7f55ba8431fcd83fdc9de31aaacee9c6ab12b541f`.

Remaining work: document acquisition/release history, preserve runnable mapping/checker code in version control, characterize lengths/gaps/action episodes and physical-ID overlap, audit semantic disagreements and varied associations visually, and declare the training acceptance rule. The merger currently hardcodes Waymo `training` paths; this matches the inspected subset but must be changed before processing another split. No model or data adapter is implemented by this inspection.

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

Action tubes are label-specific sequences; do not sum them as unique pedestrians or equate them to LOKI's distinct tracks per action. The **516 waiting tubes** are especially relevant to supervision scale. The acquired 9,573 train/validation pedestrian tracks match the paper's non-test total; current paired counts above do not include test supervision. Paired lengths, gaps and action-episode/track counts still need characterization.
