# LOKI — Long Term and Key Intentions

Status: Paper and official dataset page verified; local release layout and first action counts inspected
Last updated: 2026-10-06

## Role and scope

3D-first dataset in [E001](../../experiments/E001-kinematic-transfer/README.md), evaluated within-dataset and in both transfer directions with ROAD-Waymo. E001 accepts an experiment-specific four-state projection and includes eligible 3D-only tracks. Neither direction is permanently primary. LOKI-source training is separate from ROAD-Waymo → LOKI strict zero-shot, which excludes LOKI from training/model selection. See [evaluation access rules](../../experiments/README.md).

Recorded from an instrumented Honda SHUTTLE in the Tokyo area: 644 scenarios (mean 12.6 s), more than 28,000 agents across eight traffic classes, and 886,000 linked 2D/3D boxes across all classes. These are **not pedestrian-track counts**. [Paper, §3](https://arxiv.org/pdf/2108.08236); [official dataset page](https://usa.honda-ri.com/loki).

## Sensors and timing

| Item | Documented specification |
|---|---|
| RGB | One color SEKONIX SF332X-10X camera; 1928 × 1280; 60° field of view; 30 Hz capture. |
| LiDAR | Four Velodyne VLP-32C scanners; 32 beams each; 10 Hz spin; 200 m stated range; 40° vertical field of view. Their scans are ego-motion-compensated and merged into a 360° point cloud. |
| Ego sensing | MTi-G-710-GNSS/INS-2A8G4 (gyroscope, accelerometer, GPS); vehicle CAN bus used for ego-motion compensation. Scenario files include odometry poses. |
| Published annotation cadence | Synchronized RGB/LiDAR downsampled to **5 Hz** for annotation. Do not confuse this with the 30 Hz camera capture or 10 Hz LiDAR spin. |

The paper describes camera intrinsics and LiDAR-to-camera extrinsics used for calibration; whether calibration files are shipped, and their names/formats, still need checking. [Paper, §3 and §7](https://arxiv.org/pdf/2108.08236); [official dataset page](https://usa.honda-ri.com/loki).

## Labels and track data

- Same `track_id` links agents' camera-space 2D boxes and point-cloud-space 3D boxes. The published format also lists 3D position, box dimensions and yaw, and 2D attributes. Check identity continuity and field coverage in the downloaded release.
- The paper's pedestrian **current-frame action** labels are `Stopped` (32,538 annotated instances), `Moving` (241,889), `Waiting to cross` (49,576), and `Crossing the road` (64,870). These are paper table counts, not independent tracks or episodes. The paper describes them, respectively, as stopped along the street, walking, waiting to cross the intersection, and crossing the road. It later shifts actions by four 5 Hz frames (0.8 s) to construct *future intention* targets; we need the original actions.
- Additional annotations include pedestrian age/gender and potential destination; lane information, road entrance/exit, traffic controls, weather and road condition are reported. Their exact released schemas and completeness need checking. Pose keypoints are **not** listed as a provided modality; they would have to be derived from RGB.

[ICCV paper PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Girase_LOKI_Long_Term_and_Key_Intentions_for_Trajectory_Prediction_ICCV_2021_paper.pdf); [Paper, Table 4 and §7](https://arxiv.org/pdf/2108.08236); [official format description](https://usa.honda-ri.com/loki).

## Published on-disk structure

The [official dataset page](https://usa.honda-ri.com/loki) gives this scenario-level layout (abridged):

```text
scenario_xxx/
├── odometry/    odom_*.txt       # x, y, z, roll, pitch, yaw
├── label/       label2d_*.json   # class, track_id, bbox, not_in_lidar, attributes, potential_destination
│                label3d_*.txt    # class, track_id, stationary, position, dimensions, yaw, other agent fields
├── pointcloud/  pc_*.ply         # merged 360° cloud
├── image/       image_*.png      # front camera
└── map/         map.ply          # map point cloud
```

The page documents these names and fields, not a complete parser schema. Verify timestamps, filename alignment, coordinate frames, units, calibration delivery, and where the four pedestrian actions appear in actual files.

## Observed local release (2026-09-25)

The local `data/loki_data` copy has 644 flat `scenario_*` directories. In the inspected release, `image_*.png`, `label2d_*.json`, `label3d_*.txt`, `pc_*.ply`, and `odom_*.txt` sit directly inside each scenario, not in the subdirectories illustrated above. Numeric filename suffixes align these five types. The sample image `scenario_000/image_0000.png` is **1920 × 1208**, differing from the paper's sensor specification. Release version/checksum is unknown.

`label2d_*.json` has class keys; `Pedestrian` maps track IDs to records with `box` (`left`, `top`, `width`, `height`), `not_in_lidar`, and `attributes`. `label3d_*.txt` is comma-delimited with `labels`, `track_id`, 3D position/dimensions/yaw and `intended_actions`; pedestrian actions occur in `intended_actions`. These files can be joined on `(scenario, frame suffix, track_id)`. Some pedestrian records have only one side of the 2D/3D join; the reader retains them.

Same-frame PLY/box plotting is supported by a [four-frame alignment sanity check](../archive/2026-09-28-loki-inspection.md#point-cloud-and-3d-box-alignment-check). The preparation transform is now supported by the [full-release marker check](#verified-preparation-transform-2026-10-07); independent sensor accuracy and RGB projection remain unverified.

A read-only full scan with `python -m pedestrian_behavior.inspection summary --root data/loki_data` found **40,829** frame suffixes and zero missing aligned image, 2D label, 3D label, point-cloud, or odometry files. Counts below are raw pedestrian rows in 3D labels, plus distinct `(scenario, track_id)` pairs per action; one track can contribute to multiple actions. They differ from the paper's published instance counts, and release/version or counting conventions remain unverified.

| Raw `intended_actions` | Frames | Tracks |
|---|---:|---:|
| `Crossing the road` | 65,225 | 1,967 |
| `Moving` | 242,740 | 9,528 |
| `Stopped` | 31,009 | 1,270 |
| `Waiting to cross` | 52,595 | 1,634 |

### Track locators and native payload

The [shared schema](README.md#reader-and-prepared-track-schema) uses `{"scenario":"scenario_000","track_id":"<native-id>"}` as a canonical JSON track locator. Selected frame keys retain suffixes such as `"0000"`; the native reader resolves the flat filename layout and same-frame track-ID join. Original boxes remain in the source.

Prepared `native_annotations` preserves 2D attributes (including age/gender and destination references), `not_in_lidar`, original 3D `intended_actions`, `potential_destination` and `stationary`, plus original 3D record indices. These are supervision/metadata, never automatic input features. Destination UUIDs remain references; this increment does not resolve marker geometry. Literal `None`, empty strings, absent fields and JSON null stay distinct. Identical 3D duplicates retain record indices; conflicting repeats and duplicate JSON keys fail.

### Verified preparation transform (2026-10-07)

[Paper §7](https://arxiv.org/html/2108.08236#S7) states that the motion-compensated merged cloud is transformed to a virtual position at the vehicle center. The [official format](https://usa.honda-ri.com/loki) supplies odometry xyz/roll/pitch/yaw. [Full-release verification](../../scripts/verify-loki-transform.py) compares six interpretations on scene-scoped stationary road/destination markers with at least five observations.

For **2,482 road-marker identities**, full `Rz(yaw) Ry(pitch) Rx(roll)` rotation plus odometry translation gives median 3D world RMS **2.05e-14 m** (p90 **1.14e-13 m**). Raw coordinates give **11.464 m**, yaw-only **0.474 m**, inverse rotation **2.453 m**, y reflection **1.599 m** and xy swap **16.118 m**. This empirically verifies the released annotation/odometry composition, including roll/pitch/z; it does not establish independent sensor accuracy. Destination markers are less strict oracles: 2,664 identities give full-transform p90 **0.263 m**.

The evidence-backed [transform contract](../../configs/loki-transform.json) uses identity from the released virtual point-cloud frame to ego, metre/radian units and the full composition above. The native reader requires this explicit verified contract. The median ego heading/motion cosine is **0.999433** over **33,826** adjacent intervals with displacement >2 cm, supporting +x forward; +y left follows the right-handed pose interpretation; RGB projection/calibration accuracy remains unverified. Evidence: ignored `outputs/experiments/E001/reader-preparation/loki-transform-evidence.json`. Physical timestamps remain **unknown**; derivatives use the declared nominal clock.

### Unique pedestrian population (2026-10-01)

A fresh annotation-only scan of the local copy, using the existing reader and `(scenario, track_id)` identity, found:

| Quantity | Count |
|---|---:|
| Scenarios with any 3D pedestrian | 616 of 644 |
| Pedestrian tracks with 2D and/or 3D observations | 13,365 |
| Tracks with at least one 3D observation and action label | **12,364** |
| Tracks with 2D observations only | 1,001 |
| 3D tracks with no 2D box anywhere in the scenario | **4,139 (33.5%)** |
| 3D tracks with a 2D box at some point | 8,225 |
| Tracks with at least one same-frame 2D+3D pair | 8,203 |
| Tracks with at least 10 3D observations | 9,854 |
| Pedestrian 3D observation/action rows | 391,569 |

Per-track 3D observation counts: **minimum 1, lower quartile 11, median 23, upper quartile 43, maximum 102** (sorted empirical order statistics). At nominal 5 Hz, 23 consecutive observations span 4.4 s; the subsequent extent scan below checks annotation gaps. Physical timestamp units remain unverified. Track counts do not establish distinct physical people across scenarios, uninterrupted trajectories, or action episodes. Missing 2D boxes do not alone establish why RGB evidence is absent.

All observed 3D pedestrian rows have one of the four actions in the preceding table. Per-action track counts overlap and must not be added to obtain the 12,364 total. The 1,001 2D-only tracks have no behaviour label through this reader.

Reproduction: `uv run --locked --extra cpu python outputs/inspection/population/loki_counts.py`; the one-off script and `loki_counts.json` remain under ignored `outputs/`. No images or point clouds were decoded. Release checksum and identity-fragmentation audit remain open.

### Prepared inventory (2026-10-07)

The complete reader/preparation audit retains **13,365 candidates** across all **644 scenarios**, including the **1,001** 2D-only tracks. It saves **418,676** 5 Hz slots, with no missing scene-frame selections. All **391,569** native pedestrian 3D rows remain usable positions; **391,399** slots have usable velocity (170 singleton 3D observations have no adjacent support). Every candidate has an initial ego anchor. Native union observations total **413,923**; **196,620** have 2D annotations. No duplicate 3D pedestrian observations or preparation failures were found. These are preparation counts, before E001 target projection/splits.

Evidence: ignored `outputs/experiments/E001/reader-preparation/loki/{manifest,audit}.json`; the [eight native cases](../../experiments/E001-kinematic-transfer/reader-cases.json) are preserved in the portable inspector. Source annotations and physical-clock qualification are unchanged.

## Track extents and gaps (2026-10-05)

Annotation-only scan of all 644 scenarios: identity remains `(scenario, track_id)`, and context spans the first to last **2D or 3D** pedestrian observation, including empty frames in between. Every within-scenario filename increment is two (40,185 increments); `suffix / 2` is the ordinal at the documented nominal 5 Hz. Physical timestamps remain unverified. One-frame tracks span zero elapsed seconds but need one context position.

| Population | Tracks | Mean s | Median s | P90 s | P95 s | P99 s | Max s |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 2D-or-3D tracks | 13,365 | 6.07 | 4.4 | 15.8 | 19.0 | 20.0 | 20.2 |
| At least one 3D observation, full union extent | 12,364 | 6.33 | 4.6 | 16.4 | 19.2 | 20.0 | 20.2 |

702 union tracks (5.25%) have internal empty frames; these account for 1.14% of all first-to-last frame positions. The longest empty run is 65 frames (13.0 s at nominal cadence; adjacent observation boundaries are 13.2 s apart). All 3D annotation sequences are contiguous between their first and last 3D row; this does not establish uninterrupted sensor visibility or independently observed rather than source-interpolated annotations. Full union extents can include 2D-only extensions/gaps. No missing observation or GT was filled.

Conservative 5 Hz capacity `ceil(duration × 5) + 1` has median **23**, P95 **96**, maximum **102** positions across all tracks. Whole tracks containing crossing have median 11.0 s; waiting tracks 7.4 s. These overlapping action cohorts measure whole-track context, not action episodes. Durations are clip bounded; IDs are not stitched.

Local report: `outputs/inspection/track-statistics/README.md`; `loki/tracks.csv` contains every pedestrian, and `loki/summary.json` contains quantiles, cohort/action window coverage, masks and source annotation digest. The scan reconciles the population and 391,569 3D rows above. [Reproduction commands](../../README.md#inspect-track-lengths-and-gaps). These are native statistics preceding E001's agreed grid/eligibility; they do not settle identity fragmentation, behavior episodes or physical timing.

## Identity and short-track audit (2026-10-06)

The read-only protocol scan found **13,365 distinct native ID strings**, with no string reused across scenarios. Continue using `(scenario, track_id)`; string uniqueness does not prove different physical people or absence of fragmentation. The **12,364** tracks with native 3D/action GT occupy **403,813** first-to-last 2D/3D-union positions at nominal 5 Hz, including gaps and 2D extensions. This count is context positions, not 3D observations or accepted training frames.

Of these tracks, **170** have one native 3D/behavior observation and **1,300** have at most five. Short tracks are retained by E001 if they pass its [post-resampling eligibility](../../experiments/E001-kinematic-transfer/README.md#samples-and-eligibility); single positions cannot yield a velocity under the declared derivative rule. Final eligibility and pose-valid counts are **TBD**. These scans used local native annotations and the existing extent report; a versioned audit and complete release provenance remain pending.

## Prior qualitative inspection

[Seven selected tracks](../archive/2026-09-28-loki-inspection.md#selected-clip-observations-2026-09-28) show weak evidence and Stopped/Waiting ambiguity. They are informative examples, not representative statistics or GT corrections.

## Access and remaining checks

Non-commercial use requires a request using a university email, per the [official page](https://usa.honda-ri.com/loki). The first inspection does not yet establish independent episode counts/durations, transitions, timing units, missing pedestrian labels, calibration, or viable scene-level splits.

The [population gate](README.md#outstanding-dataset-checks) must verify native transforms/timing and final E001 label/scene distributions, eligibility, availability strata and split independence. Native union lengths/gaps and 2D/3D populations are measured above; post-grid usable-position counts remain **TBD**. Verify distance/point sparsity before interpreting them. Missing 2D boxes do not establish a single visibility cause. [E001](../../experiments/E001-kinematic-transfer/README.md#accepted-four-state-projection) owns the accepted comparison mapping and observation-condition definitions.

## Inspection display

`summary` counts raw actions, tracks, and missing aligned files. `gallery` creates an HTML index and whole-scenario videos for tracks containing the exact action. Each 5 FPS video shows RGB beside a point-cloud BEV, centered on the selected pedestrian in a fixed 40-coordinate-unit view (+x up, +y right). Yellow outlines mark that pedestrian's available 2D and 3D boxes. Missing boxes are marked separately; the BEV center interpolates across missing 3D labels and holds the nearest known position at either end. A missing point cloud leaves a marked blank BEV. Change `--offset` and `--output` for the next batch; existing non-empty output directories are rejected.

Runnable commands are in the [root README](../../README.md#inspect-loki).

## Semantic limits

Moving does not imply road crossing. Stopped can include gestures, road occupancy or other activity. Crossing requires road relation and audited boundaries; velocity alone is insufficient. Waiting to cross may depend on future motion, orientation or partly latent intention, and may not be uniquely identifiable from available sensing. Future context can help without guaranteeing identifiability. Score original current-frame actions, without the paper's future-target shift.

## Paper method context

Girase et al., ICCV 2021, pp. 9803–9812, studies joint recurrent trajectory prediction/intention estimation with scene-graph reasoning and long-term goal proposals. The [paper](https://arxiv.org/abs/2108.08236) reports up to **27%** improvement over cited trajectory baselines; protocol reproduction/comparison remains unverified. It does not establish portable offline pseudo-labeling.
