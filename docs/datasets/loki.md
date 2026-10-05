# LOKI — Long Term and Key Intentions

Status: Paper and official dataset page verified; local release layout and first action counts inspected
Last updated: 2026-10-02

## Role and scope

3D-first dataset in the first kinematic diagnostic, evaluated both within-dataset and in both transfer directions with ROAD-Waymo. LOKI supplies the tentative four-state output ontology; its initial visibility-comparable cohort and the eventual primary transfer direction remain open. LOKI-source training is separate from ROAD-Waymo → LOKI strict zero-shot, which excludes LOKI from training/model selection. Existing qualitative inspection is disclosed below; its final split treatment remains open. See [evaluation access rules](../../experiments/README.md).

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

Same-frame PLY/box plotting is supported by a [four-frame alignment sanity check](../archive/2026-09-28-loki-inspection.md#point-cloud-and-3d-box-alignment-check). Physical units, ego-forward direction, odometry/map transforms and RGB projection remain unverified.

A read-only full scan with `python -m pedestrian_behavior.inspection summary --root data/loki_data` found **40,829** frame suffixes and zero missing aligned image, 2D label, 3D label, point-cloud, or odometry files. Counts below are raw pedestrian rows in 3D labels, plus distinct `(scenario, track_id)` pairs per action; one track can contribute to multiple actions. They differ from the paper's published instance counts, and release/version or counting conventions remain unverified.

| Raw `intended_actions` | Frames | Tracks |
|---|---:|---:|
| `Crossing the road` | 65,225 | 1,967 |
| `Moving` | 242,740 | 9,528 |
| `Stopped` | 31,009 | 1,270 |
| `Waiting to cross` | 52,595 | 1,634 |

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

Per-track 3D observation counts: **minimum 1, lower quartile 11, median 23, upper quartile 43, maximum 102** (sorted empirical order statistics). At nominal 5 Hz, 23 consecutive observations span 4.4 s; actual gaps and timestamp units were not checked here. Track counts do not establish distinct physical people across scenarios, uninterrupted trajectories, or action episodes. Missing 2D boxes do not alone establish why RGB evidence is absent.

All observed 3D pedestrian rows have one of the four actions in the preceding table. Per-action track counts overlap and must not be added to obtain the 12,364 total. The 1,001 2D-only tracks have no behaviour label through this reader.

Reproduction: `conda run --no-capture-output -n pedestrian-behavior python outputs/inspection/population/loki_counts.py`; the one-off script and `loki_counts.json` remain under ignored `outputs/`. No images or point clouds were decoded. Release checksum and identity-fragmentation audit remain open.

## Prior qualitative inspection

[Seven selected tracks](../archive/2026-09-28-loki-inspection.md#selected-clip-observations-2026-09-28) show weak evidence and Stopped/Waiting ambiguity. They are informative examples, not representative statistics or GT corrections; declare their split treatment before evaluation.

## Access and remaining checks

Non-commercial use requires a request using a university email, per the [official page](https://usa.honda-ri.com/loki). The first inspection does not yet establish independent episode counts/durations, transitions, timing units, missing pedestrian labels, calibration, or viable scene-level splits.

The [population gate](README.md#outstanding-dataset-checks) must extend the population count with label/scene distributions, track gaps/durations, 2D+3D versus 3D-only/2D-only coverage, missing behaviour GT, and verified distance/point sparsity. Define RGB-visible and 3D-only cohorts at frame and/or track level before scoring; missing 2D boxes do not establish a single visibility cause. [E001](../../experiments/E001-kinematic-transfer/README.md#tentative-first-baseline-projection) owns the proposed comparison mapping.

## Inspection display

`summary` counts raw actions, tracks, and missing aligned files. `gallery` creates an HTML index and whole-scenario videos for tracks containing the exact action. Each 5 FPS video shows RGB beside a point-cloud BEV, centered on the selected pedestrian in a fixed 40-coordinate-unit view (+x up, +y right). Yellow outlines mark that pedestrian's available 2D and 3D boxes. Missing boxes are marked separately; the BEV center interpolates across missing 3D labels and holds the nearest known position at either end. A missing point cloud leaves a marked blank BEV. Change `--offset` and `--output` for the next batch; existing non-empty output directories are rejected.

Runnable commands are in the [root README](../../README.md#inspect-loki).

## Semantic limits

Moving does not imply road crossing. Stopped can include gestures, road occupancy or other activity. Crossing requires road relation and audited boundaries; velocity alone is insufficient. Waiting to cross may depend on future motion, orientation or partly latent intention, and may not be uniquely identifiable from available sensing. Future context can help without guaranteeing identifiability. Score original current-frame actions, without the paper's future-target shift.

## Paper method context

Girase et al., ICCV 2021, pp. 9803–9812, studies joint recurrent trajectory prediction/intention estimation with scene-graph reasoning and long-term goal proposals. The [paper](https://arxiv.org/abs/2108.08236) reports up to **27%** improvement over cited trajectory baselines; protocol reproduction/comparison remains unverified. It does not establish portable offline pseudo-labeling.
