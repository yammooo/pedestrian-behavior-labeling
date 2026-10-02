# LOKI — Long Term and Key Intentions

Status: Paper and official dataset page verified; local release layout and first action counts inspected
Last updated: 2026-10-01

## Role and scope

Main 3D-first target for strict zero-shot and low-shot evaluation, with separate scratch diagnostics. Strict zero-shot excludes LOKI from training and model selection; source/target semantic compatibility must be established first. Existing qualitative inspection is disclosed below; the treatment of inspected scenarios in the final split remains open. See [evaluation access rules](../EVALUATION_PLAN.md).

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

[Paper, Table 4 and §7](https://arxiv.org/pdf/2108.08236); [official format description](https://usa.honda-ri.com/loki).

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

### Point-cloud and 3D-box alignment check

The [official format description](https://usa.honda-ri.com/loki) says `label3d_*.txt` is annotated in point-cloud space. Local `pc_*.ply` files are binary little-endian PLY with vertex `x`, `y`, `z`, and intensity. In four matched frames (`scenario_000`: `0000`, `0050`; `scenario_001`: `0000`; `scenario_079`: `0138`), 70 of 73 nearby vehicle/pedestrian boxes contained at least one point when their raw center, dimensions, and yaw were applied directly to the same-frame PLY (`0.15` coordinate-unit tolerance). Mirroring `y`, swapping `x/y`, or shifting `x` by 10 units reduced this to 22, 10, and 20 boxes. For the 40 vehicle boxes in these frames, yaw-rotated boxes contained 11,460 points, versus 6,169 without rotation and 5,865 with yaw offset by 90°. Some boxes have few points, consistent with sparse or occluded returns; these counts are an alignment sanity check, not annotation-quality metrics.

This supports plotting **same-frame PLY `x/y` points and raw `label3d` boxes together in BEV**, with `z` as height and yaw applied in the `x/y` plane. No extra transform was needed in the checked frames. The ego-forward axis, physical units, odometry/map transforms, and RGB projection remain unverified; do not use this check as camera calibration evidence.

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

## Selected clip observations (2026-09-28)

These seven tracks were chosen as informative examples. The raw behavior value is in the 3D label row: it is present when the selected pedestrian has a 3D box but no 2D box, and absent in 2D-only frames. `Waiting to cross` often looked consistent with someone held back by traffic or signals, but the examples below show limits of the available view and a possible boundary with `Stopped`.

- `scenario_238 / daea6047-f97b-4e9f-9c54-cffaec34fe82`: Only frames `0000`–`0004` (three frames) have labels, all `Waiting to cross` with 3D boxes only. The person is outside the RGB field of view. The 3D box contains 85 and 190 points in the first and third frames, but none in `0002`, where a truck blocks the LiDAR view; the other point clusters are distinct from the truck but do not clearly look like a pedestrian in BEV. This is difficult to judge from these observations alone.
- `scenario_453 / ff3ec68b-63f7-4fe2-8e45-a6fe3fb80fef`: All 18 labeled frames are `Waiting to cross` with no 2D box; the person is outside the RGB field of view. In BEV the person appears still on a broad sidewalk across the opposing lane, without an apparent approach to the road. The reviewer would have called this `Stopped`, but the sidewalk and road layout is hard to determine from LiDAR alone.
- `scenario_461 / b4c4e4f7-3f23-426e-875c-7d9f713413af`: The label changes from `Waiting to cross` at frame `0028` to `Crossing the road` at `0030`, roughly when the person enters the road (possibly one frame off). Six later frames retain a 2D box but lack a 3D box and behavior label.
- `scenario_154 / 89e1b6ac-454d-40ab-b82b-566aa7471f3b`: Eleven frames are labeled `Stopped`. LiDAR returns are sparse and the person is distant, briefly discernible in RGB from behind near or within the road, apparently facing into a truck.
- `scenario_188 / 4a23f315-7436-4876-b222-203af44dab83`: A person seen from behind, apparently standing on a sidewalk, is labeled `Stopped` for 19 frames.
- `scenario_192 / f9a93b0d-563b-4d4d-b593-55d0d187e854`: A person is partly hidden by a metal barrier at the sidewalk–road edge and clearly visible in only a few frames; all 16 3D-labeled frames say `Stopped`.
- `scenario_197 / 0d770d93-315d-4f20-a573-921c475d6c6e`: A person apparently directing traffic in the road has planted feet but moves their arms. All 75 labeled frames say `Stopped`, consistent with the label allowing gestures while the person remains in place.

## Access and remaining checks

Non-commercial use requires a request using a university email, per the [official page](https://usa.honda-ri.com/loki). The first inspection does not yet establish independent episode counts/durations, transitions, timing units, missing pedestrian labels, calibration, or viable scene-level splits.

The [population gate](../DATASET_INSPECTION_PLAN.md#gate-3--loki-population) must extend the population count with label/scene distributions, track gaps/durations, 2D+3D versus 3D-only/2D-only coverage, missing behaviour GT, and verified distance/point sparsity. Define RGB-visible and 3D-only cohorts at frame and/or track level before scoring; missing 2D boxes do not establish a single visibility cause. The seven selected examples are informative observations, not representative population statistics or corrections to GT. [Ontology](../LABEL_ONTOLOGY.md) owns the remaining semantic comparison.
