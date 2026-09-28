# LOKI — Long Term and Key Intentions

Status: Paper and official dataset page verified; local release layout and first action counts inspected
Last updated: 2026-09-25

## Role and scope

Main candidate real-world benchmark for zero-/few-shot evaluation and a fully supervised diagnostic reference. Recorded from an instrumented Honda SHUTTLE in the Tokyo area: 644 scenarios (mean 12.6 s), more than 28,000 agents across eight traffic classes, and 886,000 linked 2D/3D boxes across all classes. These are **not pedestrian-track counts**. [Paper, §3](https://arxiv.org/pdf/2108.08236); [official dataset page](https://usa.honda-ri.com/loki).

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

## Access and remaining checks

Non-commercial use requires a request using a university email, per the [official page](https://usa.honda-ri.com/loki). The first inspection does not yet establish independent episode counts/durations, transitions, timing units, missing pedestrian labels, calibration, or viable scene-level splits. See the [inspection plan](../DATASET_INSPECTION_PLAN.md) and [ontology](../LABEL_ONTOLOGY.md).
