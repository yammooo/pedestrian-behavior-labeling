# LOKI — Long Term and Key Intentions

Status: Paper and official dataset page verified; release not inspected  
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

## Access and immediate checks

Non-commercial use requires a request using a university email, per the [official page](https://usa.honda-ri.com/loki). No LOKI release has been inspected in this repository. First count independent pedestrian tracks/episodes per action; confirm action field names, missing labels, 2D–3D–image alignment, odometry and calibration, and viable scene-level splits. See the [inspection plan](../DATASET_INSPECTION_PLAN.md) and [ontology](../LABEL_ONTOLOGY.md).
