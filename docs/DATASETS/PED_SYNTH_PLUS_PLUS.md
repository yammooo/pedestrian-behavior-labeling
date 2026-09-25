# PedSynth++ / ARCANE-PedSynth

Status: Paper and public demo description verified; full release not inspected  
Last updated: 2026-09-25

## Role and scale

Candidate synthetic pretraining source, **not yet accepted** as LOKI-compatible supervision. The [ARCANE-PedSynth paper](https://arxiv.org/pdf/2605.24950) reports 533 CARLA clips, 177,231 frames, 3,426 distinct pedestrians (2,143 crossing; 1,283 non-crossing), and 1,631,312 annotated pedestrian-frames. Clips last about 10–15 s at 30 FPS and contain roughly 5–10 pedestrians and 5–10 traffic vehicles with a moving ego car. Four maps are reported: Town01, Town02, Town03, Town05; 12 weather/lighting settings. These are paper-level statistics, not counts verified from released files.

## Sensors and timing

| Item | Paper specification | Distributed form reported |
|---|---|---|
| RGB | Ego-mounted at driver-eye level; 1280 × 720; 90° field of view; 30 FPS. | `.png` frames in the [12-clip demo](https://zenodo.org/records/20444839). |
| LiDAR | Roof-mounted; 64 channels; 100 m range; 1 million points/s; 20 Hz; 360° horizontal coverage. | `.bin` point clouds in the demo. Binary record layout not documented there. |
| DVS/event camera | Co-located with RGB; asynchronous brightness events; paper's pipeline diagram gives ±0.3 threshold in log mode. | `.npz` in the demo. Event array schema/resolution not verified. |
| Ego/3D pose and calibration | Moving ego is reported. | Exported ego poses, pedestrian metric tracks, intrinsics/extrinsics, and exact synchronization metadata **not confirmed**. Do not infer them solely from CARLA's internal availability. |

[Paper, §2.2.3 and Fig. 2](https://arxiv.org/pdf/2605.24950); [Zenodo demo record](https://zenodo.org/records/20444839). Raw DVS has no established LOKI counterpart and is not a first shared input.

## Labels and derived data

The paper reports per-visible-pedestrian, per-frame 2D boxes, FSM behavior state, binary crossing, distance to ego, time-to-crossing markers, and character archetype. Its crossing flag requires both a crossing-related FSM state **and** physical occupancy of a CARLA driving lane; it is not simply an FSM lookup. A visibility gate requires <70 m distance, camera-facing position, an in-frame box, and a minimum projected box size (15 × 30 px; 8 × 15 px beyond 50 m). Estimated COCO-17 2D pose is produced after rendering with AlphaPose using simulator-provided boxes; it is **not** perfect simulator skeleton ground truth. Whether the demo/full release includes those keypoints for every clip requires file inspection. [Paper, §2.2.3–2.2.4](https://arxiv.org/pdf/2605.24950).

The paper's 12-state list includes `RETREAT`. A prior inspection of the public generator reported `NORMAL_CROSSING` in its enum instead; retreat logic was present without an obvious corresponding enum value. **Neither paper list nor code enum proves the values in the released CSV.** Do not build a 12→4 mapping yet; see [label ontology](../LABEL_ONTOLOGY.md).

## Files and access

- The [public Zenodo package](https://zenodo.org/records/20444839) is a **12-clip demonstration subset**, one clip per listed weather/category condition, distributed as `PedSynth_for_paper_2026-05-29_v1.zip` (25.5 GB). Its record explicitly reports RGB `.png`, LiDAR `.bin`, DVS `.npz`, metadata/annotation files, and `ALL_WEATHER_combined_labels.csv`.
- The [paper's pipeline diagram](https://arxiv.org/pdf/2605.24950) describes labels in CSV/JSON and pose keypoints in JSON, with per-weather and global combined labels. This describes generation outputs, **not a verified archive tree or guaranteed contents of the demo**.
- The paper says the **full** 533-clip dataset is available from the corresponding author on reasonable request. The [generator repository](https://github.com/wielgosz-info/carla-pedestrians) is public, but its source code is not the dataset. Exact clip directory names, CSV columns, stable pedestrian IDs, timestamps, calibration, 3D trajectories, and keypoint files remain unverified.

## Immediate checks

List the demo archive before extraction, inspect CSV headers and unique values (`RETREAT` vs `NORMAL_CROSSING`), count independent tracks and state durations, then check temporal transitions and whether generic non-crossing `STOPPED` occurs. Verify metric motion and pose availability before making them shared inputs with LOKI. See the [inspection plan](../DATASET_INSPECTION_PLAN.md).
