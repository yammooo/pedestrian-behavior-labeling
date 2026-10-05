# PedSynth++ / ARCANE-PedSynth

Status: Historical source investigation; inspected local release lacks required rich supervision

Last updated: 2026-10-01

## Current role and evidence

Synthetic data is optional future augmentation, not the foundational training source. The 2026-10-01 user checkpoint reports generator inspection, developer correspondence, generated-clip review, and CARLA reproduction. Those reports are distinguished below from local release checks and paper claims.

The [paper](https://arxiv.org/pdf/2605.24950) reports 533 CARLA clips, 177,231 frames, 3,426 pedestrians, 12 weather settings and a rich behaviour FSM. These remain paper statistics, not verified behavioural coverage in the local release.

## Paper and demo descriptions

| Item | Paper specification | Demo record description; archive not revalidated here |
|---|---|---|
| RGB | 1280 × 720, 90° FOV, 30 FPS | PNG frames |
| LiDAR | 64 channels, 100 m range, 1 million points/s, 20 Hz, 360° coverage | BIN files; record layout unverified |
| DVS | Co-located event camera | NPZ files; array schema unverified |
| Pose | AlphaPose-derived COCO-17 estimates | Coverage/package agreement unverified |
| Ego/3D geometry | Moving ego and simulator geometry | Does not establish released pedestrian trajectories or ego-pose files |

Sources: [paper](https://arxiv.org/pdf/2605.24950), [12-clip Zenodo demo description](https://zenodo.org/records/20444839). The demo is a separate described package; its stated modalities must not be attributed automatically to the local/Hugging Face release.

The paper describes dense behaviour/crossing fields and a visibility gate, with crossing requiring crossing-related state and driving-lane occupancy. Table 4 includes `RETREAT`; the inspected generator enum instead includes `NORMAL_CROSSING`. Paper names and declared enums are not evidence that those states are reached or exported.

## Observed local release

The local `data/pedsynth-plusplus` copy contains weather/town/timestamp clips with PNG frames, MP4, CSV/JSON labels, and sensor metadata. Release revision/checksum and exact Hugging Face source URL remain unknown.

A read-only check on 2026-10-01 found:

- 550 sensor metadata files: RGB enabled in all; LiDAR and DVS disabled in all.
- 533 per-clip CSVs with the same header: `video_id, frame_id, pedestrian_id, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, crossing, crossing_point, behavior_type, distance_to_ego, visible`.
- No BIN, NPZ or PLY files at the inspected clip-file level. The metadata/CSV count difference needs provenance clarification; it is not proof of 550 fully annotated clips.

The sample `clear_noon/Town01/20251213-174508` has `behavior_type=normal` in inspected rows. A category field is not proof of the paper's framewise FSM labels. The current CSV header contains no 3D pedestrian geometry or ego-pose fields. These release observations do not establish the contents of every other distribution.

## Interpretation and remaining checks

The inspected active generator lacks the required rich supervision; completed CARLA geometry exports do not resolve that gap. The [synthetic investigation](../archive/2026-10-01-synthetic-supervision.md) preserves state reachability, commits, correspondence caveats, export work and the unaccepted mapping. Synthetic road/scene supervision, rare configurations or sensor stress tests may return only for an identified real-source gap. No new CARLA FSM work is planned.

Release revision/checksum, exact Hugging Face URL, metadata/CSV count discrepancy and package differences remain unknown. The paper's author-request distribution, Zenodo demo and local copy are distinct.

## Paper method context

Riaz, Wielgosz, López Peña, [ARCANE-PedSynth (2026)](https://arxiv.org/abs/2605.24950), describes a hybrid AI/manual CARLA controller for crossing-prediction data and a 12-state behavior FSM. Estimated AlphaPose keypoints come from rendered RGB, not perfect simulator skeleton ground truth. The demonstrations establish neither PedSynth++→LOKI compatibility nor real-world multi-state zero-shot transfer. Author-request distribution and the demo are distinct from the local copy.
