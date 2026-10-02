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

## Generator findings and completed side work

The local sibling `carla-pedestrians/PED_SYNTH_PAPER_CODE_GAPS.md` records inspection of the baseline scenarios commit `2e47ab3`. The October checkpoint reports this active-path distinction:

| Reachability | States |
|---|---|
| Assigned in normal generation | WALKING_SIDEWALK, CROSSING_ROAD, FINISHED_CROSSING |
| Helper methods outside active run path | SUDDEN_CROSSING, JAYWALKING, DISTRACTED_BEHAVIOR |
| Declared but not entered | LOOKING_AROUND, CHECKING_TRAFFIC, HESITATING, RUNNING_ACROSS, PAUSING_MID_CROSS, NORMAL_CROSSING |

Checking/hesitation fields were placeholders. Exporting an enum or the three active states would not supply the advertised rich behaviour implementation. This is evidence about the inspected path/version, not every possible private implementation. The [public generator](https://github.com/wielgosz-info/carla-pedestrians) is distinct from downloaded data.

The checkpoint reports successful CARLA 0.9.13 reproduction on the Aalto RTX 4080 machine and an upstream PR opened for structured pedestrian 3D/ego exports. Local code corroborates the export implementation: scenarios commit `4ac8888` adds `pedestrians_3d.csv`, `ego_pose.csv`, and LiDAR mount metadata; parent checkout commit `ac9bfb2` pins it. PR URL and current upstream status are unknown. Do not imply merging or fully validated sensor alignment.

The local gap note also records a baseline smoke clip with raw sensors, missing rich FSM export, synchronization concerns, and duplicated visible-label rows. Geometry export remains useful infrastructure history; it does not fix behavioural semantics or make the downloaded release multimodal.

## Research history

The original hypothesis was that rich synthetic framewise behaviour supervision could reduce real labels needed for LOKI adaptation. It was provisional and never an accepted training mapping.

| Former raw-state mapping candidate | Former interpretation / unresolved risk |
|---|---|
| WALKING_SIDEWALK → MOVING | Needed released motion verification |
| CHECKING_TRAFFIC, HESITATING → WAITING_TO_CROSS | Needed comparable onset/episode semantics; absent from inspected active path |
| LOOKING_AROUND → conditional/exclude | Could accompany moving, stopping, or waiting |
| CROSSING_ROAD, JAYWALKING, RUNNING_ACROSS → CROSSING | Only while road relation supports crossing |
| SUDDEN_CROSSING → conditional MOVING/CROSSING | FSM entry might precede road entry |
| PAUSING_MID_CROSS → CROSSING | Active crossing can have zero speed |
| DISTRACTED_BEHAVIOR → conditional/exclude | Distraction is orthogonal to movement |
| FINISHED_CROSSING → conditional MOVING/STOPPED | Depended on later motion |
| RETREAT (paper), NORMAL_CROSSING (enum) → unresolved | Released presence and semantics unverified |
| No obvious state → generic STOPPED gap | Non-crossing stationarity coverage was unclear |

The former evaluation idea compared native rich, binary crossing, and collapsed-state pretraining against scratch at fixed LOKI track budgets, with zero-shot as a diagnostic. It depended on source labels and common inputs that were not established. Git retains the full former plans.

The route was demoted because the inspected release lacked expected 3D/sensor exports and the active generator lacked the needed rich states. The checkpoint also reports reviewed generated clips with close/ahead spawning, early clustered crossing interactions and few useful later interactions; these qualitative examples were not a population frequency study. Developer discussion reportedly involved crossing-label flicker and temporal smoothing; correspondence dates and independent validation are unavailable.

Naveed's later inability to access the promised richer code is checkpoint-reported correspondence. The operational assumption is that implementation is unavailable; the current project must not depend on its arrival. This does not establish that synthetic pretraining is permanently impossible.

The [pivot record](../DECISIONS/0002-real-source-feasibility-first.md) owns the strategic priority. Controlled synthetic road/scene supervision, rare configurations, or sensor stress testing may return only if real-source experiments justify them; no new CARLA FSM is planned.
