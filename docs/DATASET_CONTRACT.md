# Dataset input contract

Status: Provisional comparison; minimum contract not decided  
Last updated: 2026-09-25

## Purpose and ownership

- Contains: the **three-way intersection** of usable inputs across PedSynth++ (source), LOKI (real benchmark), and ECP2.0 (possible deployment target), plus the resulting contract decision gate.
- Links out: detailed evidence and release schemas to [dataset notes](DATASETS/DATASET_MATRIX.md); target labels to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md).

## Shared-input audit

“Reported” means a paper or official dataset page describes it, **not** that we have inspected the release. “Derivable” means it could be computed with the same method from reported data; quality is untested. ECP2.0 here means its **tracking extension**, not the separate detection, 2.5D, or dense-pose packages.

| Input | PedSynth++ | LOKI | ECP2.0 tracking | Contract implication |
|---|---|---|---|---|
| RGB frames | Reported; demo has `.png` | Reported; `image_*.png` | Reported images | Shared raw modality. Resolutions and frame rates differ. |
| 2D person boxes | Reported per frame | Reported with `track_id` | Reported dense 2D trajectories | Shared candidate. Check coordinate convention, visibility, and missing frames. |
| Track identity + time order | Multi-person clips; persistent ID/timestamps **unverified** | `track_id`; 5 Hz annotations | Unique ID over sequence; dense tracks | Required conceptually, but PedSynth++ export and timing must be checked before declaring a common field. |
| Metric pedestrian position | Export **unverified**; CARLA access alone is insufficient | 3D box position reported | World-fixed BEV position + height reported | **Not yet a confirmed three-way input.** Test release availability and coordinate conversion. |
| Ego motion/pose | Moving ego; export **unverified** | Odometry and ego-motion data reported | Ego-motion data reported | Conditional; needed if deriving comparable metric trajectories. |
| 2D body pose | Estimated COCO-17 reported; file coverage unverified | Derivable from RGB, not provided | Derivable from RGB, not established in tracking package | Optional derived modality; run the same estimator in all domains. |
| Raw LiDAR | Reported and in demo | Reported | Used for trajectory generation; public tracking-package availability unverified | Not a safe common requirement. |
| Road/map semantics | Simulator context; comparable export unverified | Lane/context labels and map cloud reported | Comparable road labels unverified | Optional derived evidence, not shared GT. |
| Behavior labels | FSM + crossing reported | Four pedestrian actions reported | No matching four-state GT reported | Supervision/evaluation only; **never model input**. |

Sources: [PedSynth++ paper](https://arxiv.org/pdf/2605.24950) and [demo](https://zenodo.org/records/20444839); [LOKI paper](https://arxiv.org/pdf/2108.08236) and [official format](https://usa.honda-ri.com/loki); [ECP2.0 paper](https://doi.org/10.1109/TPAMI.2024.3471170) and [current tracking access page](https://eurocity-tracking-dataset.tudelft.nl/). Each [dataset note](DATASETS/DATASET_MATRIX.md) records caveats.

## Contract decision gate

The **candidate minimum** is a pedestrian's ordered 2D track linked to RGB frames: dataset/sequence ID, track ID, frame time or reconstructible timing, and 2D boxes. This is a proposal, not an accepted contract, because PedSynth++ track IDs/timing and release-level alignment are still unverified. Pose can be derived from RGB; metric motion, ego pose, and road context remain optional until the same usable representation is confirmed across the three datasets.

Before accepting the contract, inspect each release for: stable IDs, time units, 2D box convention, RGB/box alignment, missing observations, and whether PedSynth++ exports 3D positions and ego pose. Derived speed/acceleration must specify coordinates and units rather than assume that 2D motion is metric. Keep behavior GT, CARLA FSM/route information, and other privileged simulator facts outside model inputs. The unresolved choice is tracked in [open question Q4](OPEN_QUESTIONS.md#q4-which-inputs-can-be-produced-comparably-in-all-three-datasets).
