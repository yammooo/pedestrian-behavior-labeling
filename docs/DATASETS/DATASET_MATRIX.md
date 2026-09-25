# Dataset matrix

Status: Draft; verify against released data and annotation manuals before experimental use.  
Last updated: 2026-09-25

## Purpose and ownership

- Contains: a compact cross-dataset comparison and candidate experimental roles; update it when dataset notes change.
- Links out: detailed facts, sources, access conditions, and unresolved release fields to each dataset note. Use `unknown` where data has not been checked.

Legend: **reported** = stated in project handoff or cited paper; **unknown** = not yet verified; **N/A** = not the dataset's intended provision.

| Dataset | RGB | LiDAR | 2D boxes / IDs | 3D representation | Ego motion | Behavior labels | Reported ontology | Quality metadata | Candidate project role |
|---|---|---|---|---|---|---|---|---|---|
| PedSynth++ / ARCANE-PedSynth | Yes, paper-reported | Yes, paper-reported | Multi-pedestrian clips and dense annotations reported; exact fields need inspection | Needs release inspection | Moving ego vehicle reported; exact export needs inspection | Per-frame FSM states and binary crossing labels, paper-reported | Paper lists 12 states including `RETREAT`; post-checkpoint code inspection found `NORMAL_CROSSING` instead; CSV values unknown | Needs release inspection | Candidate synthetic pretraining source; suitability gate pending |
| LOKI | One RGB camera, paper-reported | Four LiDARs, paper-reported | Same `track_id` links 2D/3D boxes; released fields need inspection | 3D position, dimensions, yaw in official format description | CAN-bus compensation and odometry files reported | Yes, paper-reported frame-wise actions | Moving, waiting to cross, crossing road, stopped | Unknown | Main real benchmark for label-efficiency study; independent track counts pending |
| EMT | Dash-camera RGB reported | Unknown | Tracking benchmark reported; exact released fields need verification | Unknown | Unknown | Intention benchmark reported | Stopping, walking, waiting to cross, crossing reported | Unknown | Ontology comparison and cross-domain candidate |
| ECP2.0 | Yes | Auxiliary LiDAR used for uplift | Dense 2D tracks / IDs reported | World-fixed 3D position plus height; not dense full cuboids as primary representation | Yes, vehicle inertial sensing / ego compensation reported | No equivalent rich behavior labels reported | N/A | `unsure` frames and `weak` tracks reported | Deployment/enrichment and transfer candidate |
| PIE | Video reported | Unknown | Pedestrian track annotations reported | Unknown | Unknown | Crossing-related annotations reported | Needs verification | Needs verification | Literature/possible additional test |
| JAAD | Video reported | Unknown | Pedestrian annotations reported | Unknown | Unknown | Crossing-related annotations reported | Needs verification | Needs verification | Literature/possible additional test |
| ZOD / ZOD-IAC | ZOD RGB reported | ZOD LiDAR reported | ZOD-IAC tracks generated from detections | Derived 3D positions reported | Reported | Geometry/human-reviewed crossing records in prior project | ZOD-IAC-specific cross/no-cross and onset | GOLD/SILVER and review provenance in old project | Historical reference; possible future target |

## Modality and context audit

| Dataset | Radar | 3D boxes | Body orientation / pose | Road/map or scene semantics | Geography / diversity | Public availability / access |
|---|---|---|---|---|---|---|
| PedSynth++ / ARCANE-PedSynth | DVS reported; radar unknown | Needs release inspection | Estimated 2D pose keypoints reported | CARLA towns/scenarios; exact fields need inspection | 533 clips, 12 weather conditions, 4 CARLA towns reported | Generator code is open; full data by author request per paper; Zenodo demo subset |
| LOKI | Not reported | 3D boxes with position, dimensions, yaw in official format description | Pose not reported; derive from RGB if needed | Lane/context labels and `map.ply` reported; release details need inspection | Tokyo urban/suburban scenarios; varied time/weather | Non-commercial; university-email request |
| EMT | Unknown | Unknown | Unknown | Unknown | Arab Gulf region reported; weather/clothing variation reported | Public repository reported; terms need verification |
| ECP2.0 | Unknown | Not primary dense representation | Unknown | Unknown | 29 cities, 11 European countries; time/weather/season diversity reported | Non-commercial research access reported |
| PIE | Unknown | Unknown | Unknown | Unknown | Needs verification | Needs verification |
| JAAD | Unknown | Unknown | Unknown | Unknown | Needs verification | Needs verification |
| ZOD / ZOD-IAC | ZOD-specific; needs verification | Not part of the new contract | Unknown | `ego_road` polygon in historical pipeline | ZOD-specific | Needs verification |

## Candidate experimental role audit

| Dataset | Training/development | Supervised validation | Cross-domain test | Deployment/enrichment | Behavior ground truth? |
|---|---|---|---|---|---|
| PedSynth++ / ARCANE-PedSynth | Candidate source pretraining | Held-out source diagnostic | Source side for `S→R` | No | Yes, paper-reported; actual values unknown |
| LOKI | Scratch/few-shot/full-supervision reference | Main held-out real test | Target for `S→R` | Not primary current target | Yes, reported |
| EMT | Not central | Only after audit | Possible external real test | No | Reported; semantics unverified |
| ECP2.0 | No behavior GT reported | Independent manual audit only, if pursued | Possible later target | Candidate later enrichment | No equivalent rich behavior GT reported |
| PIE | Not selected | Not selected | Possible later | No | Crossing-related, needs verification |
| JAAD | Not selected | Not selected | Possible later | No | Crossing-related, needs verification |
| ZOD / ZOD-IAC | No | Historical subset only | Possible future target | Possible future target | Prior project-specific labels |

## Details still to verify

| Dataset | Temporal frequency | Road/map / scene semantics | Pose/orientation | Availability / license | Primary source |
|---|---|---|---|---|---|
| PedSynth++ / ARCANE-PedSynth | Clips about 10–15 s at 30 FPS reported; verify actual release timestamps | CARLA scenario context reported; exact annotations need inspection | Estimated 2D pose keypoints reported | Full data upon author request per paper; demo subset on Zenodo | [ARCANE-PedSynth paper](https://arxiv.org/pdf/2605.24950) |
| LOKI | 5 Hz annotation; camera captures 30 Hz, LiDAR spins at 10 Hz; verify release timing | Lane/context information reported; exact released annotations need inspection | Pose not reported; 3D box yaw in official format description | Non-commercial; university-email request | [LOKI paper](https://arxiv.org/pdf/2108.08236), [official dataset page](https://usa.honda-ri.com/loki) |
| EMT | Needs verification | Needs verification | Needs verification | Public repository reported; terms need verification | [EMT paper](https://arxiv.org/abs/2502.19260) |
| ECP2.0 | Needs verification | Needs verification | Needs verification | Non-commercial research access reported | [ECP2.0 paper](https://doi.org/10.1109/TPAMI.2024.3471170) |
| PIE / JAAD | Needs verification | Needs verification | Needs verification | Needs verification | See literature notes |
| ZOD / ZOD-IAC | ZOD-specific; needs reinspection only if reused | ZOD-specific | Unknown | ZOD access/terms need verification | Historical internal/public repository context |

This matrix deliberately does not turn an absent verification into a “yes.”
