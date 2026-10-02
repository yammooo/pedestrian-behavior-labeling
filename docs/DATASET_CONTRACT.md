# Dataset input contract

Status: Provisional heterogeneous contract; minimum requirements unresolved

Last updated: 2026-10-01

## Purpose and ownership

- Contains: identity, time, coordinate conventions, modality/annotation availability, and leakage rules.
- Links out: release evidence to [dataset notes](DATASETS/DATASET_MATRIX.md), label semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), and verification to [DATASET_INSPECTION_PLAN.md](DATASET_INSPECTION_PLAN.md).

## Contract direction

The former PedSynth++/LOKI/ECP2.0 three-way input intersection is superseded. The deployment goal assumes existing pedestrian 3D tracks, while a visual-only source may still supervise a suitable branch. A source need not expose every target modality. The minimum usable target input and exact shared representation remain open.

| Concept | Required clarification before an adapter is accepted |
|---|---|
| Dataset, sequence, pedestrian identity | Stable native IDs, identity continuity, and validated cross-release associations. Keep association provenance; a ROAD tube ID is not automatically a Waymo object ID. |
| Time and annotation cadence | Sensor timestamps, annotation times, frame indexing, gaps, and synchronization tolerance. Do not equate annotation cadence with capture rate. |
| 3D boxes and trajectory | Coordinate frame, origin, axes, units, dimensions, rotation convention, and ego/world transforms. Derive metric velocity only after verifying these. |
| RGB and 2D observations | Camera identity, box convention, calibration, crop/context policy, and per-observation availability. Missing boxes do not alone establish a particular visibility condition. |
| Point clouds | Sensor configuration, point fields, timing, accumulation, ego compensation, and pedestrian/local-scene correspondence. |
| Ego/map/scene context | Coordinate transforms, semantic definitions, coverage, and whether context is provided or derived. |
| Availability and quality | Distinguish missing modality, occlusion, sparse returns, missing annotation, and uncertain association. |
| Supervision | Preserve native labels, definitions, annotation coverage, and masks independently from model inputs. |

This table specifies inspection obligations, not a software schema or adapter framework.

## Current evidence

- [LOKI](DATASETS/LOKI.md): local identity joins and same-frame PLY/3D-box plotting have been checked. Physical units, ego-forward direction, odometry/map transforms, and RGB projection remain unverified.
- [ROAD-Waymo](DATASETS/ROAD_WAYMO.md): behaviour data has not been acquired locally; linkage to original Waymo 3D tracks is unproven.
- [nuScenes](DATASETS/NUSCENES.md): candidate 3D/scene source; its native keyframe annotations do not establish dense behaviour supervision.
- [ROAD](DATASETS/ROAD.md) and [IDD-PeD](DATASETS/IDD_PED.md): potential visual supervision; usable 3D correspondence is not assumed.
- PedSynth++ and ECP2.0 do not determine the current minimum contract.

## Missing modalities and labels

Represent actual availability explicitly. Apply a task loss only where its native annotation exists. Do not fabricate behaviour targets, treat an unlabeled frame as negative, or fill annotation gaps as ground truth. Interpolation used for a gallery camera center is not a recovered pedestrian observation.

A projected 3D box and an annotated 2D observation have different provenance. Any resampling or derived geometry must record its method and uncertainty. Exact missing-input behavior is chosen after the inspection gates.

## Leakage rules

Behaviour labels, future-derived prediction targets, and simulator-private route/intention fields stay outside inference inputs. Complete recorded observations are permitted because the task is offline.

Strict zero-shot uses no LOKI training or model selection, including unlabeled representation adaptation. Source/target semantic comparison is documented separately from training. Splits, inspected-scene handling, and low-shot access belong in [EVALUATION_PLAN.md](EVALUATION_PLAN.md).
