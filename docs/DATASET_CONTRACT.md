# Dataset input contract

Status: Provisional heterogeneous contract; minimum requirements unresolved

Last updated: 2026-10-02

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
- [ROAD-Waymo](DATASETS/ROAD_WAYMO.md): acquired on `aalto`; official same-frame associations and native geometry are preserved in the inspected merged index. ROAD defines the pedestrian population; `has_3d_box` masks 3D supervision. Vehicle-frame boxes require ego compensation before deriving world-motion features.
- [nuScenes](DATASETS/NUSCENES.md): candidate 3D/scene source; its native keyframe annotations do not establish dense behaviour supervision.
- [ROAD](DATASETS/ROAD.md) and [IDD-PeD](DATASETS/IDD_PED.md): potential visual supervision; usable 3D correspondence is not assumed.
- PedSynth++ and ECP2.0 do not determine the current minimum contract.

## Missing modalities and labels

Represent actual availability explicitly. Apply a task loss only where its native annotation exists. Do not fabricate behaviour targets, treat an unlabeled frame as negative, or fill annotation gaps as ground truth. Interpolation used for a gallery camera center is not a recovered pedestrian observation.

For the acquired ROAD-Waymo index, retain complete ROAD sequences and their missing-3D masks. Parse CSV boolean strings explicitly, preserve original Waymo types/disagreements, and define repeated-observation handling before producing one sample per track/time. ROAD train/val both come from Waymo training; preserve both split fields. Native and camera-synchronized boxes are distinct geometry sources.

A projected 3D box and an annotated 2D observation have different provenance. Any resampling or derived geometry must record its method and uncertainty. Exact missing-input behavior is chosen after the inspection gates.

## First-baseline temporal representation

The latest handoff proposes whole variable-length pedestrian tracks on a common regular temporal grid, possibly 5 Hz. Preserve internal same-identity gaps and elapsed time; a list containing only valid detections is insufficient. Track start/end rules, grid alignment, resampling tolerances and GT assignment at grid times remain open. Resampling must not invent a behaviour label or conceal missing evidence.

Preserve separate metadata such as `has_3d`, `has_2d`, `has_rgb` and `has_behavior`, plus provenance and quality. An available RGB image does not imply a usable pedestrian crop or 2D observation. ROAD-Waymo's `has_3d_box` remains the native paired-box mask; applicable four-state supervision also depends on the accepted semantic mapping.

The proposed learned missing embedding belongs after the pedestrian encoder, while valid ego data remains available to its branch. This does not discard availability masks in the data. Missing input, missing label and sequence padding must remain distinguishable. Define partial-feature validity, missing ego handling and a minimum usable observation criterion before implementation; avoid a large short-track threshold that removes the hard population.

Candidate position/velocity features require verified units, axes, timing and transforms in both datasets. An ego-centric feature frame does not justify interpreting ego-relative displacement as pedestrian world velocity. Specify velocity semantics, derivative support near gaps/endpoints, and normalization using only the permitted training data. Yaw/dimensions/acceleration are optional later ablations.

For ROAD-Waymo, the initial eligible population is behaviour-labeled ROAD tracks with successful official 3D pairs; the required paired coverage and track start/end remain open. Preserve internal unpaired timesteps as gaps within accepted tracks. A visibility-comparable initial LOKI subset is under discussion, not selected. Retain all population metadata so later 2D-visible versus 3D-only evaluation remains possible.

## Leakage rules

Behaviour labels, future-derived prediction targets, and simulator-private route/intention fields stay outside inference inputs. Complete recorded observations are permitted because the task is offline.

For each strict zero-shot direction, its target dataset is excluded from training and model selection, including unlabeled representation adaptation. LOKI supervision is permitted in the separately declared LOKI-source and LOKI within-dataset runs. Source/target semantic comparison is documented separately from training. Splits, inspected-scene handling, and low-shot access belong in [EVALUATION_PLAN.md](EVALUATION_PLAN.md).
