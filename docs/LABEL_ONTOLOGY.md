# Label ontology

Status: Native semantic audit in progress; cross-dataset mappings unaccepted

Last updated: 2026-10-02

## Purpose and ownership

- Contains: native annotation meanings, target-state semantics, temporal granularity, possible shared concepts, and evidence needed for evaluation mappings.
- Links out: release fields/counts to [dataset notes](DATASETS/DATASET_MATRIX.md), audit work to [DATASET_INSPECTION_PLAN.md](DATASET_INSPECTION_PLAN.md), and experiments to [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## LOKI target actions

The [LOKI paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Girase_LOKI_Long_Term_and_Key_Intentions_for_Trajectory_Prediction_ICCV_2021_paper.pdf) reports four current-frame actions. Its prediction experiments shift an action four frames into the future; the offline target here is the original current-frame action. The [local note](DATASETS/LOKI.md) records raw `intended_actions` values and selected observations.

| Raw action | Candidate output name | Semantic uncertainty |
|---|---|---|
| Moving | `MOVING` | Movement alone does not assert road crossing. |
| Stopped | `STOPPED` | Stationarity can coexist with gestures, road occupancy, or another activity. |
| Waiting to cross | `WAITING_TO_CROSS` | Includes crossing-related context or inferred intention; may be poorly identifiable. |
| Crossing the road | `CROSSING` | Requires scene relation, not velocity alone; boundaries need audit. |

These names are candidate target output names, not a universal taxonomy for source datasets.

## Native annotation audit

This is the initial audit structure, not a completed ontology comparison. Read the linked notes for sources and verification status.

| Dataset | Annotation / semantic meaning | Temporal granularity | Available modalities | Possible shared concept | Uncertainties |
|---|---|---|---|---|---|
| [ROAD-Waymo](DATASETS/ROAD_WAYMO.md) | Observed Ped actions: Mov, MovAway, MovTow, PushObj, Stop, Wait2X, Xing, XingFmLft, XingFmRht; native location labels separate | Frame/box labels and tubes; exact gaps/boundaries to characterize | FRONT RGB/2D and partial official native 3D pairs; LiDAR/ego files referenced; map coverage unknown | Motion, scene relation, crossing behaviour | Definitions, overlap, timing, visual association acceptance and LOKI compatibility |
| [nuScenes](DATASETS/NUSCENES.md) | `pedestrian.moving`, `pedestrian.standing`, `pedestrian.sitting_lying_down`; separate scene/map annotations | Native box keyframes at 2 Hz; no dense behaviour GT assumed | RGB, LiDAR, 3D boxes, calibration, ego pose, scene expansions | Motion and 3D scene grounding | Attribute coverage, scene-target derivation, package alignment, and supervision feasibility |
| [ROAD](DATASETS/ROAD.md) | Native action/location annotations | Frame/box labels and tubes documented; release coverage to inspect | Video/2D tracks; no convenient 3D supervision assumed | Visual behaviour and scene relation | Exact pedestrian subset, semantic agreement with ROAD-Waymo, temporal coverage |
| [IDD-PeD](DATASETS/IDD_PED.md) | Separate crossing, interaction, activity, attention, social and stationary attributes | Frame-level attributes documented | Video/2D tracks and contextual annotations; 3D correspondence unverified | Visual behaviour/context | Multi-attribute semantics, applicability, annotation coverage, domain-selection differences |
| [LOKI](DATASETS/LOKI.md) | Four actions above, attached to 3D rows in inspected release | 5 Hz annotation; labelled coverage varies per track | RGB/2D when available, point clouds, 3D boxes, odometry/map files | Target behaviour evaluation | Stopped/waiting identifiability, boundaries, coordinates, visibility populations |

## Comparison policy

Do not equate nuScenes standing, ROAD stop, and LOKI Stopped solely because their names look similar. Waiting-to-cross semantics and crossing-direction variants also require definitions and timelines.

Preserve each source's native semantics. Dataset-specific supervision heads are a candidate way to share representations without forcing all annotations into LOKI's four states. Missing labels remain missing; neither a universal `OTHER` class nor forced mappings are accepted.

A zero-shot ROAD-Waymo-to-LOKI test requires a documented projection supported by definitions and examples, fixed before target performance is inspected. Specify conditional/excluded cases and evaluated population. If semantics do not support a projection, report that limitation rather than inventing compatible labels.

The candidate motion/scene/crossing factorization helps organize questions; it does not define accepted shared losses or latent axes. Optional derived segments, onset/end, and waiting duration also require explicit definitions.

## Superseded synthetic mapping

The former PedSynth++ mapping investigation is preserved in [research history](DATASETS/PED_SYNTH_PLUS_PLUS.md#research-history). It is not active training supervision. The unavailable rich implementation invalidates using its advertised states as the present foundation, without ruling out future synthetic augmentation.
