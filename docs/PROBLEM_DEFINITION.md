# Problem definition

Status: Accepted offline scope; provisional formulation

Last updated: 2026-10-01

## Purpose and ownership

- Contains: annotation unit, temporal context, task inputs and outputs, and exclusions.
- Links out: motivation to [RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md), fields to [DATASET_CONTRACT.md](DATASET_CONTRACT.md), and semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md).

## Primary task

Given an existing pedestrian track over a recorded sequence, assign a dense per-frame behaviour-state sequence. The labeler may use the complete track, including future observations. The unit is a tracked pedestrian over a recorded temporal segment; it may contain several states.

The task is offline dataset annotation. It assumes existing tracks whose suitability must be checked, rather than reconstructing detections and associations. The intended research artifact is an operational offline labeler with reproducible experimental evidence.

## Inputs and missing evidence

Ideal target inputs include pedestrian identity and time, 3D boxes/trajectory, point clouds, ego pose, RGB and 2D boxes where available, and potentially scene/map information. RGB is not a universal requirement: useful labeling of 3D-only pedestrians is a central research hypothesis.

Source datasets may provide complementary subsets of these inputs and different annotations. Their usable fields, coordinates, and minimum requirements remain unresolved in the dataset contract. Neither absent RGB nor absent ground truth implies a behaviour class.

## Outputs and observability

- Primary: a per-frame state sequence using an explicitly documented output ontology.
- Derived, if meaningful: segments, transitions, and crossing onset/end.
- End goal: confidence or uncertainty supporting automatic acceptance and optional human review; method and coverage targets remain open.

Moving and stopping are largely observable. Crossing also depends on road relation. Waiting to cross can include inferred intention that limited observations cannot uniquely identify. Future context may help but cannot guarantee identifiability. Preserve uncertainty rather than describing every target label as a directly observable physical state.

## Task boundary

| Task | Temporal evidence | Output |
|---|---|---|
| Offline annotation | Complete recorded track; future frames allowed | Current-frame states; optionally derived segments/events |
| Online prediction | Observations up to the current time | Future actions or intention |

Offline annotation is [accepted scope](DECISIONS/0001-offline-annotation-primary-task.md). Initial work excludes rebuilding target detection/tracking, onboard prediction, and substantial model implementation before the feasibility and protocol gates. Synthetic route/FSM knowledge and annotation ground truth are supervision or history, not deployable inputs.
