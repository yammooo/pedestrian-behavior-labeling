# Problem definition

Status: Accepted scope; provisional formulation  
Last updated: 2026-09-24

## Purpose and ownership

- Contains: annotation unit, permitted temporal context, task inputs and outputs, and exclusions.
- Links out: scientific motivation to [RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md), input fields to [DATASET_CONTRACT.md](DATASET_CONTRACT.md), and state meanings to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md).

## Primary task

Given a complete recorded pedestrian track and dataset-provided observations, assign a coherent per-frame observable behavior-state sequence. Future frames may be used. The labeler may later generate segments, transitions, and derived crossing events from that sequence.

The unit is a tracked pedestrian over a recorded temporal segment. A frame can have a current behavior label; complete tracks can contain several states. The intended target is a state sequence, while any crossing event or segment summary is derived from that sequence. The exact [ontology](LABEL_ONTOLOGY.md) remains provisional.

## Candidate inputs (not settled)

Track identity, timestamps, and observations that can be produced for the same pedestrian across source and target datasets. The minimum shared representation is unresolved; see the [dataset contract](DATASET_CONTRACT.md). Complete recorded context may be used because annotation is offline.

## Outputs

- Primary: a per-frame behavior-state sequence for each tracked pedestrian; class meanings remain provisional.
- Derived, if the accepted ontology supports them: behavior segments, state transitions, and crossing onset/end.
- Confidence and abstention, later if transfer and error analysis justify them.

## Non-goals for the initial phase

- Rebuilding detection, LiDAR uplift, or tracking for a target dataset.
- Treating a future-derived label as an online observable prediction.
- Treating CARLA FSM state, perfect simulator route, or future simulator intent as deployable model input.

## Critical distinction

| Task | Input at time `t` | Future frames allowed? | Output |
|---|---|---:|---|
| Offline annotation | Complete recording/track | Yes | Behavior segments or events |
| Online prediction | Observations up to `t` | No | Future behavior or latent intention |

Offline annotation is the accepted project scope; online prediction may be a later comparison, not the primary problem.
