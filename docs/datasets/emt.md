# EMT

Status: Background dataset; no current experiment selected

Last updated: 2026-10-01

## Why it matters to this project

EMT was considered as a cross-domain comparison under the earlier direction. Its similar action terminology remains useful background, but no EMT experiment is selected in the current ROAD-Waymo-to-LOKI plan. Preserve the facts below for deliberate later reconsideration.

## Available modalities

The paper reports over 30,000 dash-camera frames and tracking, trajectory forecasting, and intention-prediction benchmarks. LiDAR, 3D representation, ego motion, road semantics, and exact annotation fields are not yet verified for this project.

## Pedestrian tracking and behavior labels

Tracking is a stated benchmark. Reported categories are `Stopping`, `Walking`, `Waiting to cross`, and `Crossing`; confirm whether these are frame labels, prediction targets, or both.

## What could be input / ground truth

Observed trajectories are a potential input, and behavior/intention annotations a potential comparison target. Neither should be assumed compatible with LOKI without guideline and sample inspection.

## Possible role and portability concerns

A possible future cross-domain test would require new justification and an ontology/input audit. Its camera-centric representation may constrain direct reuse of 3D/geometry-dependent methods.

## Temporal structure / sampling frequency

Needs verification.

## Scene / road annotations

Needs verification.

## Quality / uncertainty metadata

Needs verification.

## Open questions

- Exact label definitions and temporal resolution.
- Released track identity, occlusion, quality, coordinate, and ego-pose fields.
- Access/license and the intended interpretation of `Stopping`.

## Sources

- Abdel Madjid et al., 2025: [arXiv](https://arxiv.org/abs/2502.19260), [project repository](https://github.com/AV-Lab/emt-dataset).

## Paper benchmark context

The [paper](https://arxiv.org/abs/2502.19260) reports **570,000 annotated boxes** alongside its >30,000 frames. It supplies task benchmarks rather than a reusable offline labeler. Exact fields/license, whether `Stopping` is a state or transition, and LOKI compatibility remain unverified.
