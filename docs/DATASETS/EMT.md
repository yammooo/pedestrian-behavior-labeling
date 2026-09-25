# EMT

Status: Draft  
Last updated: 2026-09-22

## Why it matters to this project

EMT is a potential ontology-harmonization and cross-domain evaluation dataset: its reported pedestrian behavior categories overlap substantially with LOKI while its Arab Gulf driving domain differs from the likely LOKI domain.

## Available modalities

The paper reports over 30,000 dash-camera frames and tracking, trajectory forecasting, and intention-prediction benchmarks. LiDAR, 3D representation, ego motion, road semantics, and exact annotation fields are not yet verified for this project.

## Pedestrian tracking and behavior labels

Tracking is a stated benchmark. Reported categories are `Stopping`, `Walking`, `Waiting to cross`, and `Crossing`; confirm whether these are frame labels, prediction targets, or both.

## What could be input / ground truth

Observed trajectories are a potential input, and behavior/intention annotations a potential comparison target. Neither should be assumed compatible with LOKI without guideline and sample inspection.

## Possible role and portability concerns

Candidate cross-domain test; domain shift in road topology, traffic, clothing, and weather may be scientifically useful. Its camera-centric representation may constrain direct reuse of 3D/geometry-dependent methods.

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
