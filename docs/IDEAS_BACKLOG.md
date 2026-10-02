# Ideas backlog

Status: Uncommitted methods; promote only after evidence

Last updated: 2026-10-01

## Purpose and ownership

- Contains: optional techniques linked to a specific observed gap.
- Links out: open decisions to [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), active hypotheses to [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md), and comparisons to [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## Representation and temporal context

- Motion/scene/crossing factorization if native annotation analysis supports distinguishable factors.
- Pedestrian crop versus local visual context; derived pose only if error analysis identifies missing posture/activity evidence.
- Semantic BEV, road-relative features, barriers/accessibility, or scene encoders if kinematics cannot explain road-related states.
- Whole-track processing versus fixed chunks; recurrent, convolutional, Transformer, or ASFormer-like temporal encoders after a minimal baseline.

## Missing RGB and sensor shift

- Modality dropout, paired consistency, or teacher/student distillation if multimodal supervision demonstrably helps a 3D-only pathway. Complementary embeddings need not be identical.
- LiDAR ring/beam subsampling, angular/range-dependent sparsification, point dropout, FOV restrictions, noise, sweep variation, or intensity removal for a diagnosed sensor shift.
- RGB resolution, blur, crop, lighting/weather, and FOV perturbations where source shortcuts are observed.
- Canonical coordinates and verified road-relative quantities to reduce sensor-specific dependence.

## Multi-source conflict and reliability

- Gradient-similarity diagnostics if source ablations reveal negative transfer.
- PCGrad, GradNorm, CAGrad, or task/source adapters only if ordinary joint training exhibits a problem and simpler changes fail.
- Calibration, abstention, selective acceptance, and human review after errors and reliable evidence are characterized.
- Unlabeled target adaptation only as a separately specified future regime; it is excluded from current strict zero-shot.

## Optional future data

- Synthetic controlled scene/road supervision, sensor stress tests, or rare configurations if real-source experiments expose a gap. No new CARLA behaviour FSM is planned.
- ECP2.0 application if data availability and independent audit permit it; not required before 17 December.
