# Research direction

Status: Current direction; methods are working hypotheses

Last updated: 2026-10-02

## Purpose and ownership

- Contains: motivation, central question, hypotheses, and criteria for continuing or pivoting.
- Links out: task to [PROBLEM_DEFINITION.md](PROBLEM_DEFINITION.md), methods to [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md), and comparisons to [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## Current question

**How can an offline pedestrian-behaviour labeler transfer to a new 3D-first driving dataset with minimal target-specific supervision?**

The broad goal is automatic weak labeling of recorded datasets with existing pedestrian tracks. Complete past and future observations may be used. The intended artifact is a working offline research labeler supported by correct experimental methods, reproducible runs, and traceable results. Reliable automatic coverage and minimal human review are end goals; no numeric target has been accepted.

ZOD-IAC exposed the cost of jointly constructing reliable trajectories and labeling behaviour. This project begins after tracking; the [historical note](DATASETS/ZOD_IAC.md) retains that motivation. The synthetic route was subsequently demoted because the inspected PedSynth++ release and active generator did not provide the required rich supervision. [PedSynth++ history](DATASETS/PED_SYNTH_PLUS_PLUS.md#research-history) records the evidence and completed side work.

## Current investigation priority

1. Complete visual acceptance and reproducibility of the acquired ROAD-Waymo/Waymo index; structural joins and partial 3D coverage are now verified.
2. Establish native annotation meanings and characterize LOKI's pedestrian population.
3. Build simple target diagnostics and a ROAD-Waymo source baseline before testing extensions.
4. Test whether complementary real supervision improves transfer beyond that baseline.

ROAD-Waymo is the baseline source with [verified partial 3D correspondence](DATASETS/ROAD_WAYMO.md#acquired-index-and-access-2026-10-02); native semantics and training acceptance remain open. nuScenes or ROAD may be added next; neither inclusion nor order is selected. IDD-PeD is a later option. [Dataset roles](DATASETS/DATASET_MATRIX.md) distinguish these candidates from LOKI, the main target, and possible later applications.

LOKI is 3D-first: behaviour labels can accompany pedestrians without a 2D box. Source camera selection and target population differences may matter as much as geography or sensors. Strict zero-shot excludes LOKI from training and model selection and requires a defensible semantic comparison. Existing exploratory inspection must be disclosed; exact split safeguards belong in the evaluation plan.

## Working hypotheses

| Hypothesis | Comparison or evidence needed |
|---|---|
| H1: heterogeneous real sources improve generalization | ROAD-Waymo alone versus individually justified source additions. |
| H2: native supervision heads support a shared representation without forcing incompatible taxonomies | Test dataset-specific supervision after auditing semantics and missing labels. |
| H3: 3D scene/road understanding adds value beyond kinematics | Compare trajectory-only and trajectory-plus-scene representations. |
| H4: paired multimodal supervision helps when target RGB is unavailable | Compare matched methods on RGB-visible and 3D-only target populations. |
| H5: source pretraining reduces target annotation needs | Pretrained versus scratch models at exactly equal target-label budgets. |
| H6: source additions can cause negative transfer | Retain and report additions that hurt as well as those that help. |

These hypotheses do not select a factorization, encoder, fusion method, or training algorithm. Zero-shot is an important test when semantics permit it; project success does not require excellent zero-shot performance.

## Gates and boundaries

The [inspection plan](DATASET_INSPECTION_PLAN.md) owns the immediate gates. Failure of ROAD-Waymo linkage requires reconsidering source supervision before proceeding. Synthetic data remains optional future augmentation; no new CARLA FSM work is planned. ECP2.0 is a possible later application, not an internship dependency.

The [strategic-pivot decision](DECISIONS/0002-real-source-feasibility-first.md) records investigation priority only. [Open questions](OPEN_QUESTIONS.md) retains unresolved research choices; the [gap analysis](LITERATURE/GAP_ANALYSIS.md) owns the unverified novelty question.
