# Research direction

Status: Current direction; methods are working hypotheses

Last updated: 2026-10-02

## Purpose and ownership

- Contains: motivation, central question, hypotheses, and criteria for continuing or pivoting.
- Links out: task to [PROBLEM_DEFINITION.md](PROBLEM_DEFINITION.md), methods to [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md), and comparisons to [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## Current goal and provisional questions

The goal is offline pedestrian behaviour labeling from existing tracks, using whatever recorded sensing is available. LOKI's four states provide the tentative output ontology for the first experiment. The following questions come from the latest 2026-10-02 user handoff; they are working questions, not final thesis/paper claims.

**Provisional RQ1: Can behavior supervision available only for camera-visible pedestrians be transferred into a 3D-centric temporal representation that can label pedestrians without visual observations?**

ROAD-Waymo selects behaviour-labeled pedestrians through the front camera, while LOKI attaches behaviour to a broader 3D track population. The focus is transfer from camera-visible supervision to a representation usable without visual observations. ROAD-Waymo → LOKI also changes geography, sensors, scene structure, annotation conventions and class frequencies. These shifts are confounded; this RQ does not claim to isolate generic sensor or geographic generalization.

**Provisional RQ2: Does factorizing pedestrian behavior into transferable motion, scene-relation, and crossing representations improve transfer to the 3D-only setting?**

The candidate `z_motion / z_scene / z_crossing` factors could accommodate complementary supervision without equating native taxonomies. Motion alone may explain Moving/Stopped; Crossing requires road relation, and Waiting may also depend on future trajectory, orientation, context or partly latent intent. Whether these factors are necessary, separable or better than a shared temporal representation remains untested.

The primary transfer direction is not fixed. The first diagnostic compares both ROAD-Waymo → LOKI and LOKI → ROAD-Waymo, alongside within-dataset performance. The reverse direction may expose transfer asymmetry; it does not make either dataset a permanently designated source or target.

The broad goal is automatic weak labeling of recorded datasets with existing pedestrian tracks. Complete past and future observations may be used. The intended artifact is a working offline research labeler supported by correct experimental methods, reproducible runs, and traceable results. Reliable automatic coverage and minimal human review are end goals; no numeric target has been accepted.

ZOD-IAC exposed the cost of jointly constructing reliable trajectories and labeling behaviour. This project begins after tracking; the [historical note](DATASETS/ZOD_IAC.md) retains that motivation. The synthetic route was subsequently demoted because the inspected PedSynth++ release and active generator did not provide the required rich supervision. [PedSynth++ history](DATASETS/PED_SYNTH_PLUS_PLUS.md#research-history) records the evidence and completed side work.

## Current investigation priority

1. Complete visual acceptance and reproducibility of the acquired ROAD-Waymo/Waymo index; structural joins and partial 3D coverage are now verified.
2. Establish native annotation meanings and characterize LOKI's pedestrian population.
3. Define the minimal comparable kinematic representation and the first framewise MLP versus whole-track BiLSTM comparison in both datasets and both transfer directions.
4. Inspect those failures before adding scene information, RGB, factorization or heterogeneous supervision.

ROAD-Waymo has [verified partial 3D correspondence](DATASETS/ROAD_WAYMO.md#acquired-index-and-access-2026-10-02); native semantics and training acceptance remain open. It and LOKI form the initial two-dataset diagnostic. nuScenes, ROAD and IDD-PeD are possible later additions, with no selected inclusion or order. [Dataset roles](DATASETS/DATASET_MATRIX.md) distinguish these candidates and possible later applications.

LOKI is 3D-first: behaviour labels can accompany pedestrians without a 2D box. For each strict zero-shot direction, exclude that run's target from training and model selection. Existing exploratory inspection and any target-informed development must be disclosed; exact split safeguards belong in the evaluation plan. A clean initial visibility-comparable subset is still under discussion and would not itself test the full 3D-only problem.

## First diagnostic and later hypotheses

The [pipeline design](PIPELINE_DESIGN.md#first-diagnostic-baseline) specifies the proposed kinematics-only MLP/BiLSTM comparison. Its purpose is to measure the value of full-track temporal context and transfer in both directions before selecting a larger method. Stronger reverse transfer would be consistent with a visibility-selection hypothesis, but would not establish its cause. Similar failures in both directions could reflect ontology/domain mismatch; class-wise errors and within-dataset results are needed to interpret either outcome.

The earlier hypotheses remain possible follow-ups rather than requirements of the first baseline:

| Hypothesis | Comparison or evidence needed |
|---|---|
| H1: heterogeneous real sources improve transfer | A single-source baseline versus individually justified source additions. |
| H2: native supervision heads support a shared representation without forcing incompatible taxonomies | Test dataset-specific supervision after auditing semantics and missing labels. |
| H3: 3D scene/road understanding adds value beyond kinematics | Compare trajectory-only and trajectory-plus-scene representations. |
| H4: paired multimodal supervision helps when target RGB is unavailable | Compare matched methods on RGB-visible and 3D-only target populations. |
| H5: source pretraining reduces target annotation needs | Pretrained versus scratch models at exactly equal target-label budgets. |
| H6: source additions can cause negative transfer | Retain and report additions that hurt as well as those that help. |

These hypotheses do not select the final factorization, encoder, fusion method, or training algorithm. The small MLP/BiLSTM comparison is a diagnostic proposal, not the final contribution. Zero-shot is an important test when semantics permit it; project success does not require excellent zero-shot performance.

## Gates and boundaries

The [inspection plan](DATASET_INSPECTION_PLAN.md) owns the immediate gates. Failure of ROAD-Waymo linkage requires reconsidering source supervision before proceeding. Synthetic data remains optional future augmentation; no new CARLA FSM work is planned. ECP2.0 is a possible later application, not an internship dependency.

The [strategic-pivot decision](DECISIONS/0002-real-source-feasibility-first.md) records investigation priority only. [Open questions](OPEN_QUESTIONS.md) retains unresolved research choices; the [gap analysis](LITERATURE/GAP_ANALYSIS.md) owns the unverified novelty question.
