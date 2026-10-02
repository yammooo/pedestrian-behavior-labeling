# Dataset inspection plan

Status: Immediate feasibility and protocol gates

Last updated: 2026-10-01

## Purpose and ownership

- Contains: checks and required evidence before source selection, shared losses, or model implementation.
- Links out: findings to [dataset notes](DATASETS/DATASET_MATRIX.md), native semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), field requirements to [DATASET_CONTRACT.md](DATASET_CONTRACT.md), and protocol to [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## Record provenance first

For every inspected asset, record source URL, acquisition date, release/version or commit, checksum where available, files, schema, timing, units, IDs, and missing/duplicate rows. Distinguish paper descriptions, official format documentation, local observations, and checkpoint-reported evidence. Keep raw data and generated media outside Git.

ROAD-Waymo has not been acquired and no linkage implementation exists in this workspace. The new gates must not inherit a claim that its behaviour annotations are already 3D supervision.

## Gate 1 — ROAD-Waymo ↔ Waymo linkage

| Check | Evidence required |
|---|---|
| Clip ↔ original segment | Reproducible segment mapping, release compatibility, and unmatched/ambiguous cases. |
| Frame alignment | Camera identity, timestamps/frame indices, resampling and offset checks; filenames alone are insufficient. |
| ROAD object ↔ Waymo object | Match tubes/2D observations to native object identities; record ambiguity, occlusion, fragmentation, and confidence. |
| Waymo object ↔ 3D track | Verify camera-to-LiDAR associations and track continuity; do not assume camera and 3D IDs coincide. |
| Manual validation | Visual checks on varied tracks/frames, including crowded, small, occluded, and failed associations; record selection and observed mismatches. |
| Population/coverage | Matched, unmatched, ambiguous, and excluded counts; determine which behaviour-labeled pedestrians actually obtain usable 3D observations. |

Start with a small matched subset before broader processing. The manual sample size, acceptable matching quality, and acceptance rule are open and must be declared before calling the gate passed. Preserve mismatch examples and the mapping evidence.

If robust linkage is not supported, reconsider the source-training plan before building the proposed multimodal method. A visual ROAD-Waymo release alone does not pass this gate.

## Gate 2 — native annotation ontology

Audit ROAD-Waymo, nuScenes, ROAD, IDD-PeD, and LOKI using the table in [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md). Extract exact native names/IDs, pedestrian applicability, definitions, temporal cadence, missingness, multi-label behavior, annotation population, and available modalities. Inspect representative labeled timelines and boundary cases.

Verify ROAD-Waymo/LOKI semantic compatibility before any zero-shot projection. Keep source-native supervision distinct from a target evaluation mapping. Do not design shared losses or equate standing/stopping/waiting labels until their definitions are clear.

## Gate 3 — LOKI population

Measure distinct pedestrian tracks overall and by action, action episodes/transitions, track lengths/gaps, and class/scene coverage. Count per-frame and per-track 2D+3D, 3D-only, and 2D-only observations, including missing behaviour labels.

Define RGB-visible versus 3D-only cohorts with evidence rather than treating all missing boxes as outside FOV. Characterize distance and LiDAR sparsity only after verifying coordinates and point-association conventions. Review hard Stopped/Waiting pairs and the weak evidence cases already recorded in the dataset note.

Resolve release provenance, timing, ego compensation, map/context usability, and split-safe grouping. Identify previously inspected scenarios so their evaluation treatment can be declared.

## Gate 4 — baseline readiness and protocol freeze

Define usable trajectory features and scene inputs, if feasible, before introducing complex models. Specify the LOKI scratch diagnostics, ROAD-Waymo source baseline, and conditional zero-shot comparison. The evaluation plan owns split/access rules, metrics, and budget accounting.

Before a run, settle its applicable metric definitions, label-budget unit, sampling/seed policy, association acceptance rule, missing-GT handling, and model-selection access. Keep future source additions open rather than freezing an untested architecture.

## Outputs

Update the relevant dataset note once with measured evidence; link it from the matrix and ontology. Record unresolved questions and gate outcomes, then add a short log pointer. Use [experiment records](EXPERIMENTS/README.md) for reproducible feasibility runs and later comparisons.

Write small native inspection/association utilities only after inspecting sample files. This documentation update does not implement the model, acquire large datasets, or pass these gates.
