# Gap analysis

Status: Working comparison; no novelty claim

Last updated: 2026-10-02

## Purpose and ownership

- Contains: evidence-backed comparisons and limits of possible contribution claims.
- Links out: primary sources through [INDEX.md](INDEX.md), reviewed details to paper notes, and current hypotheses to [RESEARCH_DIRECTION.md](../RESEARCH_DIRECTION.md).

| Topic | Existing evidence / boundary | Current research implication | Review still required |
|---|---|---|---|
| Automatic crossing pseudo-labeling | [Riaz et al. 2025](paper_notes/minimizing_human_labeling.md) reports synthetic-to-real binary labeling and temporal smoothing | Automatic labels or smoothing alone are not a contribution | Compare target-data access, label budgets, temporal semantics, and independent audits |
| Native actions versus future intention | [LOKI](paper_notes/loki.md) transforms current actions for prediction | Use native frame actions; analyze inferred Waiting semantics and weak evidence | Annotation definitions and ROAD-Waymo projection |
| Rich synthetic supervision | [ARCANE-PedSynth](paper_notes/arcane_pedsynth.md) reports rich labels, while local findings limit current usability | Historical route, not evidence for the current foundation | Only revisit if suitable versioned supervision becomes available |
| Heterogeneous modalities and incomplete annotations | Current source candidates expose different observations/tasks; their suitability is conditional | Test useful transfer without fabricated labels | Closest multi-dataset, partial-supervision and missing-modality methods |
| Non-identical behaviour taxonomies | Native labels need not be equivalent across datasets | Dataset-specific heads are a hypothesis, not automatically a novel method | Existing multi-task/native-head approaches and precise semantic audit |
| Scene grounding beyond kinematics | LOKI inspection suggests road relation and accessibility can matter | Test scene value against trajectory-only baselines | Prior road-relative/3D scene representations and task-matched baselines |
| Camera-visible source → 3D-only target | ROAD-Waymo frontal annotation and LOKI 3D-first labels imply a population mismatch | Provisional RQ1; test both transfer directions but do not attribute asymmetry solely to selection, since domain/ontology shifts are confounded | Closest cross-modal transfer/distillation work and population-aware evaluation |
| Motion/scene/crossing factorization | Proposed semantic decomposition; no experimental evidence yet | Provisional RQ2; compare against a simpler shared representation only after the kinematic diagnostic | Closest factorized behaviour/scene representations and evidence that factors are identifiable |
| Zero-shot and low-shot annotation efficiency | Proposed comparisons require verified semantics and equal label access | Generalization and target-label savings must be measured | Comparable held-out protocols, label-cost accounting, and uncertainty in learning curves |
| Negative transfer and selective labeling | More sources may hurt; reliable coverage is an eventual objective | Retain negative ablations; add abstention only after error evidence | Prior negative-transfer diagnostics, calibration, and independent quality/coverage audit |

The potential contribution concerns offline behaviour supervision usable without visual observations and, if justified, factorized transfer under differing taxonomies. The first bidirectional kinematic diagnostic is intended to reveal failures; it is not itself the proposed final contribution. The exact RQs/contribution depend on results and a focused primary-source review. Shared heads, fusion, or a temporal architecture alone do not establish novelty.

The former rich-synthetic label-efficiency hypothesis remains in [PedSynth++ history](../DATASETS/PED_SYNTH_PLUS_PLUS.md#research-history). Do not carry its gap claim into the current study without new evidence.
