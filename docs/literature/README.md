# Literature and contribution comparison

Updated 2026-10-05. Working comparison; no novelty claim. Dataset notes own native facts and release findings. Add a separate method review only when its detail changes a hypothesis, baseline, protocol or interpretation.

## Reference index

| Reference | Finding and project use |
|---|---|
| [Minimizing Human Labeling — Riaz, Wielgosz, López Peña, T-ITS 2025](minimizing-human-labeling.md) | Substantive review: S2R-UDA-CP binary crossing self-labeling, smoothing, target-access accounting and independent PedGraph+ validation |
| [LOKI — Girase et al., ICCV 2021, pp. 9803–9812](https://arxiv.org/abs/2108.08236) | Recurrent trajectory/intention prediction with scene graphs and long-term goals; current actions differ from future targets. [Native evidence](../datasets/loki.md) |
| [ARCANE-PedSynth — Riaz, Wielgosz, López Peña, 2026](https://arxiv.org/abs/2605.24950) | Hybrid AI/manual CARLA controller and advertised rich FSM; claims differ from active generator/export findings. [Native evidence](../datasets/pedsynth-plusplus.md); [investigation](../archive/2026-10-01-synthetic-supervision.md) |
| [Stop and Go Forecasting — Guo, Mordan, Alahi, 2022](https://arxiv.org/abs/2203.02489) | Online future stop/go on TRANS assembled from existing data; hybrid pedestrian/video/scene fusion. Reports a benchmark gain (exact metrics unreviewed). Motivates transition errors and temporal context, not adoption of its architecture; offline transition constraints remain open |
| Rasouli & Kotseruba taxonomy (exact bibliographic record unverified) | Conceptual distinction among intention estimation, action prediction and risk assessment; no dataset/method adopted. Prevents future-derived labels/hindsight being called current observable behavior; primary terminology needs advisor alignment |
| [PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/) and [JAAD](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) — Rasouli et al. (exact citations unverified) | Video crossing/intention and joint-attention/context references, commonly prediction-oriented. [Native evidence and release unknowns](../datasets/pie-jaad.md); not selected sources |
| [EMT — Abdel Madjid et al., 2025](https://arxiv.org/abs/2502.19260) | Gulf-region visual tracking/forecasting/intention benchmark; [native evidence](../datasets/emt.md), no current experiment |
| [ECP2.0 — Krebs, Braun, Gavrila, TPAMI 2024, 46(12), 10929–10943](https://doi.org/10.1109/TPAMI.2024.3471170) | Offline track pseudo-GT construction, separate from behavior enrichment; [native evidence](../datasets/ecp2.md), possible later application |
| [ROAD-Waymo repository](https://github.com/salmank255/Road-waymo-dataset), [Waymo labeling specifications](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md) | [ROAD-Waymo note](../datasets/road-waymo.md): frontal population, partial 3D linkage; semantic/visual acceptance open |
| [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md), [map tutorial](https://www.nuscenes.org/tutorials/map_expansion_tutorial.html) | [nuScenes note](../datasets/nuscenes.md): candidate limited motion/scene supervision |
| [ROAD paper](https://doi.org/10.1109/TPAMI.2022.3150906), [repository](https://github.com/gurkirt/road-dataset) | [ROAD note](../datasets/road.md): possible visual action/location source |
| [IDD-PeD project](https://cvit.iiit.ac.in/research/projects/cvit-projects/iddped), [repository](https://github.com/Ruthvik9/IDD-PeD) | [IDD-PeD note](../datasets/idd-ped.md): later optional visual/context source |

## Contribution boundaries and review gaps

| Topic | Existing evidence / boundary | Current research implication | Review still required |
|---|---|---|---|
| Automatic crossing pseudo-labeling | [Riaz et al. 2025](minimizing-human-labeling.md) reports synthetic-to-real binary labeling and temporal smoothing | Automatic labels or smoothing alone are not a contribution | Compare target-data access, label budgets, temporal semantics, and independent audits |
| Native actions versus future intention | [LOKI](../datasets/loki.md) transforms current actions for prediction | Use native frame actions; analyze inferred Waiting semantics and weak evidence | Annotation definitions and ROAD-Waymo projection |
| Rich synthetic supervision | [ARCANE-PedSynth](../datasets/pedsynth-plusplus.md) reports rich labels, while local findings limit current usability | Historical route, not evidence for the current foundation | Only revisit if suitable versioned supervision becomes available |
| Heterogeneous modalities and incomplete annotations | Current source candidates expose different observations/tasks; their suitability is conditional | Test useful transfer without fabricated labels | Closest multi-dataset, partial-supervision and missing-modality methods |
| Non-identical behaviour taxonomies | Native labels need not be equivalent across datasets | Dataset-specific heads are a hypothesis, not automatically a novel method | Existing multi-task/native-head approaches and precise semantic audit |
| Scene grounding beyond kinematics | LOKI inspection suggests road relation and accessibility can matter | Test scene value against trajectory-only baselines | Prior road-relative/3D scene representations and task-matched baselines |
| Camera-visible source → 3D-only target | ROAD-Waymo frontal annotation and LOKI 3D-first labels imply a population mismatch | Provisional RQ1; test both transfer directions but do not attribute asymmetry solely to selection, since domain/ontology shifts are confounded | Closest cross-modal transfer/distillation work and population-aware evaluation |
| Motion/scene/crossing factorization | Proposed semantic decomposition; no experimental evidence yet | Provisional RQ2; compare against a simpler shared representation only after the kinematic diagnostic | Closest factorized behaviour/scene representations and evidence that factors are identifiable |
| Zero-shot and low-shot annotation efficiency | Proposed comparisons require verified semantics and equal label access | Generalization and target-label savings must be measured | Comparable held-out protocols, label-cost accounting, and uncertainty in learning curves |
| Negative transfer and selective labeling | More sources may hurt; reliable coverage is an eventual objective | Retain negative ablations; add abstention only after error evidence | Prior negative-transfer diagnostics, calibration, and independent quality/coverage audit |

The potential contribution concerns offline behaviour supervision usable without visual observations and, if justified, factorized transfer under differing taxonomies. The first bidirectional kinematic diagnostic is intended to reveal failures; it is not itself the proposed final contribution. The exact RQs/contribution depend on results and a focused primary-source review. Shared heads, fusion, or a temporal architecture alone do not establish novelty.

The former rich-synthetic label-efficiency hypothesis is preserved in the [synthetic investigation](../archive/2026-10-01-synthetic-supervision.md). Do not carry its gap claim into this study without new evidence. Missing reviews are not evidence of novelty.
