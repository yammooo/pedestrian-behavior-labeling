# Research documentation

Status: Active ownership guide

Last updated: 2026-10-02

Read [RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md) for the current question, then [PROBLEM_DEFINITION.md](PROBLEM_DEFINITION.md) for task scope. The [dataset inspection plan](DATASET_INSPECTION_PLAN.md) owns immediate feasibility gates. Current methods are hypotheses, not architecture decisions.

| Document | Owns |
|---|---|
| [Research direction](RESEARCH_DIRECTION.md) | Motivation, central question, hypotheses, and pivot criteria. |
| [Problem definition](PROBLEM_DEFINITION.md) | Annotation unit, temporal evidence, inputs/outputs, observability, and exclusions. |
| [Label ontology](LABEL_ONTOLOGY.md) | Native meanings/granularity, candidate shared concepts, and justified evaluation mappings. |
| [Dataset contract](DATASET_CONTRACT.md) | Identity/time/coordinates, heterogeneous availability, provenance, and leakage rules. |
| [Dataset inspection plan](DATASET_INSPECTION_PLAN.md) | ROAD-Waymo linkage, ontology, LOKI population, and baseline/protocol gates. |
| [Pipeline design](PIPELINE_DESIGN.md) | Baseline progression and candidate representations, heads, and training procedures. |
| [Evaluation plan](EVALUATION_PLAN.md) | Access regimes, splits, budgets, candidate metrics, comparisons, and audit protocol. |
| [Roadmap](ROADMAP.md) | Conditional work order and schedule through 17 December, with protected buffer. |
| [Open questions](OPEN_QUESTIONS.md) | Unresolved decisions and brief superseded-question pointers. |
| [Ideas backlog](IDEAS_BACKLOG.md) | Optional methods justified only by a measured gap. |
| [Experiments](EXPERIMENTS/README.md) | Run/comparison index and reproducible observed results; not the global protocol. |

## Evidence and chronology

- [Dataset matrix](DATASETS/DATASET_MATRIX.md) compares current roles; individual notes own release facts, primary references, and local findings.
- [Literature index](LITERATURE/INDEX.md) and [gap analysis](LITERATURE/GAP_ANALYSIS.md) navigate prior work and qualify contribution claims.
- [Decision records](DECISIONS/README.md) explain settled material choices. Do not create architecture decisions for unresolved hypotheses.
- `LOG/` records chronology with pointers to the owners; preserve past entries.
- Superseded synthetic reasoning and completed CARLA side work live in [PedSynth++ research history](DATASETS/PED_SYNTH_PLUS_PLUS.md#research-history). Git retains former full plans.

## Scientific method and provenance

The latest user-provided 2026-10-02 research-question/baseline handoff updates the 2026-10-01 checkpoint: two provisional RQs and a first kinematics-only MLP/BiLSTM diagnostic in both ROAD-Waymo/LOKI transfer directions. It authorizes documentation updates, not implementation. These handoffs are project context, not independent verification of every dataset claim. Label facts as paper-reported, official-format-documented, locally observed, or checkpoint-reported; keep source URLs, versions, and unknowns explicit.

Predeclare each comparison before its run. Keep configurations in `configs/`, data and large artifacts outside Git, and results once in [experiment records](EXPERIMENTS/README.md). Document failures and deviations. Update canonical conclusions only when evidence changes understanding.

The intended artifact is a working offline research labeler. Baseline evidence precedes heterogeneous training; metrics, budget units, source additions, and architecture remain open until their gates.
