# Research documentation

Status: Active guide  
Last updated: 2026-09-24

Read [RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md) for the current question and why it matters, then [PROBLEM_DEFINITION.md](PROBLEM_DEFINITION.md) for the exact task. The [dataset inspection plan](DATASET_INSPECTION_PLAN.md) is the immediate work gate. Each main document starts with a **Purpose and ownership** section; update the owner of a fact and link to it elsewhere.

| Document | Owns |
|---|---|
| [Research direction](RESEARCH_DIRECTION.md) | Motivation, central question, hypotheses, and reasons to continue or pivot. |
| [Problem definition](PROBLEM_DEFINITION.md) | Task boundary: annotation unit, available temporal context, inputs, outputs, and exclusions. |
| [Label ontology](LABEL_ONTOLOGY.md) | State meanings and evidence for source-to-target mappings. |
| [Dataset contract](DATASET_CONTRACT.md) | Shared input fields, availability, coordinate conventions, and leakage rules. |
| [Dataset inspection plan](DATASET_INSPECTION_PLAN.md) | Checks and outputs for the current source/target suitability gate. |
| [Pipeline design](PIPELINE_DESIGN.md) | Current conceptual processing flow and candidate modeling choices. |
| [Evaluation plan](EVALUATION_PLAN.md) | Splits, label budgets, baselines, metrics, and comparison protocol. |
| [Roadmap](ROADMAP.md) | Order of work and the next gate; detailed checklists live in specific plans. |
| [Open questions](OPEN_QUESTIONS.md) | Unresolved decisions, grouped by when they need answers. |
| [Ideas backlog](IDEAS_BACKLOG.md) | Optional techniques to consider if evidence reveals a need. |

## Evidence and chronology

- [Dataset matrix](DATASETS/DATASET_MATRIX.md) gives a short comparison; `DATASETS/` notes own release-specific facts, access, and schema findings.
- [Literature index](LITERATURE/INDEX.md) and [gap analysis](LITERATURE/GAP_ANALYSIS.md) point to paper notes and qualified prior-work claims.
- `DECISIONS/` records why an important choice was accepted. Do not create a record for an open question.
- `LOG/` records dated activity with links to changed documents, not repeated conclusions.

## When experiments begin

Keep versioned configurations in `configs/` and large outputs outside Git. Add `docs/EXPERIMENTS/` with a compact comparison index and a record for each meaningful experiment or tightly related run group. A record should identify dataset/release, split, ontology mapping, input representation, real-label budget, configuration and seed, metrics, result, and interpretation. The evaluation plan defines the protocol; experiment records contain observed results. Update canonical conclusions and open questions only when results change the current understanding.
