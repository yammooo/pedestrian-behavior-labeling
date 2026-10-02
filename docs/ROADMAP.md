# Roadmap

Status: Provisional schedule, conditional on evidence

Last updated: 2026-10-01

## Purpose and ownership

- Contains: work order, feasibility gates, and internship schedule.
- Links out: gate details to [DATASET_INSPECTION_PLAN.md](DATASET_INSPECTION_PLAN.md), protocols to [EVALUATION_PLAN.md](EVALUATION_PLAN.md), and run recording to [EXPERIMENTS/README.md](EXPERIMENTS/README.md).

## Schedule

The traineeship ends **17 December 2026**. These windows express priorities, not guaranteed outcomes or frozen architecture decisions.

| Window | Intended work / gate |
|---|---|
| 1–9 October | Acquire and test ROAD-Waymo/Waymo linkage; native ontology audit; LOKI population/visibility characterization; define the first source/target protocol. |
| 12–23 October | Small verified data layer and simple baselines: LOKI trajectory-only, scene diagnostic if feasible, ROAD-Waymo source baseline, basic transfer. Assess the next source; do not assume nuScenes must precede ROAD. |
| 26 October–13 November | If baselines justify it, test shared representations and a first complementary source. Compare with ROAD-Waymo alone; architecture and native heads remain hypotheses. |
| 16–27 November | Conditional generalization experiments: missing RGB, justified sensor perturbations/alignment, source additions, and negative transfer. IDD-PeD is a later option. |
| 30 November–4 December | Main supported comparisons: strict zero-shot where semantics permit, low-shot versus scratch at equal budgets, source/cohort ablations, class-wise failures; uncertainty/coverage only if feasible. |
| 7–17 December | Protected buffer and consolidation: delayed runs, essential ablations, report, documentation, cleanup, handover, and presentation. |

## Priority and fallback rules

ROAD-Waymo is the baseline source candidate only if linkage passes. A failed gate triggers reconsideration of source supervision; no substitute source is silently selected. Simple baseline evidence comes before complex heterogeneous training.

Adding nuScenes, ROAD, or IDD-PeD requires a specific hypothesis and feasibility evidence. Adding all sources is not a deliverable. Synthetic generation and ECP2.0 deployment are not dependencies.

The intended result is a working offline research labeler with reproducible experimental evidence. Scale the supported claims to results; preserve the December buffer instead of making optional methods mandatory or planning required work beyond the traineeship.
