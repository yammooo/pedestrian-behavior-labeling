# Roadmap

Status: Provisional schedule, conditional on evidence

Last updated: 2026-10-02

## Purpose and ownership

- Contains: work order, feasibility gates, and internship schedule.
- Links out: gate details to [DATASET_INSPECTION_PLAN.md](DATASET_INSPECTION_PLAN.md), protocols to [EVALUATION_PLAN.md](EVALUATION_PLAN.md), and run recording to [EXPERIMENTS/README.md](EXPERIMENTS/README.md).

## Schedule

The traineeship ends **17 December 2026**. These windows express priorities, not guaranteed outcomes or frozen architecture decisions.

| Window | Intended work / gate |
|---|---|
| 1–9 October | Complete visual/reproducibility acceptance of the acquired index; audit the tentative ROAD-Waymo/LOKI four-state mapping; verify common coordinates, ego/velocity semantics, timing and population/cohort rules; freeze the first diagnostic protocol. |
| 12–23 October | Once implementation is authorized, build the minimal data representation and kinematic framewise MLP/BiLSTM baselines; evaluate within both datasets and transfer in both directions; inspect failures before selecting an extension. |
| 26 October–13 November | If baseline failures justify it, test scene evidence, missing-visual-evidence methods, factorization or a complementary source. No requirement to add all of them; final architecture and direction remain open. |
| 16–27 November | Conditional generalization experiments: missing RGB, justified sensor perturbations/alignment, source additions, and negative transfer. IDD-PeD is a later option. |
| 30 November–4 December | Main supported comparisons: strict zero-shot where semantics permit, low-shot versus scratch at equal budgets, source/cohort ablations, class-wise failures; uncertainty/coverage only if feasible. |
| 7–17 December | Protected buffer and consolidation: delayed runs, essential ablations, report, documentation, cleanup, handover, and presentation. |

## Priority and fallback rules

ROAD-Waymo participates in the first two-dataset matrix only after linkage/semantic acceptance. A failed gate triggers reconsideration; no substitute source is silently selected. Both datasets can be source or target in separate runs. Simple baseline evidence comes before complex heterogeneous training. The current handoff authorizes documentation only, not model implementation.

Adding nuScenes, ROAD, or IDD-PeD requires a specific hypothesis and feasibility evidence. Adding all sources is not a deliverable. Synthetic generation and ECP2.0 deployment are not dependencies.

The intended result is a working offline research labeler with reproducible experimental evidence. Scale the supported claims to results; preserve the December buffer instead of making optional methods mandatory or planning required work beyond the traineeship.
