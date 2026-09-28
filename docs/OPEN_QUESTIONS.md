# Open questions

Status: Active  
Last updated: 2026-09-28

## Purpose and ownership

- Contains: unresolved **decisions**, their current evidence, and what would answer them, grouped by when they block work. No decision deadlines have been agreed.
- Links out: detailed facts to the [ontology](LABEL_ONTOLOGY.md), [dataset notes](DATASETS/DATASET_MATRIX.md), and [evaluation plan](EVALUATION_PLAN.md); optional methods belong in [IDEAS_BACKLOG.md](IDEAS_BACKLOG.md).

## Current dataset gate

### Q1. Is the full PedSynth++ release accessible, and what does it actually label?

Status: Open. The paper lists `RETREAT`, while post-checkpoint generator inspection found `NORMAL_CROSSING` in its enum; the full CSV has not been inspected. Resolve with release provenance, raw label values/counts, transitions, track and episode counts, and the exact code version. The public Zenodo package is only a demo subset. See [inspection plan](DATASET_INSPECTION_PLAN.md).

### Q2. Does LOKI support a fair low-label benchmark?

Status: Open. A first local release scan found raw action counts and distinct `(scenario, track_id)` counts (see the [LOKI note](DATASETS/LOKI.md)); track durations, episodes, transitions, rare-class coverage across scenes, and release provenance remain unchecked. Resolve these and define a scene-aware split before setting feasible track budgets and a selection rule that avoids easy-case bias.

### Q3. Which PedSynth++ behaviors can supervise the LOKI states?

Status: Open. A pure 12→4 dictionary is not defensible. Determine conditional or excluded source states, whether generic non-crossing `STOPPED` occurs, whether checking/hesitating matches real `WAITING_TO_CROSS`, and whether `UNKNOWN`/`OTHER` is needed. Resolve with primary definitions, released timelines, motion/road context, and examples recorded in [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md).

### Q4. Which inputs can be produced comparably in all three datasets?

Status: Open. The [three-way audit](DATASET_CONTRACT.md) suggests RGB plus ordered 2D boxes as the smallest candidate, but PedSynth++ IDs/timing and release alignment remain unverified. [Selected LOKI clips](DATASETS/LOKI.md#selected-clip-observations-2026-09-28) include labeled pedestrians outside the RGB view and 2D-only frames without behavior labels; decide how those cases enter an RGB/2D benchmark. Metric positions and ego compensation are not confirmed across PedSynth++, LOKI, and ECP2.0 tracking. Resolve by inspecting all relevant releases and testing comparable feature/pose extraction. Keep simulator-private state out of model inputs.

## Experimental decisions after the gate

### Q5. Which target states are observable from the chosen representation?

Status: Open. `WAITING_TO_CROSS` versus `STOPPED` may require approach/future context; `CROSSING` may require road relation. Resolve with a properly split LOKI-trained `R→R` diagnostic and class-wise errors before attributing failure to domain shift.

### Q6. Does rich synthetic supervision save real labels?

Status: Open. Compare the same architecture and inputs from scratch versus synthetic pretraining at fixed LOKI track budgets. If supported by source labels, compare native rich states with binary crossing and a defensible collapsed four-state objective. Resolve with held-out learning curves and per-class results, not a full-supervision F1 change alone.

### Q7. Which representations transfer, and which introduce domain shift?

Status: Open. Test motion first, then pose and visual inputs where residual errors justify them. Check whether RGB helps source performance but hurts `S→R`, whether one pose estimator works in both domains, and whether crop and scene context need separate treatment. Resolve through `S→S`, `S→R`, and `R→R` ablations.

### Q8. What temporal and sampling protocol is defensible?

Status: Open. Offline context is allowed, but useful duration, whole-track versus chunked processing, and few-shot selection are unchosen. Resolve after inspecting track lengths and transitions; predeclare split, context window, budget units, and repeated small-budget draws in the [evaluation plan](EVALUATION_PLAN.md).

## Later scope and contribution

### Q9. What contribution remains after the closest literature is compared?

Status: Open. Synthetic-to-real binary C/NC self-labeling already exists. Resolve through a focused primary-source comparison of multi-state annotation, rich synthetic pretraining, target-label efficiency, and independent label audits in [GAP_ANALYSIS.md](LITERATURE/GAP_ANALYSIS.md).

### Q10. When is automated enrichment trustworthy enough for an unlabeled dataset?

Status: Open. Adaptation labels alone omit audit/review cost. Resolve with a predeclared quality-versus-coverage target, independent manual audit, and total human budget before applying labels to ECP2.0. Confidence/abstention remains optional until transfer works.

### Q11. Can EMT serve as an external real-world test?

Status: Open. `Stopping` may not mean LOKI `Stopped`, and shared modalities are unverified. Resolve with EMT annotation definitions, frame semantics, track structure, and a common-input audit before choosing a cross-dataset test.
