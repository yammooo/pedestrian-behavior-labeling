# E004a — Head adaptation versus labeled-track budget

**Protocol agreed 2026-11-12; implemented and checked. Full runs remain user-launched.** This is the first [E004 label-efficiency](../E004-label-efficiency/README.md) comparison. It asks how target performance changes as more complete target-training tracks supervise only the source classifier. Target-only controls remain necessary before claiming annotation savings.

## Initial model and data flow

Use the original ROAD-trained [E002f all-modality BiLSTM](../E002f-temporal-architecture/README.md#fixed-comparison), selected at source epoch28: `outputs/experiments/E002f/runs/temporal-architecture-20261110-091857/source/bilstm-96/best.pt`. Pinned SHA256: `ac9bacf4c8c4bea16a63cf4ff2b9bf916079b7474b3a49e5b36ef7fe46df4574`. This is a fixed competitive reference, not a claim of the universally best model.

```text
LOKI complete-track inputs, normalized with frozen ROAD statistics
  → frozen geometry / DINO / D-DITR spatial branches and fusion
  → frozen BiLSTM, projection and final LayerNorm
  → cached per-timestep 128-D representation
  → original source Linear(128, 4), fine-tuned on selected target GT only
```

Freeze the complete body, including final LayerNorm, in evaluation mode. Only the existing classifier's 516 parameters learn; do not initialize a fresh random head for this curve. No dropout during frozen extraction or head fitting. Reuse E002f's saved pre-temporal fusion cache. Derive and save post-BiLSTM/final-LayerNorm 128-D records once, then reuse them across all fits; no new encoder extraction or point-cache rebuild is needed. Keep full-track observations, including unlabeled context, without providing GT to model inputs.

First direction: ROAD-Waymo → LOKI. Select subsets only from the existing 8,574-track LOKI training pool; the 1,861-track validation population has already informed development. No test access in this pilot. The reverse direction is later work, not a second initial source-training run.

## Budget grid and subset draws

Fixed absolute training-track counts: **0, 1, 5, 10, 25, 50, 100, 250, 500, 1,000, all**. Zero is the original source classifier without target fitting. Each positive budget labels all accepted GT timesteps in the selected complete tracks. Report their raw GT count/duration, class support and distinct recordings alongside track count.

Use five independent subset draws for reduced budgets, with increasing subsets nested within each draw. Reuse identical subsets across later methods. Subset-draw seeds are 0–4 and the model-training seed is 0 for every fit. The two RNG uses are separate; variability is primarily from subset selection. Zero and all-data anchors do not become independent repetitions by duplicating the same inputs/initialization.

For each draw, start from canonical scenario/track identities, shuffle scenarios and tracks within each scenario, then take one track per scenario in rounds, skipping exhausted scenarios. Increasing budgets are prefixes of that single ordering. This is recording-diverse sampling, not uniform random sampling of tracks; only selected tracks are labeled, not all tracks in their scenarios. Preserve existing train/validation/test groups. Do not balance classes or redraw until rare classes appear. All-data uses one full-pool fit; zero is one original-head evaluation. This gives 45 reduced-budget fits plus one full-data fit, not five duplicated anchors.

## Evaluation and annotation accounting

The fixed fitting pool is the original 8,574-track LOKI training split. Shuffle the 93 original LOKI validation scenarios with an independent RNG at seed 0, after canonical sorting: the first 46 are **development/calibration**, and the remaining 47 are **assessment**. Keep every track of a scenario in its assigned partition. Save exact IDs, GT counts and class support before fitting; do not use GT to rebalance or redraw this split.

Use development/calibration only for checkpoint selection and cutoff selection. Evaluate assessment only after reloading the selected checkpoint; it must not affect stopping, source choice, hyperparameters or cutoffs. The same partitions apply to every budget/draw. Both pools were inspected earlier in the project; the E004a partition does not make them historically unseen.

Record fitting labels and the fixed development/calibration overhead separately, with their union as training/selection label access. Report assessment labels separately as evaluation/audit effort and retain total labeled-evidence counts. A one-track fit is not one-track total target annotation. The deterministic split contains **930 development/calibration tracks /29,626 GT timesteps** and **931 assessment tracks /30,605 GT timesteps**. All four classes are represented in both; exact IDs and supports are saved in `subset-plans.json`. Retain all four classes, failed fits and negative transfer.

## Fixed head-fitting recipe

Each fit copies the same original source head and resets its optimizer; never continue from a smaller-budget adapted head. Only 516 classifier parameters learn, with frozen ROAD normalization and no dropout. Use deterministic float32 and the existing accepted-GT mean-track CE, averaged over the actual batch.

| Setting | Value |
|---|---|
| Optimizer | AdamW, LR 3e-4, weight decay 1e-4, betas(0.9,0.999), epsilon 1e-8 |
| Schedule / clipping | Constant LR, no warmup; global gradient clipping 1.0 |
| Batches | Up to 64 complete labeled tracks; retain final partial batches |
| Training ceiling | 10,000 optimizer updates |
| Development evaluation | At update 0 and every 100 updates |
| Selection | Highest four-class track-weighted development F1; lower CE then earlier update for exact ties |
| Early stopping | 15 checks without strict F1 improvement; CE ties do not reset patience |
| Training order | Persistent seeded shuffled loader; restart its iterator only after exhaustion |
| Saved state | Original head is an update 0 candidate; atomic best/last head state dictionaries and source/cache references |

One training interval is 100 optimizer updates, **not** an epoch over the dataset. Track actual updates, track presentations, context/GT slots and runtime. The fixed ceiling is a practical optimizer budget, not a convergence guarantee; retain budget-limited attempts. Zero budget performs no gradient update.

## Fixed reporting specification (2026-11-12)

Produce two main figures: `budget-f1.png` / `.svg` and `budget-coverage.png` / `.svg`. Use absolute labeled-training-track count on the x-axis, linear from 0 to 1 then logarithmic above 1 so the zero-fitting-label anchor is visible. Place the all-data anchor at its actual track count. Show the fixed development/calibration-label overhead in captions and numerical records; this is not total-label cost on the x-axis.

| Figure | Y-axis and grouping | Repeats |
|---|---|---|
| **Budget versus F1** | Full-coverage four-class track-weighted macro-F1 (%) on the declared assessment population; one E004a head-adaptation series | Mean ± sample SD (`ddof=1`) over the five subset draws |
| **Budget versus automatic coverage** | Track-weighted coverage (%); four colored series for calibration error targets **1%, 5%, 10%, 20%**, with 5% the central reporting point | Mean ± sample SD of each run's achieved assessment coverage; never pool predictions across models |

Zero and all-data anchors have one fixed model result, not five independent copies; omit their SD bars. Use the same assessment population for every budget/draw. SD describes subset-draw variation conditional on the fixed source model/training seed, not uncertainty over new recordings. If the reporting population is also used to select a checkpoint or threshold, label results as development results rather than independent assessment.

### Acceptance rule and cutoff selection

- Acceptance unit: individual scored GT timestep. Score: maximum softmax probability of the predicted class, without a learned calibrator or rejection head.
- Primary weights: `1 / original_track_GT_count`, retained unchanged in every accepted subset. Primary accepted error is weighted misclassification fraction, not `1 - macro-F1`. Save raw-frame companions.
- For each selected checkpoint (each budget/draw, not each training epoch), compute the full curve at every distinct score cutoff. Accept equal-score ties together; no 5%-coverage quantization. Do not assume empirical error is monotone or stop at the first constraint violation.
- For each error target, choose the cutoff with **maximum calibration coverage** among those with observed calibration error at or below the target. If no nonempty accepted set qualifies, reject all: coverage 0, error undefined. No arbitrary minimum-support exclusion is added to this descriptive pilot; save accepted frames/tracks/groups and error counts.
- Freeze that run-specific cutoff and apply it unchanged to assessment predictions. The model may need a different cutoff at each budget/draw/error target. Never tune cutoffs on assessment GT.
- Plot the resulting assessment coverage even if its achieved error exceeds the calibration target; do not hide failed draws or readjust using assessment labels. Label legend entries as **calibration error targets**, not guaranteed assessment errors. Use hollow budget markers if any nonempty draw exceeds its target; the numerical summary gives the count of violations and reject-all cases.
- Coverage means/SD include reject-all runs as zeros. Accepted-error means/SD use nonempty accepted sets only and retain their repeat count; undefined error is never replaced by zero.

The coverage figure answers how much can be accepted using a rule selected to meet a calibration target. It does not guarantee that the target is achieved on unseen data. Save actual achieved error alongside every plotted point.

### Class retention and saved data

For each GT class `c`, retain the original assessment denominator and report:

```text
GT-class retention = accepted GT-c weight / all GT-c weight
accepted precision = accepted correct-c weight / accepted predicted-c weight
correctly automated fraction = accepted correct-c weight / all GT-c weight
```

Zero accepted predicted-c weight gives undefined precision; zero original GT-c support gives undefined retention/correctly automated fraction. Do not substitute recall computed only within the accepted subset for class retention. Numerical records include class supports and natural-availability cohorts; aggregate accuracy must not hide rejection of rare behaviors.

Save per-run F1/CE, full calibration and assessment error–coverage curves, selected cutoffs, achieved coverage/error, raw and weighted counts, class retention/precision, budget/draw identity and annotation-access counts. Save aggregate means/SD and repeat counts separately. Full curves are supporting numerical evidence, not a third required main figure; 5%-coverage tables or representative curves can be produced later without changing the calculation. Keep artifacts local under ignored `outputs/experiments/E004a/`, following existing prediction/provenance conventions.

## Implementation and scope

Reuse the E002f feature reader/source model, cached-track loader, original-head initialization and shared training/checkpoint/metric operations. E004a adds the nested subset plan, fixed-update interval wrapper, coverage analysis and two-figure summary. No test preparation, encoder fine-tuning, point-cache changes, architecture search or full experiment execution is part of preparation.

The new fitting/partition policy differs from E002f's old full-label adaptation recipe, so the all-data anchor is fitted afresh. No previously target-adapted head is reused as initialization. Statistical risk guarantees and uncertainty over new recordings remain outside this descriptive pilot.

## Run

On Aalto, after pulling the reviewed revision:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run --locked --extra cu118 python scripts/train-e004a.py --device cuda
```

This builds the compact frozen representation cache once, then runs 45 reduced-budget fits and one all-data fit sequentially, plus the zero-fitting-label evaluation. Outputs are fresh under `outputs/experiments/E004a/runs/<timestamp>/`; the reusable compact cache is `outputs/experiments/E004a/features/postnorm-bilstm-96/`. Original checkpoints/caches remain read-only. One W&B run per fit logs step CE, online interval training CE/F1, development CE/F1, and final assessment tables. Full predictions and checkpoints remain local. The two main figures and numerical summaries are in the attempt's `report/` directory.

`--smoke` instead runs budgets1/5, draw0, 200 updates each, with the full fixed development/assessment partitions and **offline W&B**, in a separate attempt. It is an integration check, not evidence for the label-efficiency comparison. Production logging errors stop the attempt; there is no retry/resume or silent offline fallback.

## Preparation checks (2026-11-12)

- Required CPU suite: **286 tests pass,13 CUDA skips**; focused sampler/update-cadence and coverage regressions pass. Documentation links/anchors and `git diff --check` pass.
- Aalto CUDA smoke at `2dc5d5a`: budgets1/5, draw0,200 updates each, full fixed development/assessment partitions, offline W&B. Checkpoint reloads,200 sequential step records per fit, prediction exports, four cutoff analyses and both PNG/SVG figures pass; figures visually checked. This is integration evidence only. Both tiny smoke fits retained update0; no label-efficiency conclusion follows.
- Compact cache is approximately165.4MB. Copied-head predictions on its new post-temporal representations reproduce the original source logits. Source checkpoint SHA remains unchanged. No native cache rebuild or held-out test access.
- Remote evidence: `outputs/experiments/E004a/runs/20261112-091726-841239Z/verification.json`, with local inspection copies in `outputs/experiments/E004a/acceptance/20261112/`. The first failed smoke at `20261112-091530-427492Z` is preserved: numeric `point_tokens` provenance was incorrectly treated as availability; cohort analysis now uses only `*_valid` fields.

Full46-fit training was **not started**. Production online W&B uses the existing logging path; the bounded check deliberately used offline W&B. Empirical calibration targets are not risk guarantees.
