# E004a — Head adaptation versus labeled-track budget

**Full pilot completed on 2026-11-12; ROAD→LOKI, validation assessment only.** This is the first [E004 label-efficiency](../E004-label-efficiency/README.md) comparison. It asks how target performance changes as more complete target-training tracks supervise only the source classifier. Target-only controls remain necessary before claiming annotation savings.

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

At preparation, full46-fit training had **not started**; the completed user-launched pilot is recorded below. Production uses online W&B; the bounded check deliberately used offline W&B. Empirical calibration targets are not risk guarantees.

## Results

**All 46 fits and the original-head anchor completed on 2026-11-12**, at clean revision `96c0a2d`, in 11.1 minutes using the previously cached frozen representations. No new encoder extraction or test access. All fits stopped by patience; none reached 10,000 updates. Six selected update 0, retaining the original source head. Exact subset plans, access counts, checkpoints and predictions remain in the original Aalto attempt: `outputs/experiments/E004a/runs/20261112-092028-490189Z/`.

Assessment is the same 931 tracks /47 scenarios /30,605 GT timesteps for every result. Reduced budgets show mean ±sample SD across five subset draws, with training seed 0 fixed. Zero/all anchors have one result each. These are historically inspected validation data, separated from this pilot's fitting/selection; they are not untouched test results.

### F1 and confidence coverage by fitting budget

All F1, coverage and error values below are percentages; CE is mean track CE. Coverage uses the **5% calibration error target**, but achieved assessment error is measured separately. Overruns count checkpoint results above that target; these results share the same assessment population and are not independent datasets.

| Fitting tracks | Assessment F1 ±SD | CE | Assessment coverage ±SD | Assessment accepted error, mean | 5% overruns |
|---|---:|---:|---:|---:|---:|
| 0 | 41.67 | 1.236 | 0.84 | 7.83 | 1/1 |
| 1 | 42.27 ±1.33 | 1.210 | 2.77 ±4.32 | 7.31 | 5/5 |
| 5 | 47.06 ±3.62 | 0.973 | 19.41 ±17.51 | 8.33 | 4/5 |
| 10 | 47.70 ±3.49 | 0.962 | 18.69 ±12.97 | 9.30 | 5/5 |
| 25 | 51.14 ±1.93 | 0.851 | 27.47 ±7.07 | 8.46 | 5/5 |
| 50 | 50.57 ±1.14 | 0.808 | 26.89 ±10.84 | 10.42 | 5/5 |
| 100 | 52.83 ±2.44 | 0.731 | 36.33 ±5.74 | 9.26 | 5/5 |
| 250 | 55.56 ±3.01 | 0.677 | 46.63 ±6.02 | 11.39 | 5/5 |
| 500 | 55.81 ±2.75 | 0.642 | 50.23 ±9.58 | 10.49 | 5/5 |
| 1,000 | 56.84 ±1.25 | 0.581 | 57.51 ±3.40 | 10.69 | 5/5 |
| 8,574 | 55.28 | 0.560 | 54.83 | 8.99 | 1/1 |

![F1 versus fitting-track budget](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/report/budget-f1.png) ([SVG](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/report/budget-f1.svg))

**F1 improves with target fitting, with diminishing gains.** The original 41.67% becomes 51.14% ±1.93 at 25 tracks, 52.83% ±2.44 at 100, and 56.84% ±1.25 at 1,000. Gains beyond 250 tracks are modest relative to subset-draw variation. The single all-data anchor is 55.28%; this does not establish that fewer labels outperform all labels or that the methods are equivalent. Full-data CE is lowest, despite its lower F1 than the 1,000-track mean.

**Count the fixed label overhead:** every budget also accesses 930 development/calibration tracks /29,626 GT timesteps. Thus 100 fitting tracks means 1,030 training/selection tracks, and 1,000 means 1,930; assessment's 931 labeled tracks are separate audit effort. The 1,000-track draws cover all 430 training scenarios and 31,495–32,027 GT timesteps; the all-data fit uses 8,574 tracks /262,071 GT. This is recording-diverse sampling, not uniform random track sampling. [Per-fit access counts](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/report/budget-runs.csv) retain exact denominators and nominal GT durations.

### Automatic coverage does not meet its calibration targets

![Coverage versus fitting-track budget](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/report/budget-coverage.png) ([SVG](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/report/budget-coverage.svg))

Hollow points flag an assessment overrun in at least one draw. The 5% target is exceeded in 46/47 checkpoint results, including all five 1,000-track draws. At 1,000 tracks, 57.51% ±3.40 coverage comes with 10.69% mean accepted error. With all data, 54.83% coverage has 8.99% error. These are **not** coverage estimates at a guaranteed 5% assessment error.

The all-data anchor makes the distinction clear:

| Calibration error target | Observed calibration error | Assessment coverage | Assessment accepted error |
|---|---:|---:|---:|
| 1% | 0.71% | 3.27% | 0.24% |
| 5% | 5.00% | 54.83% | 8.99% |
| 10% | 10.00% | 82.84% | 16.89% |
| 20% | 16.10% | 100.00% | 22.66% |

Only the all-data 1% cutoff among these four all-data operating points meets its target on assessment, with 3.27% coverage. Across all 47 checkpoint results, 1%/5%/10%/20% targets overrun in 43/46/47/47 cases; there are no empty accepted sets. This describes this fixed split and confidence rule, not a population-level failure probability. Cutoffs were not retuned on assessment labels.

Aggregate coverage also hides class rejection. For the **all-data fit**, the table compares full-coverage class F1 with the 5%-calibration-cutoff acceptance statistics. All entries are percentages; undefined accepted precision is shown as—.

| Class | Development F1 | Assessment F1 | GT-class retention | Accepted precision | Correctly automated GT fraction |
|---|---:|---:|---:|---:|---:|
| Moving | 91.63 | 88.27 | 69.28 | 91.17 | 68.91 |
| Stopped | 55.41 | 40.31 | 1.01 | — | 0.00 |
| Waiting to cross | 73.90 | 55.74 | 6.99 | 66.46 | 1.32 |
| Crossing | 52.72 | 36.79 | 44.21 | 88.97 | 12.53 |

Stopped and Waiting are largely rejected. Crossing retains 44.21% of GT weight but only 12.53% is correctly automated; retained GT can still be assigned another class. High accepted precision for a predicted class does not mean high GT-class retention or safe automation of that behavior.

### Conclusions and limits

- **The frozen source representation supports useful target adaptation.** Head fitting raises assessment F1 by roughly 10–15 points with tens to hundreds of fitting tracks under this recipe. It does not establish annotation savings over target-only training; that matched control is still missing.
- **Selection/calibration does not transfer cleanly across these target scenarios.** The all-data fit has 68.41% development F1 /0.463 CE versus 55.28% /0.560 on assessment. This gap and the cutoff overruns warrant caution about representativeness and selection optimism; they do not identify a cause or prove annotation inconsistency. The older E002f 64.56% head result used the whole 1,861-track validation pool and a different fitting recipe, so it is not directly comparable to this assessment score.
- **A large automatic-label coverage claim is unsupported by the current confidence rule.** Report achieved errors and class retention alongside coverage. These results justify the next matched E004 controls, not a new architecture search or a reliability guarantee.

Numerical evidence: [budget summary CSV](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/report/budget-summary.csv), [per-fit CSV](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/report/budget-runs.csv), [complete report JSON](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/report/budget-report.json), [subset plan](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/subset-plans.json), [all-data selected metrics](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/budget-all-selected-metrics.json), [provenance](../../outputs/experiments/E004a/results-review/20261112-092028-490189Z/provenance.json). Report aggregates were checked against all 47 saved result rows; plots were visually checked. Source checkpoint/normalization references remain pinned. No retraining or inference was performed for this review.

## Early-check rerun (2026-11-13)

**Agreed diagnostic; full rerun remains user-launched.** Check whether the original 100-update evaluation spacing missed useful low-label checkpoints. Keep the same source head/body, optimizer, learning rate, normalization, training seed, all 46 fits, five nested subset draws, development/assessment split and coverage analysis. There is no class balancing or resampling. Original results above remain the baseline.

The original histories select update 0 in four of five one-track draws; five-track fits select0/100, and ten-track fits select0/100/200. With at most 64 tracks, 100 updates already make 100 passes over the subset. All five one-track subsets contain only Moving; early checking cannot fix that missing class support.

| Setting | Early-check diagnostic |
|---|---|
| Development checks / checkpoint candidates | Updates 0, 1, 5, 10, 25, 50, 100, then 200, 300, …, 10,000 |
| Patience | Stop at the first scheduled check at least 1,500 updates after the last strict development-F1 improvement; update 0 starts the clock |
| Exact F1 ties | Lower CE, then earlier update; CE-only improvements do not reset patience |
| Training order | One persistent seeded loader iterator across all short intervals; no reset at checks |
| Evidence axis | Actual cumulative optimizer updates for histories, W&B, selected checkpoints and plots |
| Outputs | Fresh attempt; `early-checks` in configuration and W&B run names; existing outputs untouched |

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run --locked --extra cu118 python scripts/train-e004a.py --device cuda --early-checks
```

Omitting `--early-checks` retains the original protocol. Add `--smoke` for the two-fit, 200-update offline-W&B integration check only. The rerun reuses the compact cache and performs no encoder extraction.

This is a target-informed follow-up to the inspected curves, not a new untouched comparison. Every check may select `best.pt`; assessment is computed only after selection and never determines the checkpoint. An earlier development peak alone does not establish better assessment performance or improved automatic-label reliability. Keep both attempts and their actual label-access counts.


Preparation checks at `644a944`: **288 CPU tests pass (13 CUDA skips)**; focused shared/E004a tests, CLI and documentation checks pass. Aalto's two-fit CUDA/offline-W&B smoke checks updates0/1/5/10/25/50/100/200 and selects checkpoints10/5. At update200 both last-checkpoint parameter states are bit-for-bit identical to the original200-update smoke, so the extra evaluations preserve the optimizer trajectory. All46 regenerated subset plans match the original full attempt, and the source checkpoint SHA is unchanged. Evidence: `outputs/experiments/E004a/runs/20261113-082424-712968Z/verification.json`. This is integration evidence; the full early-check comparison has **not started**.
