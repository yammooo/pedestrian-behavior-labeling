# E004c — Adaptation depth versus label budget

**Status: implemented and checked on CPU/CUDA; full comparison remains user-launched.** ROAD→LOKI, one fixed all-modality E002f BiLSTM-96 architecture. This comparison extends [E004a](../E004a-head-label-budget/README.md), using its ordinary nested subsets and early-check results. It asks whether source behavior training saves target labels, and whether freezing the temporal representation restricts adaptation.

## Comparison

| Method | Initialization and trainable components | Fitting budgets |
|---|---|---|
| Source model | Original ROAD checkpoint; no target fitting | Fixed existing reference |
| Head only | Original source classifier; body frozen | Reuse E004a early-check fits |
| Temporal + head | Original source weights; BiLSTM, temporal output projection, final LayerNorm and classifier train (199,044 parameters) | 25 /100 /1,000 tracks × five draws |
| Target only | Random task branches, spatial readout, fusion, temporal block, norm and classifier (552,196 parameters); same frozen external encoders | Same budgets and exact subsets |

There are **30 new sequential fits**. The five repeats are subset draws 0–4 with training seed 0, not five optimization seeds. Full task-model fine-tuning from source is deferred. The existing LOKI-trained Transformer is contextual evidence, not a matched full-data BiLSTM baseline.

Reuse the E004a source checkpoint SHA256 `ac9bacf4c8c4bea16a63cf4ff2b9bf916079b7474b3a49e5b36ef7fe46df4574` and early-check attempt `20261113-082847-737608Z`. The ordinary draw prefixes remain unchanged; E004b's GT-informed sampling is not used. All selected tracks' accepted GT is fitted, with no class balancing or redrawing absent classes.

## Inputs and access

Keep frozen populations, complete-track future context, four classes, natural missingness and effective batches up to 64 tracks with partial batches retained. Microbatches accumulate to an effective batch. Loss remains accepted-GT mean CE per track, averaged over the actual effective batch.

Temporal adaptation uses the source-normalized, frozen pre-temporal fusion cache. Its branches remain in evaluation mode; the trainable temporal path also has dropout disabled, matching E004a/E003c adaptation. Source ROAD statistics stay frozen for every target split.

Target-only training uses the existing DINO RGB and indexed D-DITR caches. These external encoders remain frozen, but all task-specific components train. Continuous geometry-feature statistics are fitted only from valid observations in the selected target training tracks, including their unlabeled context, and frozen for development/assessment. It never fits statistics on the full target training pool or validation. The original source-model dropout configuration is retained for this randomly initialized model. This is a comparison of explicit training methods, not an isolation of initialization alone.

Use the same fixed E004a partitions: **930 development/calibration tracks /46 scenarios /29,626 GT timesteps**, and **931 assessment tracks /47 scenarios /30,605 GT timesteps**. Development selects checkpoints and confidence thresholds; assessment is evaluated only after selection. These validation groups have historical project exposure. Official tests remain closed. Fixed development labels are additional target annotation access; fitting-track counts are not total annotation effort.

## Training and stopping

| Setting | Fixed policy |
|---|---|
| Optimizer | AdamW, WD 1e-4, betas(0.9,0.999), epsilon 1e-8 |
| Temporal adaptation LR | Classifier 3e-4; temporal/projection/final norm 3e-5 |
| Target-only LR | All trainable task parameters 3e-4 |
| Schedule / gradient clipping | Constant learning rates, no warmup; global norm 1.0 |
| Precision / seed | Existing deterministic float32 configuration; seed 0 per fit |
| Development checks | Updates 0, 1, 5, 10, 25, 50, 100, then every 100 updates |
| Selection | Highest four-class track-weighted development macro-F1; lower CE, then earlier update for exact ties |
| Stopping | First scheduled check at least 1,500 updates after the last strict F1 improvement |
| Patience reset | Strict F1 improvement only; a CE-only selection change does not reset it |
| Maximum updates | **None** in comparison runs |

Each fit starts independently; no continuation from a smaller subset. Preserve a persistent seeded shuffled loader and retain final partial batches. Log actual updates, processed tracks, equivalent data epochs and runtime; validation-check ordinals are not dataset epochs. Step 0 is eligible for checkpoint selection. Existing E004a head fits stopped by patience before their old ceiling, so no head rerun is needed.

## Evidence and reporting

Reuse E004a's track weighting and confidence-selection calculations. For every selected checkpoint, save full development and assessment predictions, CE, raw/weighted confusions, per-class precision/recall/F1, denominators, label access, history, selection/stopping evidence and compute. Checkpoints and predictions stay local; W&B receives one run per fit with native losses/F1, configuration, summaries and plots. Training CE/F1 are online interval metrics from changing weights (and active dropout for target-only); development metrics use the fixed checkpoint in evaluation mode. Logging errors propagate with local evidence retained.

The three main comparison figures are:

1. Assessment macro-F1 versus budget, mean ± sample SD over the five draws, with the fixed source reference and matching E004a head fits.
2. Per-class assessment F1 versus budget, four panels; precision/recall remain in numerical tables.
3. Assessment automatic coverage versus budget, emphasizing 5% and 10% calibration error targets, with achieved assessment error and violations shown alongside coverage. Save 1% and 20% targets too.

Confidence acceptance remains per scored timestep, not whole tracks. Keep original complete-track weights in accepted subsets, equal-score ties, maximum empirical calibration coverage among feasible cutoffs, reject-all cases, class retention and achieved assessment errors. Never choose cutoffs or model settings using assessment results. Curves describe subset-draw variation and empirical confidence behavior; no risk guarantee or annotation-time saving is established automatically.

## Execution

On Aalto, use the existing E002a isolated environment that matches the frozen extractor identity (CUDA 12.6), rather than the standard cu118 environment. Both new methods run in that environment; original E004a head references retain their recorded backend. Dependency/backend versions are recorded and identical trajectories across backends are not assumed.

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run --no-project --python "$PWD/outputs/experiments/E002a/extractor-env/bin/python" \
  python scripts/train-e004c.py --device cuda
```

Outputs are fresh under `outputs/experiments/E004c/runs/<timestamp>/`. Append `--smoke` for one temporal-head and one target-only fit at 1,000 tracks/draw 0, one update each with offline W&B and full development/assessment passes. Those bounded fixtures are not comparison evidence. Full comparisons remain user-launched; no resume or automatic retries. Measured execution cost and preparation checks follow below.


## Preparation evidence and execution cost

Implementation revision `4e56719` was verified on Aalto. The bounded smoke at `outputs/experiments/E004c/runs/20261116-135541-286473Z-smoke/` completed both methods, one update and 64 track presentations each, with all 930 development and 931 assessment tracks scored. Both best/last checkpoints and prediction archives are present; saved predictions reconstruct metrics. Source initialization matches the pinned direct-transfer predictions. Protected-input changes are empty. W&B was explicitly offline; no full comparison was launched.

CPU validation: **302 tests pass, 13 expected CUDA skips**; eight focused E004c tests cover subset-only normalization, aligned sample/cost ordering, prediction persistence, reporting, and unbounded update-based patience. CLI/compile and document link/diff checks pass. The three report figures were visually checked on synthetic fixtures only; those are not results. Earlier failed startup attempts (cache-path lookup and cu118 extractor-identity mismatch) remain preserved under Aalto E004c runs.

Measured on the RTX 4080 using PyTorch `2.7.1+cu126`, the frozen feature environment and 1,000-track draw 0:

| Method | First optimizer update (64 tracks) | Mean full development pass (930 tracks) | Assessment prediction pass (931 tracks) |
|---|---:|---:|---:|
| Temporal + head |0.068 s|0.191 s|0.061 s|
| Target only |5.82 s|39.65 s|41.28 s|

Target-only subset normalization took 1.40 s. The development timing averages two complete checks (updates 0/1); the training timing is a **single first update**, including its normal feature preparation. These numbers describe this checked build and input batch, not all tracks or future revisions.

At that measured target-only rate, **1,500 updates plus 15 later development checks is about 2.6 hours** after the final strict improvement, excluding setup/final prediction and earlier training. Applying that same rate to all 15 target-only fits would be roughly 39 hours, but smaller budgets have different effective batch sizes and lengths, and every strict improvement extends training. With no update ceiling this is a scale estimate, not a queue completion guarantee.

[Local smoke status](../../outputs/experiments/E004c/acceptance/20261116-135541-286473Z-smoke/status.json) · [protected inputs](../../outputs/experiments/E004c/acceptance/20261116-135541-286473Z-smoke/protected.json) · [temporal evidence](../../outputs/experiments/E004c/acceptance/20261116-135541-286473Z-smoke/temporal-head-n1000-d00/selected-metrics.json) · [target-only evidence](../../outputs/experiments/E004c/acceptance/20261116-135541-286473Z-smoke/target-only-n1000-d00/selected-metrics.json). Full smoke checkpoints/predictions remain on Aalto; only small numerical evidence was collected locally.
