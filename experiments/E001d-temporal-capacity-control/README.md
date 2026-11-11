# E001d: temporal capacity control

## Question and scope

Does increasing framewise MLP capacity to match the parameter count of E001's K+T+R BiLSTM change source and cross-dataset performance? This development-informed follow-up adds one model; it does not change E001 populations, labels, splits, normalization or evaluation.

Original E001 MLP/BiLSTM seed0–4 results and held-out tests informed later development. E001d evaluates both held-out datasets through the shared runner, but these are not pristine confirmatory tests. Match new attempts to the [original five-seed K+T+R MLP/BiLSTM controls](../E001b-2d-bbox-contribution/results-review.md#primary-track-weighted-macro-f1) by source and seed. Those controls are preserved under `outputs/experiments/E001/runs/seeds-20261009-084417-412679/`; no control retraining is needed.

## Model and fixed settings

The input is K+T+R, 12 dimensions. Starting from E001's framewise `KinematicClassifier`, append `Linear(64,194) → ReLU → Linear(194,64)` to its encoder. The appended block has no dropout and no final ReLU. Keep the original encoder, dropout and `64 → 4` readout. There is no recurrence or state across frames. The new model has **30,342 trainable parameters**; the matched original BiLSTM has **30,340**, while the original MLP has **5,252**.

Reuse [E001's protocol](../E001-kinematic-transfer/README.md), saved collections, K+T+R source-only normalization, condition definitions, equal-track training, loss, AdamW (`lr=0.001`, `weight_decay=0.0001`), batch size64, float32, gradient clipping1.0, 75 epoch maximum and patience8. Select on source validation, reload its best checkpoint, and evaluate both held-out tests with the source normalization. Original E001 defaults and artifacts remain unchanged.

The exactly ten new attempts are ROAD-Waymo and LOKI, each at seeds0–4. Match all settings to the original E001 seed0–4 K+T+R MLP/BiLSTM attempts. No target training or target-based selection is added. Retain failures and all metric denominators using the shared attempt runner.

## Command and outputs

On the verified Aalto evaluation checkout, the existing E001 output links provide collections/setups; no native inputs or preparation are needed. From `/home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation`:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
git rev-parse HEAD
uv run --locked --extra cu118 python scripts/train-e001d-seeds.py --jobs 2 --device cuda
```

The launcher creates a fresh `outputs/experiments/E001d/runs/matched-capacity-<UTC>/` with ten readable attempt directories, logs and `sweep.json`. Use `--jobs 1` for sequential execution or `--output <fresh-directory>` to name the batch. Each attempt follows E001 artifact handling and writes its model/settings, source statistics, predictions, metrics and checkpoint. A single source/seed attempt can be launched with `scripts/train-e001d.py` and explicit `--source`, `--seed`, `--device` and `--output`.

## Results

Reviewed 2026-11-11. The Aalto sweep at `outputs/experiments/E001d/runs/matched-capacity-20261111-142500-354558/` completed **10/10 new attempts / 20 held-out test cells** (ROAD-Waymo and LOKI, seeds0–4), with no failures. Joined to the 20 existing matched E001 controls, the comparison covers **30 attempts / 60 test cells**. All selected checkpoints were chosen by source-validation F1; target test scores were not used for training or selection. These tests had informed earlier development, so this remains a development-informed follow-up rather than a pristine confirmatory result.

The new attempts ran on Aalto's RTX 4080 Laptop GPU with two CUDA jobs, at code revision `873ddd0198bc4d19c9691068c63bf019d5b2481b`, Python3.11.16, PyTorch2.7.1+cu118 and NumPy2.4.6. All ten used the declared 75-epoch/patience-8, batch-64, float32 AdamW profile and stopped on patience. New-run selected epochs ranged5–31; completed epochs ranged13–39. The 20 old controls are complete, use the same source/seed settings and determinism profile, and match the new runs' dataset/setup/collection/policy hashes and source normalization statistics. Saved test confusion matrices reproduce the recorded track-weighted per-class and macro F1; across all 60 metric cells, groups/tracks/GT-frame counts and per-class GT-frame supports match frozen test support.

R = ROAD-Waymo; L = LOKI. Arrows mean training source → held-out test dataset. Primary metric is four-class track-weighted macro-F1 (%), reported as mean ± sample SD across five seeds (`ddof=1`). Each seed contributes its own saved test metric; confusions are not pooled across seeds. The splits are fixed, so SD describes training-seed variability, not uncertainty over new scenes or alternative splits. Parameter counts are 5,252 for the original framewise MLP, 30,342 for the matched framewise MLP and 30,340 for the BiLSTM.

### Primary test results

| Model | R→R | R→L | L→L | L→R |
|---|---:|---:|---:|---:|
| MLP (5,252) | 44.75 ± 0.24 | 42.72 ± 0.71 | 53.30 ± 2.45 | 39.51 ± 2.65 |
| MLP matched (30,342) | 45.35 ± 0.51 | 43.82 ± 0.95 | 56.37 ± 1.38 | 42.16 ± 1.21 |
| BiLSTM (30,340) | 52.84 ± 1.04 | 46.90 ± 0.83 | 61.19 ± 1.94 | 42.22 ± 1.59 |

The matched MLP improves over the small MLP in all four direction means, by **0.60 / 1.10 / 3.07 / 2.65 pp** in R→R / R→L / L→L / L→R; each difference is positive in four of five seeds. This supports a capacity contribution. The BiLSTM remains ahead of the matched MLP by **7.49 / 3.08 / 4.82 pp** in the first three directions (positive in all five seeds), while L→R is a near tie (**+0.07 ± 2.55 pp**, positive in two seeds).

![Held-out track-weighted macro-F1 by direction, with individual seed scores and mean ± SD](../../outputs/experiments/E001d/results-review/figures/primary-f1.png) ([SVG](../../outputs/experiments/E001d/results-review/figures/primary-f1.svg)).

### Paired primary differences

Same-seed paired differences in percentage points, mean ± sample SD; parentheses give seeds with a positive difference.

| Difference | R→R | R→L | L→L | L→R |
|---|---:|---:|---:|---:|
| Matched MLP − small MLP | +0.60 ± 0.65 (4/5) | +1.10 ± 1.08 (4/5) | +3.07 ± 3.19 (4/5) | +2.65 ± 3.12 (4/5) |
| BiLSTM − matched MLP | +7.49 ± 0.55 (5/5) | +3.08 ± 1.20 (5/5) | +4.82 ± 3.05 (5/5) | +0.07 ± 2.55 (2/5) |

### Optimization check

Cross-entropy is shown at the source-validation-F1-selected epoch and at the final completed epoch; values are mean ± sample SD across five runs of each source/model. Training CE is an online epoch average as weights change, with training dropout active; validation CE is measured on source-validation data at the same epoch.

| Training source | Model | Selected train CE | Selected source-val CE | Final train CE | Final source-val CE |
|---|---|---:|---:|---:|---:|
| ROAD-Waymo | MLP (5,252) | 0.761 ± 0.015 | 0.843 ± 0.002 | 0.741 ± 0.011 | 0.848 ± 0.004 |
| ROAD-Waymo | MLP matched (30,342) | 0.731 ± 0.032 | 0.849 ± 0.010 | 0.708 ± 0.019 | 0.860 ± 0.008 |
| ROAD-Waymo | BiLSTM (30,340) | 0.601 ± 0.035 | 0.835 ± 0.022 | 0.579 ± 0.031 | 0.854 ± 0.026 |
| LOKI | MLP (5,252) | 0.559 ± 0.030 | 0.573 ± 0.029 | 0.543 ± 0.019 | 0.565 ± 0.015 |
| LOKI | MLP matched (30,342) | 0.510 ± 0.013 | 0.573 ± 0.007 | 0.496 ± 0.010 | 0.579 ± 0.015 |
| LOKI | BiLSTM (30,340) | 0.386 ± 0.020 | 0.519 ± 0.020 | 0.363 ± 0.018 | 0.541 ± 0.014 |

The matched MLP lowers train CE relative to the small MLP for both sources, but does not lower selected source-validation CE. This is consistent with better training fit from extra capacity without a consistent validation-loss gain. BiLSTM training CE is lower still. These comparisons do not identify temporal context as the cause: recurrent and framewise MLP models differ in structure and optimization as well as parameterization.

![Per-class test F1 difference, BiLSTM minus matched MLP; annotations show positive seeds](../../outputs/experiments/E001d/results-review/figures/per-class-difference.png) ([SVG](../../outputs/experiments/E001d/results-review/figures/per-class-difference.svg)).

The per-class panel shows that the ROAD-source BiLSTM gains are largest for Waiting (**+21.9 ± 3.7 pp** R→R; **+9.9 ± 4.7 pp** R→L). In L→R the near-zero macro-F1 difference hides opposing class effects: Crossing improves **+4.8 ± 0.9 pp**, while Stopped averages **−5.3 ± 11.6 pp**. Neither model is uniformly better for every class and direction.

### Conclusion and limits

Increasing MLP size closes part of the original BiLSTM gap and materially improves LOKI within-domain and LOKI→ROAD means. It does not close the gap in ROAD→ROAD, ROAD→LOKI or LOKI→LOKI. In LOKI→ROAD, the primary metric is effectively tied despite a class trade-off. This is evidence that model capacity matters, but not an exact decomposition of capacity versus temporal context: equal parameter counts do not imply equal expressivity, initialization, optimization or effective capacity. Results are specific to K+T+R, these populations and one frozen split per source; they do not establish a universal temporal-model advantage or generalization uncertainty.

The reproducible review script checks the ten new attempt statuses, paired seed matrix, saved confusion-derived metrics and denominators, and equality of settings, split/setup references and source normalization against the 20 old controls. It reads saved JSON metadata/metrics only; no checkpoint was loaded and no prediction inference was run. Artifacts: [per-seed results](../../outputs/experiments/E001d/results-review/per-seed.csv), [summary and checks](../../outputs/experiments/E001d/results-review/summary.json), and [metadata snapshot with file hashes](../../outputs/experiments/E001d/results-review/metadata-snapshot.json).

Regenerate the numerical summary and both figures from the retained snapshot and original control metadata with [review-e001d-results.py](../../scripts/review-e001d-results.py):

```bash
uv run --locked --extra cpu python scripts/review-e001d-results.py
```

## Pre-run implementation check

The focused CPU test verified both parameter counts, frame independence, padding, finite backward gradients, unchanged E001 defaults and the ten-attempt matrix. CLI help and documentation links passed. The shared runner was reused; no full suite or W&B smoke was run during that preparation step.
