# E002 exploratory cross-dataset validation transfer

Eight frozen seed0 checkpoints were evaluated on the opposite dataset's validation split. F1 is four-class track-weighted macro-F1; CE is mean track CE. Each checkpoint's source-training normalization was applied to its target validation data. Source metrics are the full-precision selected-checkpoint records; source inference was not repeated.

| Training source → target | Inputs | Tracks source → target | Source val F1 / CE | Opposite val F1 / CE | Δ F1 (pp) | Inference (s) |
|---|---|---:|---:|---:|---:|---:|
| ROAD-Waymo → LOKI | Geometry | 1,110 → 1,861 | 54.00 / 1.160 | 37.39 / 1.480 | −16.61 | 6.6 |
| ROAD-Waymo → LOKI | Geometry + RGB | 1,110 → 1,861 | 64.37 / 0.805 | 44.85 / 1.185 | −19.52 | 10.1 |
| ROAD-Waymo → LOKI | Geometry + LiDAR | 1,110 → 1,861 | 66.05 / 0.719 | 42.42 / 1.003 | −23.63 | 68.9 |
| ROAD-Waymo → LOKI | Geometry + RGB + LiDAR | 1,110 → 1,861 | 66.07 / 0.803 | 46.50 / 1.257 | −19.57 | 77.5 |
| LOKI → ROAD-Waymo | Geometry | 1,861 → 1,110 | 64.77 / 0.520 | 42.12 / 1.219 | −22.65 | 5.4 |
| LOKI → ROAD-Waymo | Geometry + RGB | 1,861 → 1,110 | 68.20 / 0.520 | 46.33 / 1.242 | −21.87 | 5.7 |
| LOKI → ROAD-Waymo | Geometry + LiDAR | 1,861 → 1,110 | 76.41 / 0.376 | 43.41 / 1.416 | −33.00 | 89.8 |
| LOKI → ROAD-Waymo | Geometry + RGB + LiDAR | 1,861 → 1,110 | 74.68 / 0.385 | 45.02 / 1.349 | −29.66 | 97.4 |

Frozen validation support is **78 ROAD groups / 1,110 tracks / 45,388 GT frames** and **93 LOKI groups / 1,861 tracks / 60,231 GT frames**. Inference seconds cover the prediction loop, excluding feature-catalog and cache initialization.

![Within-source and opposite-source validation F1](../../outputs/experiments/E002/transfer-validation-20261102/figures/f1.png)

[SVG version](../../outputs/experiments/E002/transfer-validation-20261102/figures/f1.svg).

<!-- BEGIN generated E002 per-class metrics -->
## Per-class precision, recall and F1

These are global per-class metrics computed from each saved confusion matrix: precision uses the predicted-class column sum; recall uses the GT-class row sum; F1 is their harmonic mean. Source columns are the checkpoint's own-source validation results; target columns are direct opposite-source validation before adaptation. The main tables use the same track-weighted confusion matrices as the transfer summary. Metrics are percentages; class order matches the confusion figures.

### ROAD-Waymo → LOKI

| Inputs | Class | Source P (%) | Source R (%) | Source F1 (%) | Target P (%) | Target R (%) | Target F1 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Geometry | Moving | 70.14 | 80.09 | 74.79 | 84.78 | 63.88 | 72.86 |
| Geometry | Stopped | 66.54 | 63.94 | 65.21 | 20.60 | 81.38 | 32.88 |
| Geometry | Waiting to cross | 36.85 | 30.37 | 33.30 | 55.12 | 5.73 | 10.39 |
| Geometry | Crossing | 53.40 | 35.59 | 42.72 | 32.80 | 34.13 | 33.45 |
| Geometry + RGB | Moving | 77.63 | 78.36 | 77.99 | 90.63 | 63.45 | 74.64 |
| Geometry + RGB | Stopped | 70.85 | 71.81 | 71.33 | 23.96 | 83.98 | 37.28 |
| Geometry + RGB | Waiting to cross | 48.68 | 44.63 | 46.57 | 66.80 | 16.53 | 26.50 |
| Geometry + RGB | Crossing | 62.29 | 60.90 | 61.59 | 33.02 | 53.95 | 40.97 |
| Geometry + LiDAR | Moving | 78.95 | 82.07 | 80.48 | 83.87 | 80.98 | 82.40 |
| Geometry + LiDAR | Stopped | 72.12 | 74.30 | 73.19 | 31.55 | 62.80 | 42.00 |
| Geometry + LiDAR | Waiting to cross | 54.75 | 37.62 | 44.60 | 77.91 | 3.49 | 6.68 |
| Geometry + LiDAR | Crossing | 68.19 | 63.82 | 65.93 | 33.98 | 44.65 | 38.59 |
| Geometry + RGB + LiDAR | Moving | 80.23 | 78.63 | 79.42 | 93.75 | 52.70 | 67.47 |
| Geometry + RGB + LiDAR | Stopped | 75.10 | 67.81 | 71.27 | 20.15 | 81.27 | 32.29 |
| Geometry + RGB + LiDAR | Waiting to cross | 42.30 | 60.41 | 49.76 | 63.95 | 35.62 | 45.75 |
| Geometry + RGB + LiDAR | Crossing | 61.83 | 65.97 | 63.83 | 31.34 | 57.10 | 40.47 |

### LOKI → ROAD-Waymo

| Inputs | Class | Source P (%) | Source R (%) | Source F1 (%) | Target P (%) | Target R (%) | Target F1 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Geometry | Moving | 87.86 | 93.36 | 90.53 | 66.30 | 80.80 | 72.84 |
| Geometry | Stopped | 67.76 | 50.48 | 57.86 | 71.96 | 29.43 | 41.78 |
| Geometry | Waiting to cross | 59.12 | 74.29 | 65.84 | 13.18 | 28.36 | 18.00 |
| Geometry | Crossing | 63.32 | 34.73 | 44.86 | 39.51 | 32.84 | 35.87 |
| Geometry + RGB | Moving | 89.56 | 91.38 | 90.46 | 69.85 | 87.87 | 77.83 |
| Geometry + RGB | Stopped | 59.29 | 60.70 | 59.99 | 69.04 | 34.92 | 46.38 |
| Geometry + RGB | Waiting to cross | 64.60 | 77.02 | 70.27 | 17.29 | 46.97 | 25.28 |
| Geometry + RGB | Crossing | 65.50 | 43.25 | 52.10 | 70.20 | 24.07 | 35.85 |
| Geometry + LiDAR | Moving | 92.99 | 92.46 | 92.73 | 66.10 | 91.00 | 76.58 |
| Geometry + LiDAR | Stopped | 65.96 | 69.69 | 67.77 | 76.13 | 53.97 | 63.16 |
| Geometry + LiDAR | Waiting to cross | 74.19 | 73.06 | 73.62 | 21.67 | 31.34 | 25.62 |
| Geometry + LiDAR | Crossing | 71.31 | 71.73 | 71.52 | 51.52 | 4.51 | 8.29 |
| Geometry + RGB + LiDAR | Moving | 92.00 | 92.20 | 92.10 | 66.71 | 90.97 | 76.97 |
| Geometry + RGB + LiDAR | Stopped | 63.85 | 63.98 | 63.91 | 74.42 | 54.04 | 62.62 |
| Geometry + RGB + LiDAR | Waiting to cross | 74.63 | 72.96 | 73.79 | 25.10 | 37.72 | 30.14 |
| Geometry + RGB + LiDAR | Crossing | 68.73 | 69.11 | 68.92 | 68.64 | 5.60 | 10.36 |

<details>
<summary>Pooled-frame per-class metrics</summary>

These use raw accepted GT frames pooled across tracks rather than equal total mass per track.

### ROAD-Waymo → LOKI

| Inputs | Class | Source P (%) | Source R (%) | Source F1 (%) | Target P (%) | Target R (%) | Target F1 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Geometry | Moving | 72.96 | 84.82 | 78.44 | 81.37 | 69.29 | 74.85 |
| Geometry | Stopped | 73.46 | 64.47 | 68.67 | 21.36 | 84.36 | 34.09 |
| Geometry | Waiting to cross | 44.80 | 38.29 | 41.29 | 58.78 | 9.13 | 15.81 |
| Geometry | Crossing | 53.29 | 37.31 | 43.89 | 47.01 | 47.81 | 47.40 |
| Geometry + RGB | Moving | 78.30 | 81.76 | 79.99 | 86.87 | 65.56 | 74.72 |
| Geometry + RGB | Stopped | 76.70 | 72.17 | 74.36 | 26.69 | 84.19 | 40.53 |
| Geometry + RGB | Waiting to cross | 50.79 | 42.87 | 46.49 | 68.77 | 17.66 | 28.10 |
| Geometry + RGB | Crossing | 57.94 | 58.94 | 58.43 | 42.97 | 69.27 | 53.04 |
| Geometry + LiDAR | Moving | 78.59 | 86.29 | 82.26 | 80.51 | 81.42 | 80.96 |
| Geometry + LiDAR | Stopped | 78.47 | 73.87 | 76.10 | 27.49 | 74.08 | 40.10 |
| Geometry + LiDAR | Waiting to cross | 58.22 | 39.28 | 46.91 | 75.99 | 3.46 | 6.62 |
| Geometry + LiDAR | Crossing | 63.83 | 58.06 | 60.81 | 52.71 | 54.70 | 53.69 |
| Geometry + RGB + LiDAR | Moving | 80.70 | 81.37 | 81.03 | 92.21 | 54.33 | 68.38 |
| Geometry + RGB + LiDAR | Stopped | 82.48 | 66.46 | 73.61 | 20.61 | 80.41 | 32.81 |
| Geometry + RGB + LiDAR | Waiting to cross | 43.76 | 53.81 | 48.27 | 67.54 | 39.11 | 49.54 |
| Geometry + RGB + LiDAR | Crossing | 56.38 | 66.44 | 61.00 | 43.85 | 72.24 | 54.57 |

### LOKI → ROAD-Waymo

| Inputs | Class | Source P (%) | Source R (%) | Source F1 (%) | Target P (%) | Target R (%) | Target F1 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Geometry | Moving | 85.63 | 88.36 | 86.98 | 67.24 | 85.80 | 75.39 |
| Geometry | Stopped | 65.79 | 50.71 | 57.28 | 78.01 | 30.58 | 43.94 |
| Geometry | Waiting to cross | 67.90 | 87.51 | 76.47 | 15.64 | 26.44 | 19.66 |
| Geometry | Crossing | 68.04 | 50.64 | 58.06 | 43.26 | 30.00 | 35.43 |
| Geometry + RGB | Moving | 87.69 | 89.30 | 88.49 | 71.63 | 90.81 | 80.09 |
| Geometry + RGB | Stopped | 66.99 | 61.96 | 64.38 | 73.01 | 35.27 | 47.56 |
| Geometry + RGB | Waiting to cross | 72.10 | 91.44 | 80.63 | 20.27 | 44.32 | 27.81 |
| Geometry + RGB | Crossing | 76.78 | 57.38 | 65.68 | 69.47 | 23.02 | 34.58 |
| Geometry + LiDAR | Moving | 91.46 | 90.42 | 90.93 | 67.61 | 95.25 | 79.09 |
| Geometry + LiDAR | Stopped | 58.03 | 68.81 | 62.96 | 79.15 | 58.22 | 67.09 |
| Geometry + LiDAR | Waiting to cross | 79.57 | 79.09 | 79.33 | 22.78 | 19.54 | 21.03 |
| Geometry + LiDAR | Crossing | 77.87 | 75.55 | 76.69 | 72.99 | 4.79 | 9.00 |
| Geometry + RGB + LiDAR | Moving | 91.62 | 89.76 | 90.68 | 68.37 | 93.50 | 78.98 |
| Geometry + RGB + LiDAR | Stopped | 61.29 | 67.76 | 64.36 | 78.15 | 53.80 | 63.73 |
| Geometry + RGB + LiDAR | Waiting to cross | 80.15 | 83.35 | 81.72 | 26.70 | 31.69 | 28.98 |
| Geometry + RGB + LiDAR | Crossing | 76.96 | 76.82 | 76.89 | 65.94 | 8.40 | 14.90 |

</details>
<!-- END generated E002 per-class metrics -->

## Confusion-matrix pictures

These matrices compare each selected checkpoint on source validation with its **direct target validation predictions before adaptation**. Rows are GT; columns are predictions. The class order is Moving, Stopped, Waiting to cross, Crossing. Each panel reports its matching raw or track-weighted macro-F1. `GT` is the accepted-frame denominator; `w` is track-weighted row mass, not a track count. Both row-normalized figures use the same fixed 0–100% color scale.

In a count matrix `C`, class recall is the diagonal entry divided by its **row sum**: `C[c,c] / sum(C[c,:])`. The row-normalized plots have already performed that division, so each diagonal percentage is class recall. Precision uses the **column sum** instead: `C[c,c] / sum(C[:,c])`. A count-view diagonal by itself is a true-positive count or weight, not recall. These plots show predicted-label frequencies, not softmax confidence.

Dataset arrows such as **LOKI → ROAD** indicate training dataset → evaluation dataset. Classification errors are written explicitly as **GT Stopped, predicted Moving**: these are individual timesteps, not tracks undergoing a temporal behavior transition. In track-weighted plots, a row percentage is a share of weighted GT timesteps, not a percentage of tracks.

![ROAD-Waymo to LOKI track-weighted row-normalized confusion matrices](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-track-weighted-normalized.png)

[Row-normalized SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-track-weighted-normalized.svg). **Without row normalization:** [track-weighted counts SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-track-weighted-count.svg) · [frame counts SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-raw-count.svg).

![LOKI to ROAD-Waymo track-weighted row-normalized confusion matrices](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-track-weighted-normalized.png)

[Row-normalized SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-track-weighted-normalized.svg). **Without row normalization:** [track-weighted counts SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-track-weighted-count.svg) · [frame counts SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-raw-count.svg).

<details>
<summary>Raw and count confusion views</summary>

Each count figure uses one color range across its source and target panels. Raw counts are GT frames; track-weighted counts are weighted GT mass and sum to the number of evaluated tracks.

![ROAD-Waymo to LOKI raw row-normalized confusion matrices](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-raw-normalized.png)

[SVG version](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-raw-normalized.svg).

![LOKI to ROAD-Waymo raw row-normalized confusion matrices](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-raw-normalized.png)

[SVG version](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-raw-normalized.svg).

| Direction | Weighting | View | PNG | SVG |
|---|---|---|---|---|
| ROAD-Waymo → LOKI | Raw | Counts | [PNG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-raw-count.png) | [SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-raw-count.svg) |
| ROAD-Waymo → LOKI | Raw | Row-normalized | [PNG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-raw-normalized.png) | [SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-raw-normalized.svg) |
| ROAD-Waymo → LOKI | Track-weighted | Counts | [PNG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-track-weighted-count.png) | [SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-track-weighted-count.svg) |
| LOKI → ROAD-Waymo | Raw | Counts | [PNG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-raw-count.png) | [SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-raw-count.svg) |
| LOKI → ROAD-Waymo | Raw | Row-normalized | [PNG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-raw-normalized.png) | [SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-raw-normalized.svg) |
| LOKI → ROAD-Waymo | Track-weighted | Counts | [PNG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-track-weighted-count.png) | [SVG](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-track-weighted-count.svg) |

</details>

Rebuild all eight PNG/SVG grids and the [per-class metric snippet](../../outputs/experiments/E002/transfer-validation-20261102/figures/per-class-metrics.md) from the frozen manifest, [collected source-validation metrics](../../outputs/experiments/E002/transfer-validation-20261102/source-validation-metrics.json) and existing target-validation metric JSONs:

```bash
MPLCONFIGDIR=/tmp/mpl-e002-transfer uv run --locked --extra cpu python scripts/plot-e002-transfer-confusions.py
```

The script checks checkpoint lineage, source F1/CE against the manifest, target population/selection identity, finite nonnegative 4×4 matrices, class supports and confusion totals. It recalculates each class precision/recall/F1 from both raw and track-weighted matrices, checks the saved per-class metrics and macro-F1, and writes the linked Markdown snippet. It reads saved metrics only; no inference or training is run.

## Reading the result

All eight checkpoints lose **16.6–33.0 F1 points** on the opposite validation set. The best input differs by direction: geometry + RGB + LiDAR for ROAD → LOKI (46.50%); geometry + RGB for LOKI → ROAD (46.33%). No package retains its source-validation F1 or wins both directions. The larger source-validation scores of LOKI LiDAR models do not carry over to ROAD; the LOKI geometry + LiDAR checkpoint loses 33.00 points.

CE and F1 also rank inputs differently: LiDAR gives the lowest CE for ROAD → LOKI (1.003), although fusion has the highest F1; geometry gives the lowest CE for LOKI → ROAD (1.219), although RGB has the highest F1.

This is **exploratory, target-informed validation evidence**, not a strict zero-shot result or final test. Both validation populations have already informed source-specific model selection and project development. The LOKI geometry + LiDAR checkpoint was selected during an interrupted 23-epoch training attempt. Only one seed was evaluated.

The comparison manifest, checkpoint hashes and exact source metrics are in [transfer-comparison.json](transfer-comparison.json). Full target predictions and per-cell metrics remain at `/home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation/outputs/experiments/E002/transfer-validation-20261102/`; the run finished all eight cells in **392.9 s**. One [W&B evaluation run](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/2bf6650m) contains the summary table and F1 plot; model checkpoints and predictions were not uploaded.

## Focused confusion audit (2026-11-10)

All eight direct-transfer files align by locator, GT and mask with the own-source predictions for the same target dataset; both E003b E002d adapted-head files also align. Detailed examples below use the full-fusion E002d models. Percentages and quantiles retain each accepted GT timestep's weight `1 / track_GT_count`. Speed is derived from saved unnormalized 3D positions/velocities; unavailable velocities are excluded rather than treated as zero. The 0.2/0.5 m/s diagnostic bins do not redefine native labels.

- **Stopped recall improves in three modalities, while precision can deteriorate.** For the ROAD-trained fusion model, Stopped recall is 67.8% on ROAD validation and 81.3% on LOKI validation. Stopped precision is 75.1% and 20.1%, respectively; its F1 is 71.3% and 32.3%. These are different validation populations. On LOKI, the model predicts Stopped on 35.5% of all weighted GT timesteps, while only 8.8% have GT Stopped. It predicts Stopped on 28.0% of GT Moving timesteps. The LiDAR-only model is an exception to the higher-Stopped-recall pattern.
- **Observation availability is associated with several errors.** For the ROAD-trained fusion model evaluated on LOKI,37.9% of GT Moving timesteps with no usable RGB crop are predicted Stopped, versus 10.0% with a crop. Among all its GT Moving timesteps predicted Stopped,92.9% of weighted mass has valid 3D-derived velocity; the median speed within that valid subset is 1.112 m/s. For the LOKI-trained fusion model evaluated on ROAD,39.0% of GT Stopped timesteps without a usable LiDAR descriptor are predicted Moving, versus 24.2% with one. For GT Waiting timesteps, those percentages are 61.3% and 33.1%. Missing LiDAR and missing pedestrian velocity overlap, so these are not isolated causal effects of the LiDAR branch.
- **Some apparently contradictory classification errors agree with the input geometry.** For LOKI-trained fusion evaluated on ROAD, select timesteps with **GT Stopped and predicted Moving**. Velocity is valid on 81.0% of that group's weighted mass; within the valid subset the median speed is 0.884 m/s and 66.0% exceeds 0.5 m/s. For GT Stopped timesteps correctly predicted Stopped, velocity coverage is 90.3%, median speed 0.002 m/s and 1.0% exceeds 0.5. For GT Waiting timesteps predicted Moving, velocity coverage is 68.3%, median speed 0.234 m/s and 29.0% exceeds 0.5 within the valid subset.
- **GT Crossing classified as Moving is a different distinction.** For LOKI-trained fusion evaluated on ROAD, these timesteps have median speed 1.407 m/s versus 1.338 for GT Crossing correctly predicted Crossing. The percentage of GT Crossing predicted Moving is 92.6% with usable LiDAR and 73.4% without; missing LiDAR alone cannot explain this collapse. Geometry-only predicts Moving on 64.4% of GT Crossing timesteps in LOKI source validation and 64.9% in ROAD target validation. For geometry+LiDAR, those values are 27.7% and 93.4%. This shows loss of the source scene model's advantage, without isolating the frozen encoder, learned readout, classifier or annotation semantics.

For a sustained-motion check, require seven consecutive GT-Stopped, position-valid slots and use actual endpoint timestamps (approximately 1.2 s). **GT remains Stopped throughout each window; only the prediction at its center selects the error or correct group.** Among centers predicted Moving,1,498 eligible centers across 77 tracks have median net speed **0.688 m/s**; 4,106 centers across 167 tracks predicted Stopped have**0.006 m/s**. These windows cover 9.5% and 29.8% of all weighted ROAD Stopped GT respectively, not the entire error population; overlapping windows are not independent examples. The observation warrants checking native RGB, the matched 3D actor and annotation boundaries. It does not establish annotation error or independently verified physical motion.

The [2026-11-11 inspection shortlist](../../outputs/experiments/E002/transfer-confusion-diagnostic-20261111/inspection-cases-and-masking.json) pins exact tracks/windows in ROAD clips `train_00261`, `train_00111`, `train_00626` and `train_00593`, including source-model predictions and a zero-displacement control. These are selected examples, not prevalence estimates or confirmed annotation errors. A saved-prediction [E003a subgroup check](../E003a-structured-observation-masking/README.md#natural-validation-comparison) also finds that masking reduces GT Moving classified as Stopped when RGB is missing, despite worse overall ROAD→LOKI F1.

E003b head-only adaptation changes predicted labels on the same target-validation timesteps, with trade-offs:

| Training / evaluation datasets | GT label | Predicted label | Before head adaptation (%) | After head adaptation (%) |
|---|---|---|---:|---:|
| ROAD / LOKI | Moving | Stopped |28.0|2.1|
| ROAD / LOKI | Waiting to cross | Stopped |57.1|17.8|
| ROAD / LOKI | Crossing | Moving |14.7|66.0|
| LOKI / ROAD | Crossing | Moving |91.1|52.2|
| LOKI / ROAD | Waiting to cross | Stopped |21.1|39.0|

Each percentage uses the weighted timesteps of the row's **GT class** as denominator. These are classification confusions, not observed behavior transitions. Recovery demonstrates accessible target information, not one universal fix. Different class frequencies can affect precision, but cannot alone explain the changing per-GT-class error rates. No single cause is established.

[Numerical diagnostic](../../outputs/experiments/E002/transfer-confusion-diagnostic-20261110/summary.json). Reproduce on Aalto from the repository root with [analyze-e002-transfer-confusions.py](../../scripts/analyze-e002-transfer-confusions.py), using the existing E002a extractor Python environment and saved predictions/track arrays. No training, inference, native point-cache access, label correction or test-set access was performed.
