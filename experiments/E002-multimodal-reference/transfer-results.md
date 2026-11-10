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

## Confusion-matrix pictures

These matrices compare each selected checkpoint on source validation with its **direct target validation predictions before adaptation**. Rows are GT; columns are predictions. The class order is Moving, Stopped, Waiting to cross, Crossing. Each panel reports its matching raw or track-weighted macro-F1. `GT` is the accepted-frame denominator; `w` is track-weighted row mass, not a track count. Both row-normalized figures use the same fixed0–100% color scale.

![ROAD-Waymo to LOKI track-weighted row-normalized confusion matrices](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-track-weighted-normalized.png)

[SVG version](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/road-waymo-to-loki-track-weighted-normalized.svg).

![LOKI to ROAD-Waymo track-weighted row-normalized confusion matrices](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-track-weighted-normalized.png)

[SVG version](../../outputs/experiments/E002/transfer-validation-20261102/figures/confusions/loki-to-road-waymo-track-weighted-normalized.svg).

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

Rebuild all eight PNG/SVG grids from the frozen manifest, [collected source-validation metrics](../../outputs/experiments/E002/transfer-validation-20261102/source-validation-metrics.json) and existing target-validation metric JSONs:

```bash
MPLCONFIGDIR=/tmp/mpl-e002-transfer uv run --locked --extra cpu python scripts/plot-e002-transfer-confusions.py
```

The script checks checkpoint lineage, source F1/CE against the manifest, target population/selection identity, finite nonnegative 4×4 matrices, class supports and confusion totals. It reads saved metrics only; no inference or training is run.

## Reading the result

All eight checkpoints lose **16.6–33.0 F1 points** on the opposite validation set. The best input differs by direction: geometry + RGB + LiDAR for ROAD → LOKI (46.50%); geometry + RGB for LOKI → ROAD (46.33%). No package retains its source-validation F1 or wins both directions. The larger source-validation scores of LOKI LiDAR models do not carry over to ROAD; the LOKI geometry + LiDAR checkpoint loses 33.00 points.

CE and F1 also rank inputs differently: LiDAR gives the lowest CE for ROAD → LOKI (1.003), although fusion has the highest F1; geometry gives the lowest CE for LOKI → ROAD (1.219), although RGB has the highest F1.

This is **exploratory, target-informed validation evidence**, not a strict zero-shot result or final test. Both validation populations have already informed source-specific model selection and project development. The LOKI geometry + LiDAR checkpoint was selected during an interrupted 23-epoch training attempt. Only one seed was evaluated.

The comparison manifest, checkpoint hashes and exact source metrics are in [transfer-comparison.json](transfer-comparison.json). Full target predictions and per-cell metrics remain at `/home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation/outputs/experiments/E002/transfer-validation-20261102/`; the run finished all eight cells in **392.9 s**. One [W&B evaluation run](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/2bf6650m) contains the summary table and F1 plot; model checkpoints and predictions were not uploaded.
