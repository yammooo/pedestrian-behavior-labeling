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

## Focused confusion audit (2026-11-10)

All eight direct-transfer files align by locator, GT and mask with the own-source predictions for the same target dataset; both E003b E002d adapted-head files also align. Detailed examples below use the full-fusion E002d models. Percentages and quantiles retain each accepted GT timestep's weight `1 / track_GT_count`. Speed is derived from saved unnormalized 3D positions/velocities; unavailable velocities are excluded rather than treated as zero. The0.2/0.5 m/s diagnostic bins do not redefine native labels.

- **A higher diagonal is recall, not overall class quality.** ROAD→LOKI Stopped recall rises67.8→81.3%, but precision falls75.1→20.1% and F1 falls71.3→32.3%. Stopped predictions occupy35.5% of target weighted mass versus8.8% GT. This is broad Stopped overprediction, including28.0% of GT Moving, not only Waiting/Stopped confusion. The LiDAR-only model is an exception to the higher-Stopped-recall pattern.
- **Observation availability is associated with several errors.** ROAD→LOKI Moving→Stopped occurs on37.9% of GT Moving with no usable RGB crop versus10.0% with one. Those false-Stopped cases still have valid 3D-derived velocity on92.9% of their weighted mass, with median speed1.112 m/s. LOKI→ROAD Stopped→Moving is39.0% with no usable LiDAR descriptor versus24.2% with one; Waiting→Moving is61.3% versus33.1%. Missing LiDAR and missing pedestrian velocity overlap, so these are not isolated causal effects of the LiDAR branch.
- **Some apparently contradictory motion errors agree with the input geometry.** LOKI→ROAD Stopped→Moving has valid velocity on81.0% of its weighted mass; within that valid subset median speed is0.884 m/s and66.0% exceeds0.5 m/s. Correct Stopped has90.3% velocity coverage, median0.002 m/s and1.0% above0.5. Waiting→Moving is mixed:68.3% velocity coverage, median0.234 m/s and29.0% above0.5 among valid observations.
- **Crossing→Moving is a different distinction.** LOKI→ROAD fusion errors have median speed1.407 m/s versus1.338 for correct Crossing. Error rates are92.6% with usable LiDAR versus73.4% without; missing LiDAR alone cannot explain this collapse. Geometry-only already confuses Crossing with Moving on source validation (64.4→64.9% source/target), whereas geometry+LiDAR changes27.7→93.4%. This shows loss of the source scene model's advantage, without isolating the frozen encoder, learned readout, classifier or annotation semantics.

For a sustained-motion check, require seven consecutive GT-Stopped, position-valid slots and use actual endpoint timestamps (approximately1.2 s). Among centers predicted Moving,1,498 eligible centers across77 tracks have median net speed**0.688 m/s**;4,106 centers across167 tracks predicted Stopped have**0.006 m/s**. These windows cover9.5% and29.8% of all weighted ROAD Stopped GT respectively, not the entire error population; overlapping windows are not independent examples. The observation warrants checking native RGB, the matched 3D actor and annotation boundaries. It does not establish annotation error or independently verified physical motion.

Changing only the head moves these boundaries substantially, with trade-offs: ROAD→LOKI Moving→Stopped falls28.0→2.1% and Waiting→Stopped57.1→17.8%, but Crossing→Moving rises14.7→66.0%. LOKI→ROAD Crossing→Moving falls91.1→52.2%, while Waiting→Stopped rises21.1→39.0%. Recovery demonstrates accessible target information, not one universal fix. Different class frequencies can affect precision, but cannot alone explain the changing per-GT-class error rates. No single cause is established.

[Numerical diagnostic](../../outputs/experiments/E002/transfer-confusion-diagnostic-20261110/summary.json). Reproduce on Aalto from the repository root with [analyze-e002-transfer-confusions.py](../../scripts/analyze-e002-transfer-confusions.py), using the existing E002a extractor Python environment and saved predictions/track arrays. No training, inference, native point-cache access, label correction or test-set access was performed.
