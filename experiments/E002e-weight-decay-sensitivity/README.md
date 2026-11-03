# E002e — Weight-decay sensitivity

**Status:** Original four runs completed 2026-11-03; two weight-decay1.0 follow-ups prepared 2026-11-03. This is a small E002 follow-up, not a new model family.

## Question and comparison

Does stronger AdamW weight decay reduce the train/validation CE gap for the `g-rgb / temporal-small` model while preserving source-validation track-weighted macro-F1?

Run seed0 on each source with weight decay **0.01** and **0.1**, compared with the existing E002b weight-decay **0.0001** controls: [ROAD-Waymo](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/m2ov73kt), selected F1 **62.29%**, CE **0.801** (epoch9; stopped24); [LOKI](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/0cvhe1n1), selected F1 **67.63%**, CE **0.755** (epoch28; stopped43).

| Source | Arm / variant | Weight decay | Seed | Status |
|---|---|---:|---:|---|
| ROAD-Waymo | `g-rgb / temporal-small` | 0.0001 | 0 | Existing E002b control |
| ROAD-Waymo | `g-rgb / temporal-small` | 0.01 | 0 | Complete |
| ROAD-Waymo | `g-rgb / temporal-small` | 0.1 | 0 | Complete |
| LOKI | `g-rgb / temporal-small` | 0.0001 | 0 | Existing E002b control |
| LOKI | `g-rgb / temporal-small` | 0.01 | 0 | Complete |
| LOKI | `g-rgb / temporal-small` | 0.1 | 0 | Complete |

Only AdamW weight decay changes. Keep LR `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, seed0, temporal dropout0.1, geometry/RGB branch dropout0, final affine LayerNorm on both branches, model widths, frozen features, source normalization, batches, warmup, 100-epoch ceiling, patience15, clipping and source-validation checkpoint selection as in E002b. Run the four new attempts sequentially. Compare each source separately using selected F1 and CE, and inspect train/validation CE and F1 curves. This is single-seed, source-validation development; no cross-dataset or held-out test evaluation is included.

## Calibration and execution

The older independent E002b profile has stale execution hashes. The current E002d profile has the current code hash but covers only `g-dditr`, so neither was accepted. The fresh E002b calibration at Aalto `outputs/experiments/E002b/calibration-e002e-wd-20261102-185738/profile.json` (SHA-256 `211376f40ccb6d5fcc3458df4b584b0b4619966a3b8163d362855cda4775aef1`, code revision `5fa9bb6`) includes `g-rgb / temporal-small`. It uses the accepted chunk-4 packing reference (SHA-256 `8a963b9fb0f36157e120f738dfa8170ad025c16b5c0e047c74e6d82a8f8b59a1`). Calibration performed bounded CUDA CE/backward/AdamW checks on labeled development fixtures, then discarded the resource-check weights; it did not start comparison training or a W&B run. No feature cache was rebuilt.

Use the isolated checkout at `/home/user20/projects/pedestrian-behavior-labeling-e002e` with its `outputs` symlink pointing to the canonical Aalto outputs tree. The [runnable command](../../README.md#e002e-weight-decay-sensitivity) waits for the active E002d LOKI process and runs calibration before training. Outputs remain under ignored `outputs/experiments/E002e/runs/`; one W&B run is created per attempt. Do not reuse attempt directories.

## Results

All four attempts completed and stopped by patience. Their source-validation populations match the controls: ROAD-Waymo **78 groups / 1,110 tracks / 45,388 GT frames**; LOKI **93 groups / 1,861 tracks / 60,231 GT frames**. All six reach their minimum validation CE at epoch5, with near-identical values, before later CE divergence.

| Source | Weight decay / W&B run | Selected / stopped epoch | Selected F1 / CE | Final train CE / F1 | Final validation CE / F1 | Minimum validation CE |
|---|---|---:|---:|---:|---:|---:|
| ROAD-Waymo | 0.0001 control [m2ov73kt](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/m2ov73kt) | 9 / 24 | 62.29% / 0.80060 | 0.10523 / unavailable | 1.34996 / 59.76% | 0.75679 (epoch5) |
| ROAD-Waymo | 0.01 [rh865rsq](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/rh865rsq) | 9 / 24 | 62.29% / 0.80024 | 0.10490 / 95.32% | 1.37336 / 57.14% | 0.75680 (epoch5) |
| ROAD-Waymo | 0.1 [mwddazs0](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/mwddazs0) | 9 / 24 | 62.26% / 0.79698 | 0.10934 / 94.69% | 1.32154 / 59.23% | 0.75690 (epoch5) |
| LOKI | 0.0001 control [0cvhe1n1](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/0cvhe1n1) | 28 / 43 | 67.63% / 0.75453 | 0.10406 / unavailable | 0.92840 / 65.36% | 0.51468 (epoch5) |
| LOKI | 0.01 [ygx96v4f](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ygx96v4f) | 19 / 34 | 66.86% / 0.68344 | 0.12701 / 92.73% | 0.82069 / 65.55% | 0.51456 (epoch5) |
| LOKI | 0.1 [guc3873l](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/guc3873l) | 28 / 43 | 67.05% / 0.74248 | 0.11884 / 93.17% | 0.86625 / 65.55% | 0.51343 (epoch5) |

Training CE and F1 are online epoch aggregates with changing weights and active dropout, not an evaluation of one fixed checkpoint. Historical E002b controls predate train-F1 logging, so their train F1 is unavailable. Every run's `config.json`, `provenance.json`, `history.json` and validation metrics are preserved under Aalto's `outputs/experiments/E002e/runs/wd-20261102-185738-*`; the matched controls are under `outputs/experiments/E002b/runs/independent-20261025-*`.

**Readout:** Selected F1 is essentially unchanged on ROAD (−0.02 pp at weight decay0.1) and lower on LOKI (−0.59 to −0.77 pp). Higher decay does not improve the selected classification result. It does reduce late CE divergence for weight decay0.1: at the matched final epoch43 on LOKI, validation CE is0.866 versus0.928 for the control while validation F1 is similar; at ROAD's matched epoch24, validation CE is1.322 versus1.350, also with nearly unchanged F1. The final validation-minus-training CE gap is ROAD1.245 for the control,1.268 at0.01 and1.212 at0.1; on LOKI it is0.824 for the control versus0.747 at0.1, both at epoch43. Weight decay0.01 stopped at LOKI epoch34; at that matched epoch its CE gap is0.694 versus0.675 for the control, so it did not reduce divergence there. E002e's train F1 reaches92.7–95.3% while final validation F1 is57.1–65.6%; this gap remains large. The small CE-gap change did not produce a meaningful selected-F1 gain, and this is single-seed source-validation evidence only. F1 differences are calculated from unrounded saved metrics; displayed F1 values are rounded to two decimals.

The four-panel training/validation CE and F1 figure overlays the three weight-decay settings for each source and notes that historical controls have no training-F1 trace. Aalto artifacts: `/home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E002e/results-20261103/weight-decay-curves.png` and `.svg`.

## Comparison with dropout

Across the matched `g-rgb / temporal-small` seed0 runs, RGB descriptor dropout0.5 gives the clearest selected-F1 gain: **+2.08 pp ROAD / +0.57 pp LOKI**, with nearly unchanged selected CE on ROAD and lower CE on LOKI. Temporal dropout0.5 gives **+0.74 / −1.39 pp**; neither higher weight-decay setting improves selected F1. RGB dropout softens CE divergence, but substantial train/validation gaps remain. These are separate interventions, not combined settings; single-seed validation does not establish a reliable winner.

![Selected source-validation F1 and CE for weight decay and dropout](../../outputs/experiments/E002-regularization-comparison-20261103/selected-metrics.png)

The curves show the control and the largest tested value of each knob. Dashed training traces use changing weights and active dropout; solid traces are source validation. Historical controls have no train-F1 trace. Runs stop at different epochs, so their terminal gaps are not matched-exposure comparisons.

![Training and validation curves for the control and three regularization settings](../../outputs/experiments/E002-regularization-comparison-20261103/training-validation-curves.png)

[Full comparison table](../E002-multimodal-reference/README.md#single-axis-regularization-comparison-2026-11-03) · [Selected-metrics SVG](../../outputs/experiments/E002-regularization-comparison-20261103/selected-metrics.svg) · [Curves SVG](../../outputs/experiments/E002-regularization-comparison-20261103/training-validation-curves.svg).

## Additional high-decay check (2026-11-03)

The user requested one final setting after reviewing the original comparison. Run AdamW weight decay **1.0** on the same `g-rgb / temporal-small` seed0 model, once per source. Keep LR `3e-4`, temporal dropout0.1, geometry/RGB branch dropout0, optimizer betas/epsilon, model, caches, source normalization and all other E002b settings fixed. This adds two source-validation attempts; it does not replace or change the completed0.01/0.1 runs and does not access held-out tests.

| Source | Arm / variant | Weight decay | Seed | Status |
|---|---|---:|---:|---|
| ROAD-Waymo | `g-rgb / temporal-small` | 1.0 | 0 | Ready for user launch |
| LOKI | `g-rgb / temporal-small` | 1.0 | 0 | Ready for user launch |

Reuse Aalto's existing accepted profile `outputs/experiments/E002b/calibration-e002e-wd-20261102-185738/profile.json` (SHA-256 `211376f40ccb6d5fcc3458df4b584b0b4619966a3b8163d362855cda4775aef1`) and RGB preparation `outputs/experiments/E002a/rgb-cache-preparation-03/cache.json`. The model/execution modules covered by calibration are unchanged; this option only supplies a different copied optimizer weight-decay value to `e002e_run.py`. The [two-run command](../../README.md#e002e-weight-decay-10-follow-up) runs ROAD then LOKI in fresh output directories. No training has been started for this follow-up.
