# E002e — Weight-decay sensitivity

**Status:** Prepared; user will launch after the current E002d LOKI run. This is a small E002 follow-up, not a new model family.

## Question and comparison

Does stronger AdamW weight decay reduce the train/validation CE gap for the `g-rgb / temporal-small` model while preserving source-validation track-weighted macro-F1?

Run seed0 on each source with weight decay **0.01** and **0.1**, compared with the existing E002b weight-decay **0.0001** controls: [ROAD-Waymo](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/m2ov73kt), selected F1 **62.29%**, CE **0.801** (epoch9; stopped24); [LOKI](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/0cvhe1n1), selected F1 **67.63%**, CE **0.755** (epoch28; stopped43).

| Source | Arm / variant | Weight decay | Seed | Status |
|---|---|---:|---:|---|
| ROAD-Waymo | `g-rgb / temporal-small` | 0.0001 | 0 | Existing E002b control |
| ROAD-Waymo | `g-rgb / temporal-small` | 0.01 | 0 | Prepared |
| ROAD-Waymo | `g-rgb / temporal-small` | 0.1 | 0 | Prepared |
| LOKI | `g-rgb / temporal-small` | 0.0001 | 0 | Existing E002b control |
| LOKI | `g-rgb / temporal-small` | 0.01 | 0 | Prepared |
| LOKI | `g-rgb / temporal-small` | 0.1 | 0 | Prepared |

Only AdamW weight decay changes. Keep LR `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, seed0, temporal dropout0.1, geometry/RGB branch dropout0, final affine LayerNorm on both branches, model widths, frozen features, source normalization, batches, warmup, 100-epoch ceiling, patience15, clipping and source-validation checkpoint selection as in E002b. Run the four new attempts sequentially. Compare each source separately using selected F1 and CE, and inspect train/validation CE and F1 curves. This is single-seed, source-validation development; no cross-dataset or held-out test evaluation is included.

## Calibration and execution

The older independent E002b profile has stale execution hashes. The current E002d profile has the current code hash but covers only `g-dditr`, so neither is accepted for these runs. Preserve both. The launch command creates a fresh current-code E002b calibration from the latest accepted chunk-4 packing reference, then uses its `g-rgb / temporal-small` coverage. Calibration performs bounded CUDA CE/backward/AdamW checks on labeled development fixtures, then discards the resource-check weights; it does not start comparison training or a W&B run. No feature cache is rebuilt.

Use the isolated checkout at `/home/user20/projects/pedestrian-behavior-labeling-e002e` with its `outputs` symlink pointing to the canonical Aalto outputs tree. The [runnable command](../../README.md#e002e-weight-decay-sensitivity) waits for the active E002d LOKI process and runs calibration before training. Outputs remain under ignored `outputs/experiments/E002e/runs/`; one W&B run is created per attempt. Do not reuse attempt directories.

## Results

Pending the four source-validation runs. Preserve all outcomes, including failed or interrupted attempts.
