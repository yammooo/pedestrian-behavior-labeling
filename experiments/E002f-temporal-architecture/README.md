# E002f — Temporal architecture on ROAD→LOKI

**Status: full seed0 comparison complete, 2026-11-11; validation only.** This comparison asks whether a one-layer Transformer or a one-layer BiLSTM changes source-trained and target-adapted behavior labeling when the shared E002d fusion input and training budgets stay fixed. It covers ROAD-Waymo → LOKI only. No reverse transfer or additional temporal variants are in scope.

## Fixed comparison

Both models use E002d's `g-rgb-dditr` input: geometry/RGB/LiDAR branches, branch LayerNorm, RGB descriptor dropout0.5, and trainable spatial readout, fusion, temporal module, final LayerNorm and classifier. DINO and D-DITR features are precomputed, frozen inputs. ROAD statistics normalize both source training/validation and LOKI train/validation. Source training selects on ROAD validation; direct LOKI validation is then recorded before target adaptation.

| Architecture | Temporal module | Temporal + final norm + 4-class head |
|---|---|---:|
| `transformer-1layer` | One 128-wide, 4-head RoPE block; FFN512; dropout0.1 |199,044 |
| `bilstm-96` | Packed one-layer BiLSTM, hidden96/direction; output dropout0.1; 192→128 projection |199,044 |
| Existing two-layer reference | Two 128-wide, 4-head RoPE blocks; FFN512; dropout0.1 |397,316 |

The existing [E002d source model](../E002d-rgb-lidar-complementarity/README.md) and [E003c source-initialized adaptations](../E003c-temporal-adaptation/README.md#full-source-initialized-results-2026-11-06) supply the two-layer reference; the launcher does not retrain it. Its ROAD→LOKI validation F1 is46.50% before adaptation,63.90% after head-only fitting and68.17% after temporal/head fitting. Wider historical two-layer models are not additional launcher arms.

The single-layer LSTM uses ordinary PyTorch biases; complete-track lengths are packed so backward recurrence never processes padded suffixes. Both fusion outputs are128-D. Each whole source model has552,196 trainable parameters, including the shared trainable branches/readout.

Each source-best checkpoint gets two independently initialized LOKI fits:

| Arm | Target features | Trainable parameters |
|---|---|---:|
| `head` | Frozen source temporal backbone + final LayerNorm outputs |516 |
| `temporal-head` | Architecture-specific pre-temporal fusion output |199,044 |

Both arms copy the exact original source classifier; the temporal arm also copies the selected source temporal module and final norm. No target-adapted head is reused. Source branch/fusion weights and input encoders remain frozen during target adaptation. Epoch0 target logits must reproduce direct source predictions on the same validation population.

## Recipes and access

Source fits use seed0, float32, batch64, AdamW (`lr=3e-4`, weight decay`1e-4`, betas`0.9/0.999`, epsilon`1e-8`), three-epoch warmup, global clip1, maximum100 epochs and patience15. The inherited E002d input profile retains its accepted complete-track and point-token bounds. The E002f calibration separately checks both backbones at64×102 through forward, masked track CE, backward and optimizer update. Actual source smokes exercise the prepared RGB and indexed-point paths.

Target fits use the E003c recipe: all8,574 LOKI training tracks and1,861 validation tracks; whole-track batches64; AdamW with weight decay`1e-4`; head LR`3e-4`; temporal+final-norm LR`3e-5`; fresh optimizer per arm; three-epoch warmup; clip1; maximum100 epochs; patience15; dropout disabled. Epoch0 is a checkpoint candidate. Selection maximizes track-weighted four-class macro-F1, breaks exact F1 ties by lower CE and then earlier epoch; patience resets only on strict F1 improvement. Training and validation use GT accepted by the existing target policy. No test split is accessed.

This is full target-label training and target-validation selection, not zero-shot or held-out transfer. E002f arms are paired within each source checkpoint but are not matched to a different architecture's source checkpoint beyond the fixed seed/recipe. Results can support a descriptive comparison under this protocol, not a causal claim that temporal architecture alone explains transfer differences.

## Run

On Aalto, from the active checkout, the sequential launcher calibrates both temporal shapes and then performs the two source fits, each architecture's fresh LOKI cache, and four target fits:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
bash scripts/run-e002f-sequential.sh
```

Bounded integration acceptance uses the existing explicit E003a three-train/two-validation track selection and creates separate smoke outputs:

```bash
E002F_SMOKE=1 bash scripts/run-e002f-sequential.sh
```

The launcher uses `outputs/experiments/E002a/extractor-env/bin/python`, prepared read-only caches, fresh timestamped output roots and one W&B run per fit. It stops on failure; it does not resume or retry. `E002F_RUN_ROOT`, `E002F_FEATURE_ROOT`, `E002F_PROFILE`, `E002F_PYTHON`, `E002F_DEVICE`, `E002F_SMOKE_SELECTION`, `E002F_PROBE`, `E002F_RGB_CACHE_ROOT`, and `E002F_RGB_PREPARATION` can select explicit paths. A non-complete existing profile or output root is rejected.

Standalone source, extraction and adaptation commands are available through `scripts/train-e002f.py`; resource calibration is `scripts/calibrate-e002f.py`. Inputs, caches and source checkpoints remain read-only. Full training is user-launched only.

## Acceptance and results

At revision `76589ff`, the full CPU suite passes272 tests with13 expected CUDA skips; all272 CUDA tests pass. Both backbones pass64×102 forward/masked-CE/backward/AdamW probes. These synthetic probes omit point tokens; actual read-only RGB/LiDAR paths are exercised by the source smokes. They verify execution bounds, not optimal GPU utilization or full-epoch runtime.

All six one-epoch GPU/W&B smokes complete with empty failure lists, using three training and two validation tracks per dataset. Source checkpoint comparisons confirm updates in geometry, RGB, spatial readout, fusion and temporal parameters. Saved predictions reconstruct selected metrics; both adaptation arms reproduce their source model's direct target predictions at epoch0 and retain ROAD normalization. Best/last checkpoints and PNG/SVG plots exist. W&B readback includes step loss, epoch CE/F1 and aggregate tables/plots; no model/checkpoint or full-prediction uploads. Original E002d checkpoint and protected inputs remain unchanged.

| Architecture | Source smoke | Head smoke | Temporal/head smoke |
|---|---|---|---|
| Transformer ×1 |[9dgb5s09](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/9dgb5s09)|[jp9x5dr3](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/jp9x5dr3)|[2sjb6pix](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/2sjb6pix)|
| BiLSTM |[ppyqf4ey](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ppyqf4ey)|[ch2j6chl](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ch2j6chl)|[woh9j6zm](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/woh9j6zm)|

Aalto acceptance evidence: `outputs/experiments/E002f/acceptance-20261110/{cuda-tests.log,smoke.log,verification.json}` and `calibration/acceptance-20261110/profile.json`. The six full E002f fits completed with empty failure lists; selected/stopped epochs and W&B runs are in the table. Source training processes 4,380 ROAD tracks and evaluates all 1,110 ROAD validation tracks. Target adaptation uses all 8,574 LOKI training tracks and selects on all 1,861 LOKI validation tracks. No held-out test split was accessed. Full runs and feature caches remain on Aalto under `outputs/experiments/E002f/{runs,features}/temporal-architecture-20261110-091857/`.

## Full results (2026-11-11)

Cells show track-weighted four-class macro-F1 (%) / mean track CE. Epochs in parentheses are selected / last completed; direct transfer uses the source checkpoint's selected epoch. The two-layer reference is the E002d RGB-dropout0.5 source checkpoint followed by the original-source E003c adaptations, not the E003b fresh-head runs.

| Architecture | Temporal + norm + head parameters | ROAD source validation | LOKI direct | LOKI head-only | LOKI temporal + head |
|---|---:|---:|---:|---:|---:|
| Transformer ×1 |199,044 |[65.11 /0.795 (e16/31)](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/iy94qap8) |44.03 /1.174 (e16) |[63.09 /0.530 (e27/42)](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/rip98fjj) |[66.72 /0.466 (e21/36)](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/mkiu997e) |
| BiLSTM |199,044 |[65.43 /0.890 (e28/43)](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/qbd56oyy) |45.01 /1.171 (e28) |[64.56 /0.500 (e59/74)](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/4k4gnen5) |[67.73 /0.461 (e35/50)](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/hfvk38it) |
| Transformer ×2 reference |397,316 |[66.07 /0.803 (e17/32)](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/9k11kn3b) |46.50 /1.257 (e17) |[63.90 /0.520 (e54/69)](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/27vi7vgw) |[68.17 /0.525 (e45/60)](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/h1yg4q96) |

Both one-layer architectures finish within 1.45 percentage points F1 of the two-layer temporal-adaptation reference. Temporal + head adaptation exceeds head-only F1 for each architecture (+3.17 pp BiLSTM, +3.64 pp Transformer ×1; +4.27 pp for the two-layer E003c reference). Direct LOKI transfer is weakest for Waiting to cross: per-class F1 is 24.68% for Transformer ×1 and 31.41% for BiLSTM, versus 45.75% for the two-layer source model. The target-adapted per-class scores, including precision and recall, are in the [per-class tables](../../outputs/experiments/E002f/results-review/per-class-metrics.md).

**Overfitting:** selected ROAD online-training/validation F1 is79.08/65.11% for Transformer ×1,86.59/65.43% for BiLSTM, and82.74/66.07% for Transformer ×2. All retain a substantial source gap. After temporal adaptation, those pairs are71.85/66.72%,79.73/67.73% and84.45/68.17% respectively: the one-layer Transformer has a smaller target gap and both smaller models have lower target CE, while the two-layer model retains slightly higher F1. Capacity affects fitting, but this comparison does not show that changing architecture resolves direct transfer.

![Selected validation F1 and CE](../../outputs/experiments/E002f/results-review/figures/summary-f1-ce.png) ([Summary SVG](../../outputs/experiments/E002f/results-review/figures/summary-f1-ce.svg))

![Online training and validation curves](../../outputs/experiments/E002f/results-review/figures/training-curves.png) ([Curves SVG](../../outputs/experiments/E002f/results-review/figures/training-curves.svg))

![Source, direct and adapted validation confusion matrices](../../outputs/experiments/E002f/results-review/figures/validation-confusions-track-weighted-row-normalized.png) ([Confusion SVG](../../outputs/experiments/E002f/results-review/figures/validation-confusions-track-weighted-row-normalized.svg))

GT rows and prediction columns are used throughout. In the row-normalized plot, the diagonal is recall; precision uses predicted-class column totals in the original matrix. Raw counts total45,388 ROAD /60,231 LOKI accepted GT frames; track-weighted counts total1,110 /1,861, with each complete track contributing one unit. Additional [track-weighted counts](../../outputs/experiments/E002f/results-review/figures/validation-confusions-track-weighted-count.svg), [raw-frame row-normalized matrices](../../outputs/experiments/E002f/results-review/figures/validation-confusions-raw-row-normalized.svg) and [raw-frame counts](../../outputs/experiments/E002f/results-review/figures/validation-confusions-raw-count.svg) are linked as SVGs; PNGs are saved alongside them. The [summary JSON](../../outputs/experiments/E002f/results-review/summary.json) retains both matrix weightings and per-class scores.

These are descriptive validation results, not a causal architecture comparison: the two-layer reference was trained in E002d and adapted in E003c. All adaptation fits use full target labels and target-validation selection; held-out tests remain closed. Training curves use online metrics from changing weights (source dropout active, adaptation dropout disabled). P/R/F1 and macro-F1 are checked against the saved confusion matrices; predictions were not re-evaluated for this review.

The run metadata auto-records revision `6424f81` from the separate outputs-root checkout, while its program path points to the evaluation checkout. That checkout is currently clean at `65620d2`, but the exact runtime code revision is not independently pinned in the run record.

<details>
<summary>Per-class precision, recall and F1 (track-weighted, %)</summary>

### ROAD source validation · ROAD-Waymo

| Architecture | Class | Precision (%) | Recall (%) | F1 (%) |
|---|---|---:|---:|---:|
| Transformer ×2 | Moving | 80.23 | 78.63 | 79.42 |
| Transformer ×2 | Stopped | 75.10 | 67.81 | 71.27 |
| Transformer ×2 | Waiting to cross | 42.30 | 60.41 | 49.76 |
| Transformer ×2 | Crossing | 61.83 | 65.97 | 63.83 |
| BiLSTM | Moving | 77.82 | 79.53 | 78.67 |
| BiLSTM | Stopped | 69.26 | 68.70 | 68.98 |
| BiLSTM | Waiting to cross | 47.49 | 50.74 | 49.06 |
| BiLSTM | Crossing | 67.97 | 62.34 | 65.03 |
| Transformer ×1 | Moving | 78.74 | 79.49 | 79.11 |
| Transformer ×1 | Stopped | 74.13 | 67.16 | 70.47 |
| Transformer ×1 | Waiting to cross | 43.21 | 53.56 | 47.83 |
| Transformer ×1 | Crossing | 61.96 | 64.11 | 63.02 |

### LOKI direct transfer · LOKI

| Architecture | Class | Precision (%) | Recall (%) | F1 (%) |
|---|---|---:|---:|---:|
| Transformer ×2 | Moving | 93.75 | 52.70 | 67.47 |
| Transformer ×2 | Stopped | 20.15 | 81.27 | 32.29 |
| Transformer ×2 | Waiting to cross | 63.95 | 35.62 | 45.75 |
| Transformer ×2 | Crossing | 31.34 | 57.10 | 40.47 |
| BiLSTM | Moving | 90.73 | 61.05 | 72.99 |
| BiLSTM | Stopped | 20.10 | 86.73 | 32.63 |
| BiLSTM | Waiting to cross | 66.46 | 20.57 | 31.41 |
| BiLSTM | Crossing | 41.42 | 44.71 | 43.00 |
| Transformer ×1 | Moving | 89.16 | 60.93 | 72.39 |
| Transformer ×1 | Stopped | 30.71 | 61.81 | 41.03 |
| Transformer ×1 | Waiting to cross | 67.48 | 15.10 | 24.68 |
| Transformer ×1 | Crossing | 25.33 | 76.42 | 38.04 |

### LOKI head-only · LOKI

| Architecture | Class | Precision (%) | Recall (%) | F1 (%) |
|---|---|---:|---:|---:|
| Transformer ×2 | Moving | 85.09 | 93.80 | 89.23 |
| Transformer ×2 | Stopped | 59.58 | 47.61 | 52.93 |
| Transformer ×2 | Waiting to cross | 69.48 | 70.40 | 69.94 |
| Transformer ×2 | Crossing | 68.09 | 31.96 | 43.50 |
| BiLSTM | Moving | 86.52 | 94.21 | 90.20 |
| BiLSTM | Stopped | 56.07 | 55.68 | 55.87 |
| BiLSTM | Waiting to cross | 69.61 | 61.32 | 65.21 |
| BiLSTM | Crossing | 68.35 | 35.77 | 46.96 |
| Transformer ×1 | Moving | 85.54 | 94.18 | 89.66 |
| Transformer ×1 | Stopped | 57.13 | 47.20 | 51.70 |
| Transformer ×1 | Waiting to cross | 64.72 | 60.07 | 62.31 |
| Transformer ×1 | Crossing | 69.71 | 37.41 | 48.69 |

### LOKI temporal + head · LOKI

| Architecture | Class | Precision (%) | Recall (%) | F1 (%) |
|---|---|---:|---:|---:|
| Transformer ×2 | Moving | 88.74 | 92.86 | 90.76 |
| Transformer ×2 | Stopped | 59.77 | 59.28 | 59.52 |
| Transformer ×2 | Waiting to cross | 66.68 | 67.69 | 67.18 |
| Transformer ×2 | Crossing | 68.14 | 46.41 | 55.22 |
| BiLSTM | Moving | 89.49 | 92.62 | 91.03 |
| BiLSTM | Stopped | 57.73 | 61.06 | 59.35 |
| BiLSTM | Waiting to cross | 66.56 | 61.49 | 63.92 |
| BiLSTM | Crossing | 64.13 | 50.68 | 56.62 |
| Transformer ×1 | Moving | 88.11 | 93.36 | 90.66 |
| Transformer ×1 | Stopped | 56.91 | 57.90 | 57.40 |
| Transformer ×1 | Waiting to cross | 68.55 | 64.81 | 66.63 |
| Transformer ×1 | Crossing | 67.05 | 42.75 | 52.21 |

</details>
