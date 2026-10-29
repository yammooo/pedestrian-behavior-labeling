# E002c — RGB dropout diagnostics

The temporal-dropout comparison completed 2026-10-28; the RGB branch-dropout follow-up completed afterward. The temporal-dropout runs were launched with the E002b runner before this separate logical record was agreed; their existing E002b output paths and W&B names remain unchanged.

## Comparison

Use `g-rgb` with E002b `norm` (temporal width256, FFN1024) and `temporal-small` (width128, FFN512). Within each variant and source, compare the original **0.1** dropout control with **0.3** and **0.5**. The new probability applies together to attention probabilities, projected attention output, FFN hidden activations and FFN output in both temporal blocks. Geometry/RGB projections have no dropout; this arm has no LiDAR spatial block. This comparison does not isolate any one dropout site.

All 12 cells use seed0, the same frozen geometry and 3× DINO CLS features, source-only normalization, full track populations, four-class GT, effective 64-track batches, AdamW/warmup, 100-epoch ceiling, patience15 and highest source-validation track-weighted macro-F1 checkpoint selection. ROAD validation has78 groups /1,110 tracks /45,388 GT frames; LOKI has93 /1,861 /60,231. The matched controls and new runs have identical source settings, feature identities, model specifications except dropout, and validation denominators. Eight new attempts completed with patience stopping; all eight report zero changed protected inputs. No held-out or cross-dataset test was opened.

## Selected source-validation results

F1 is four-class track-weighted macro-F1. CE is mean per-track CE at the **F1-selected checkpoint**, not minimum CE over epochs. ΔF1 compares each row with dropout0.1 for the same source and capacity. Epochs show selected /last completed. F1 links open the original W&B runs.

| Source | Temporal model | Dropout | F1 (%) | ΔF1 (pp) | CE | Epochs |
|---|---|---:|---:|---:|---:|---:|
| ROAD | `norm` | 0.1 | [62.71](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/y6lc0vg8) | — | 1.423 | 28/43 |
| ROAD | `norm` | 0.3 | [62.40](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/rim81kp9) | −0.31 | 1.840 | 36/51 |
| ROAD | `norm` | 0.5 | [61.98](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/2ulhiwq7) | −0.73 | 0.930 | 9/24 |
| ROAD | `temporal-small` | 0.1 | [62.29](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/m2ov73kt) | — | 0.801 | 9/24 |
| ROAD | `temporal-small` | 0.3 | [62.18](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/u2tyg0q5) | −0.11 | 1.403 | 24/39 |
| ROAD | `temporal-small` | 0.5 | [63.02](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/rfvz2naj) | +0.74 | 1.140 | 17/32 |
| LOKI | `norm` | 0.1 | [67.02](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/svxc47w7) | — | 0.605 | 11/26 |
| LOKI | `norm` | 0.3 | [66.90](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/4d8fv2h8) | −0.12 | 0.654 | 14/29 |
| LOKI | `norm` | 0.5 | [66.42](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/wrdorvaz) | −0.60 | 0.660 | 13/28 |
| LOKI | `temporal-small` | 0.1 | [67.63](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/0cvhe1n1) | — | 0.755 | 28/43 |
| LOKI | `temporal-small` | 0.3 | [67.38](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/74n84rxr) | −0.25 | 0.708 | 19/34 |
| LOKI | `temporal-small` | 0.5 | [66.24](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/wxe3lrl9) | −1.39 | 0.554 | 7/22 |

**Stronger dropout does not consistently improve selected F1.** ROAD `temporal-small` at0.5 gains0.74 pp over its0.1 control but has higher selected CE (1.140 versus0.801). All four LOKI higher-dropout rows lose F1; LOKI `temporal-small` at0.5 has lower selected CE but loses1.39 pp F1, with Crossing class F1 falling52.80→45.82%. The F1 and CE preferences therefore differ by source and model. These are small single-seed differences, not evidence of a reliable new winner.

## Overfitting diagnostic

The new runs log online training F1 during changing-weight, dropout-active training. Their final epoch values show what increased regularization changed; they are not fixed-checkpoint train scores.

| Source /model | Dropout | Final train /val F1 (%) | Final train /val CE |
|---|---:|---:|---:|
| ROAD `norm` | 0.3 | 98.22 /59.82 | 0.041 /2.010 |
| ROAD `norm` | 0.5 | 94.43 /60.17 | 0.114 /1.633 |
| ROAD `temporal-small` | 0.3 | 96.95 /56.43 | 0.061 /1.817 |
| ROAD `temporal-small` | 0.5 | 94.78 /57.19 | 0.100 /1.783 |
| LOKI `norm` | 0.3 | 90.61 /64.12 | 0.157 /0.888 |
| LOKI `norm` | 0.5 | 88.61 /65.53 | 0.190 /0.853 |
| LOKI `temporal-small` | 0.3 | 91.29 /66.81 | 0.153 /0.791 |
| LOKI `temporal-small` | 0.5 | 86.28 /65.41 | 0.224 /0.754 |

Within every source/model pair, increasing dropout0.3→0.5 lowers final online train F1 and final validation CE, while final validation F1 moves only slightly or falls. Validation CE still rises from its earlier minimum in all eight new runs. The classification gap remains. This pattern is consistent with reduced training fit and less late CE deterioration, but it does not identify the mechanism or which dropout site matters.

## Evidence and decision

Canonical new runs: `aalto:/home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E002b/runs/dropout-20261028-114606-{road-waymo,loki}/`; original0.1 controls: `E002b/runs/independent-20261025-125024-264808/` (ROAD) and `E002b/runs/independent-20261025-134338-803720/` (LOKI). New runs used code revision `ca46415` and the completed RTX4080 [calibration](../E002b-normalization-capacity/README.md#rgb-dropout-probe--prepared-2026-10-28); local checkpoints, full float32 predictions, metrics, histories and plots remain in their original folders. [Local summary CSV](../../outputs/experiments/E002c/results-20261028/summary.csv) and copied JSON inputs retain the numerical evidence for this record. Each selected F1/CE above matches its saved validation metrics; class/support denominators agree within source across all 12 cells. W&B links for all 12 cells are in the F1 table; no artifacts were uploaded for this review.

Keep the 0.1 controls as the current RGB references. Do not promote 0.5 from one ROAD F1 increase or start a wider dropout search on this evidence. A repeated-seed comparison would be needed to establish a robust gain; the current result is an exploratory source-validation diagnostic and does not alter E002b's original 18-model comparison or the E002a seed-repeat findings.

## RGB branch dropout — completed 2026-10-29

This narrower follow-up tests whether regularizing the RGB descriptor itself changes the geometry + RGB validation gap. Use the E002b `g-rgb` `temporal-small` model with temporal dropout fixed at0.1. Apply dropout after the RGB projection and LayerNorm, before the existing validity mask and fusion. The dropout mask is shared across every timestep of a track, independently sampled for each track during training. Geometry, validity flags and evaluation inputs are unchanged. Compare RGB branch dropout0.3 and0.5 with the existing zero-branch-dropout E002b controls above, separately on ROAD-Waymo and LOKI. This isolates the regularization site more closely than the completed temporal-dropout comparison.

Four new seed0 source-validation attempts use the same frozen populations, features, optimizer,100-epoch/patience15 budget and F1-selected checkpoint policy as the controls. They are stored under `outputs/experiments/E002c/runs/` with `e002c-<source>-g-rgb-temporal-small-rgbdrop{30|50}-s0` W&B names. This is exploratory validation-guided development; do not use held-out tests to choose the dropout value. If a gain looks meaningful, repeat seeds before claiming it. [The manual command](../../README.md#e002c-rgb-branch-dropout) is the execution reference.

### Selected source-validation results

F1 is the four-class track-weighted macro-F1; CE is mean per-track CE at the **F1-selected epoch**. ΔF1 is relative to the matched zero-branch-dropout control, not to the temporal-dropout rows above. Epochs are selected /last completed. Links open the original W&B runs.

| Source | RGB branch dropout | F1 (%) | ΔF1 (pp) | CE | Epochs |
|---|---:|---:|---:|---:|---:|
| ROAD | 0 (E002b control) | [62.29](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/m2ov73kt) | — | 0.801 | 9/24 |
| ROAD | 0.3 | [64.21](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/tzr354j7) | +1.92 | 1.377 | 40/55 |
| ROAD | 0.5 | [64.37](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/iqsomnq7) | +2.08 | 0.805 | 16/31 |
| LOKI | 0 (E002b control) | [67.63](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/0cvhe1n1) | — | 0.755 | 28/43 |
| LOKI | 0.3 | [67.83](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/m5nqlrf0) | +0.20 | 0.548 | 12/27 |
| LOKI | 0.5 | [68.20](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ck3vsjb4) | +0.57 | 0.520 | 11/26 |

All four new runs completed with patience stopping, no recorded failures or changed protected inputs. Their validation denominators match the controls: ROAD78 groups /1,110 tracks /45,388 accepted GT frames; LOKI93 /1,861 /60,231. No held-out or cross-dataset test was opened.

The F1 gain is larger on ROAD. Relative to its control, ROAD0.5 gains3.62 pp on Waiting to Cross and4.01 pp on Crossing, while Moving falls0.92 pp; ROAD0.3 gains7.07/5.00 pp on those two classes but loses2.17/2.20 pp on Moving/Stopped and has much worse selected CE. On LOKI, both settings chiefly improve Stopped (+3.62/+2.87 pp), while Waiting to Cross and Crossing fall slightly. The average gain therefore does not mean every class improved.

### Training trace and interpretation

These are **last-epoch** traces, separate from the selected-checkpoint table. Training F1 and CE are accumulated online while weights change and dropout is active; validation uses the fixed end-of-epoch model in evaluation mode.

| Source | RGB dropout | Final train /val F1 (%) | Final train /val CE |
|---|---:|---:|---:|
| ROAD | 0.3 | 96.23 /61.31 | 0.074 /1.643 |
| ROAD | 0.5 | 89.02 /62.08 | 0.218 /1.085 |
| LOKI | 0.3 | 89.69 /64.35 | 0.176 /0.740 |
| LOKI | 0.5 | 88.09 /67.40 | 0.200 /0.612 |

Every run's final validation F1 is below its selected F1, and final validation CE is above its selected CE. The train/validation classification gap remains substantial, especially on ROAD, despite the lower final training F1 at0.5. RGB branch dropout appears more promising than increasing temporal dropout in this seed0 comparison, but these are validation-guided, single-seed results. The0.16 pp ROAD and0.37 pp LOKI differences between0.5 and0.3 do not establish a preferred probability. A matched repeated-seed control versus branch-dropout comparison is the minimal confirmation before changing the reference model; preserve the current controls meanwhile.

### Evidence

The four new attempts are at `aalto:/home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation/outputs/experiments/E002c/runs/rgb-branch-20261028-150200-{road-waymo,loki}-rgbdrop{30,50}/`. They used code revision `e201ebf`, the completed RTX4080 calibration `E002b/calibration-rgb-branch-20261028/profile.json`, and the shared E002a RGB preparation. Each local attempt retains status, history, checkpoint, validation predictions/metrics and plots; W&B links are in the result table. The launch recorded no source diff. These runs add no new E002 held-out evidence.
