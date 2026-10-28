# E002a and E002b — source-validation comparison

Reviewed **27 October 2026**, with an E002a RGB seed-repeat update on **28 October**. The main comparison covers **26 selected checkpoints: eight E002a controls and 18 E002b models**, 13 per dataset, all using **seed 0 and source validation**. The separate RGB update adds seeds1–4 for each source; it does not turn the E002b comparisons into repeated-seed results. Fifteen E002b trainings stopped normally; three LOKI LiDAR trainings were interrupted after 23 epochs. Their saved best checkpoints now have complete evaluation-only exports, without retraining. These exports are not additional training replicates.

Designs and historical execution evidence: [E002a](../E002a-feature-feasibility/README.md), [E002b](README.md). The [E001-family report](../E001b-2d-bbox-contribution/results-review.md) concerns five-seed **held-out tests**; those scores are not directly comparable to this validation table.

## Main findings

- **Geometry + LiDAR with the smaller temporal representation is the strongest development candidate on both sources:** **66.05% ROAD /76.41% LOKI** track-weighted macro-F1. Relative to E002a LiDAR, gains are **3.48 /0.54 percentage points (pp)**, with lower selected-checkpoint CE on both. LOKI's gain is small and its three E002b LiDAR trainings are incomplete; repeated-seed superiority is not established.
- **LayerNorm is not a universal fix.** It improves ROAD RGB substantially, but lowers geometry F1 on both sources and lowers LOKI LiDAR F1. The compact-branch variants offer no consistent improvement.
- **RGB's F1 and loss trade-offs depend on source.** On ROAD, temporal-small nearly matches normalized RGB F1 with much lower CE. On LOKI, its slightly higher F1 comes with worse CE than the other RGB controls. Smaller capacity does not uniformly solve overfitting.
- **LiDAR is more useful than RGB in these selected runs.** E002a's geometry + RGB + LiDAR is below geometry + LiDAR on both sources. There is no E002b three-modality arm, so this does not establish that RGB can never help a better combined model.
- **The new E002a RGB repeats show a classification gap as well as worsening validation CE.** Online train F1 ends at94–99% ROAD /88–91% LOKI, while validation F1 ends at54–58% /64–66%. Historical seed0 models lack train F1, and these changing-weight, dropout-active traces are not fixed-checkpoint train scores; the cause of the gap remains unknown.
- **A later, separate [E002c RGB dropout diagnostic](../E002c-rgb-regularization/README.md) did not find a consistent selected-F1 gain.** Its eight additional seed0 runs retain the E002b runner/output names and are not part of the 26-model comparison below.

## Comparison contract

`g` = the E001b geometry descriptor and its own validity, including bbox geometry; it is not a pure 3D-only input. `g-rgb` adds frozen DINOv3 descriptors from 3× target crops; `g-dditr` adds frozen D-DITR point features through target-centered spatial readout. Only enabled descriptors/validity enter fusion (`enabled-branches-v2`). The native populations, accepted GT and source-only geometry normalization are unchanged.

| Configuration | Final branch LayerNorm | Geometry /RGB /LiDAR widths | Shared temporal width /FFN |
|---|---|---|---|
| E002a | No |64 /128 /128|256 /1024|
| E002b `norm` | Yes |64 /128 /128|256 /1024|
| E002b `temporal-small` | Yes |64 /128 /128|128 /512|
| E002b `branches-small` | Yes |32 /64 /64|256 /1024|

All use two bidirectional temporal blocks, full tracks, the same masks/loss, AdamW, effective batch 64, warmup and maximum 100 epochs /patience 15. `temporal-small` reduces fusion output, temporal blocks, FFN and classifier input together; it does not isolate temporal reasoning. `branches-small` reduces the geometry and enabled scene branches together, including LiDAR spatial readout width. Adding a scene branch also adds availability information and learned parameters, so these are representation-package comparisons, not isolated causal sensor effects.

Primary F1 comes from an equal-track confusion matrix: each track contributes total GT weight one, then F1 averages **all four classes**, including absent-class zeros. It is neither pooled-frame macro-F1 nor mean per-track F1. Checkpoint selection maximizes validation F1, with CE and earlier-epoch tie-breaking; the lowest-CE epoch is not silently substituted. All results remain validation-guided development, following previous E001-family test inspection. Frozen pretrained encoders retain their documented exposure limitations.

## Selected checkpoints

F1 is percent; CE is track-averaged. Epochs are **selected /last completed**. † = interrupted training, not a patience stop. F1 links open the corresponding W&B record (recovered rows use linked evaluation-only records).

| Model | ROAD F1 | ROAD CE | Epochs | LOKI F1 | LOKI CE | Epochs |
|---|---|---|---|---|---|---|
| E002a `g` | [58.61](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/v5e4nxvh) | 0.779 | 14 /29 | [63.78](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/sy9nl5tk) | 0.527 | 16 /31 |
| E002a `g-rgb` | [56.97](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/c3fh137p) | 2.179 | 50 /65 | [66.44](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/hwhjnm5h) | 0.578 | 11 /26 |
| E002a `g-dditr` | [62.57](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/9e5nbdbz) | 0.752 | 12 /27 | [75.87](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/626j7xmj) | 0.420 | 18 /33 |
| E002a `g-rgb-dditr` | [59.08](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/3yaxj76g) | 1.419 | 23 /38 | [72.67](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/39cdefmz) | 0.464 | 9 /24 |
| E002b `g` /`norm` | [56.30](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/67dt83g8) | 0.833 | 9 /24 | [63.34](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/43pm98zn) | 0.546 | 13 /28 |
| E002b `g` /`temporal-small` | [54.00](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/stwwtyyd) | 1.160 | 26 /41 | [64.77](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ucu45075) | 0.520 | 11 /26 |
| E002b `g` /`branches-small` | [54.77](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/1efxhn4q) | 0.838 | 9 /24 | [63.49](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/y21s85sw) | 0.539 | 11 /26 |
| E002b `g-rgb` /`norm` | [62.71](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/y6lc0vg8) | 1.423 | 28 /43 | [67.02](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/svxc47w7) | 0.605 | 11 /26 |
| E002b `g-rgb` /`temporal-small` | [62.29](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/m2ov73kt) | 0.801 | 9 /24 | [67.63](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/0cvhe1n1) | 0.755 | 28 /43 |
| E002b `g-rgb` /`branches-small` | [60.11](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/cva6eski) | 0.997 | 9 /24 | [67.13](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/bllqgs3s) | 0.605 | 11 /26 |
| E002b `g-dditr` /`norm` | [63.31](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/du8ch133) | 0.758 | 11 /26 | [74.27†](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/kzit2eqi) | 0.426 | 11 /23† |
| E002b `g-dditr` /`temporal-small` | [66.05](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ewbrevnb) | 0.719 | 17 /32 | [76.41†](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/fbqzw2dr) | 0.376 | 12 /23† |
| E002b `g-dditr` /`branches-small` | [61.05](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/wdxi83zq) | 0.759 | 9 /24 | [73.11†](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/t6fbkaox) | 0.457 | 17 /23† |

| Source | Arm | Norm − E002a (pp) | Temporal-small − norm (pp) | Branches-small − norm (pp) |
|---|---|---|---|---|
| ROAD | g | -2.31 | -2.30 | -1.53 |
| ROAD | g-rgb | +5.74 | -0.42 | -2.60 |
| ROAD | g-dditr | +0.74 | +2.75 | -2.25 |
| LOKI | g | -0.44 | +1.43 | +0.15 |
| LOKI | g-rgb | +0.58 | +0.61 | +0.11 |
| LOKI | g-dditr | -1.60 | +2.14 | -1.17 |

| Validation source | Groups | Tracks | Context frames | GT frames | GT frames M/S/W/C | Supporting tracks M/S/W/C |
|---|---|---|---|---|---|---|
| ROAD | 78 | 1,110 | 87,824 | 45,388 | 24,018 /11,232 /3,317 /6,821 | 658 /306 /91 /262 |
| LOKI | 93 | 1,861 | 61,900 | 60,231 | 37,410 /4,206 /8,320 /10,295 | 1453 /181 /226 /306 |

Class-supporting track counts overlap when a track changes state. All 13 models per source use exactly the same GT/track population.

![Primary validation F1](../../outputs/experiments/E002b/results-review-20261027-final/primary-macro-f1.png)

[SVG](../../outputs/experiments/E002b/results-review-20261027-final/primary-macro-f1.svg). All three LOKI E002b LiDAR rows are interrupted checkpoints.

## Loss behavior and capacity

The epoch training CE uses changing weights and active dropout. Validation CE evaluates a fixed epoch model in evaluation mode. Their difference is a useful diagnostic, not a matched train/test evaluation or direct calibration measure. The original seed0 models have no train-F1 curve; the [new RGB repeat](../E002a-feature-feasibility/README.md#selected-rgb-checkpoints-across-five-seeds) records online train F1 for seeds1–4.

On ROAD RGB, E002a selects epoch 50 with **train CE 0.035 /validation CE 2.179** despite reaching its minimum validation CE much earlier. The normalized model improves F1 but still selects CE 1.423. Temporal-small reaches almost the same F1 at CE **0.801**. This is a useful F1/loss compromise, not proof that RGB recording shortcuts or a particular layer caused the failure.

The geometry control prevents a blanket capacity conclusion: every ROAD E002b geometry model is below E002a, and temporal-small has worse selected CE as well. On LOKI RGB, temporal-small selects a later checkpoint with higher F1 but worse CE. Keep classification recovery and probabilistic quality separate.

| Source | Scene arm /model | Selected train /val CE | Minimum val CE (epoch) | Last train /val CE | Last F1 (%) |
|---|---|---|---|---|---|
| ROAD | E002a `g-rgb` | 0.035 /2.179 | 0.885 (3) | 0.027 /2.271 | 53.93 |
| ROAD | E002b `g-rgb` /`norm` | 0.070 /1.423 | 0.820 (6) | 0.030 /1.907 | 61.15 |
| ROAD | E002b `g-rgb` /`temporal-small` | 0.385 /0.801 | 0.757 (5) | 0.105 /1.350 | 59.76 |
| ROAD | E002a `g-dditr` | 0.523 /0.752 | 0.752 (12) | 0.258 /1.025 | 59.74 |
| ROAD | E002b `g-dditr` /`norm` | 0.475 /0.758 | 0.728 (8) | 0.177 /1.153 | 59.32 |
| ROAD | E002b `g-dditr` /`temporal-small` | 0.397 /0.719 | 0.705 (9) | 0.185 /1.069 | 61.46 |
| LOKI | E002a `g-rgb` | 0.288 /0.578 | 0.543 (5) | 0.178 /0.842 | 64.28 |
| LOKI | E002b `g-rgb` /`norm` | 0.246 /0.605 | 0.511 (5) | 0.136 /0.852 | 62.84 |
| LOKI | E002b `g-rgb` /`temporal-small` | 0.146 /0.755 | 0.515 (5) | 0.104 /0.928 | 65.36 |
| LOKI | E002a `g-dditr` | 0.231 /0.420 | 0.400 (12) | 0.124 /0.632 | 69.53 |
| LOKI | E002b `g-dditr` /`norm` | 0.275 /0.426 | 0.400 (14) | 0.132 /0.620 | 68.36 |
| LOKI | E002b `g-dditr` /`temporal-small` | 0.277 /0.376 | 0.376 (12) | 0.151 /0.507 | 73.02 |

![Training and validation CE](../../outputs/experiments/E002b/results-review-20261027-final/loss-curves.png)

[CE curves SVG](../../outputs/experiments/E002b/results-review-20261027-final/loss-curves.svg) · [Validation F1 PNG](../../outputs/experiments/E002b/results-review-20261027-final/validation-f1-curves.png) /[SVG](../../outputs/experiments/E002b/results-review-20261027-final/validation-f1-curves.svg). Circles mark selected epochs; LOKI E002b LiDAR histories end at the interruption.

### E002a RGB repeat: seed variation and classification gap

The [five-seed table with per-run W&B links](../E002a-feature-feasibility/README.md#selected-rgb-checkpoints-across-five-seeds) reports E002a `g-rgb` selected source-validation F1 as **58.12% ±0.90 pp ROAD** and **66.74% ±0.50 pp LOKI** (mean ± sample SD). Seeds1–4 alone give58.40% ±0.72 pp and66.81% ±0.54 pp. The repeat's ROAD selected epochs span8–42 and CE0.876–1.881; LOKI spans6–22 and CE0.545–0.717. Seed0 remains in the five-seed F1 summary but lacks online training F1.

The [ROAD CE/F1 plot](../../outputs/experiments/E002a/rgb-seed-curves-20261028/figures/road-waymo-g-rgb-seed-curves.png) ([SVG](../../outputs/experiments/E002a/rgb-seed-curves-20261028/figures/road-waymo-g-rgb-seed-curves.svg)) and [LOKI plot](../../outputs/experiments/E002a/rgb-seed-curves-20261028/figures/loki-g-rgb-seed-curves.png) ([SVG](../../outputs/experiments/E002a/rgb-seed-curves-20261028/figures/loki-g-rgb-seed-curves.svg)) put train/validation CE over train/validation F1 for each repeat seed; dotted lines show selected epochs. Online train F1 continues rising after validation F1 largely plateaus, while validation CE worsens. This supports training-set classification fitting in addition to confidence sharpening, with the online-measure caveat. It does not isolate an RGB shortcut or establish a modality effect: geometry controls were not repeated, and all scores are source validation.

The compact branches barely reduce total parameters when the 256-wide temporal stack stays intact. Their disappointing results do not establish that all capacity reduction fails; they compress a different part of the model.

| Configuration | Geometry | Geometry + RGB | Geometry + LiDAR |
|---|---|---|---|
| E002a | 1,603,460 | 1,734,916 | 1,843,588 |
| E002b /norm | 1,603,588 | 1,735,300 | 1,843,972 |
| E002b /temporal-small | 411,396 | 526,596 | 635,268 |
| E002b /branches-small | 1,591,556 | 1,657,540 | 1,662,788 |

Counts are trainable parameters, excluding frozen encoders. Identical model specifications apply across sources; the recovered LOKI metadata omits these counters, so this table uses the complete ROAD provenance. Relative to `norm`, temporal-small removes **74.3% /69.7% /65.5%** of trainable parameters; branches-small removes only **0.8% /4.5% /9.8%**.

Shared LiDAR epochs execute models sequentially over common prepared inputs. Recorded group time and peak memory describe the shared process, not independently timed model latency. The earlier [bounded execution benchmark](README.md#backend-matched-benchmark-and-calibration) remains the appropriate measured speed comparison; this review does not rank architectures by training speed.

## Class recovery and confusions

### ROAD class F1 (%)

| Model | Moving | Stopped | Waiting | Crossing |
|---|---|---|---|---|
| E002a `g` | 77.53 | 72.48 | 44.72 | 39.71 |
| E002a `g-rgb` | 75.32 | 63.54 | 35.55 | 53.45 |
| E002a `g-dditr` | 77.37 | 72.37 | 44.19 | 56.33 |
| E002a `g-rgb-dditr` | 74.40 | 65.57 | 40.75 | 55.60 |
| E002b `g` /`norm` | 76.76 | 70.67 | 32.88 | 44.90 |
| E002b `g` /`temporal-small` | 74.79 | 65.21 | 33.30 | 42.72 |
| E002b `g` /`branches-small` | 77.17 | 74.00 | 29.06 | 38.86 |
| E002b `g-rgb` /`norm` | 77.47 | 69.99 | 42.35 | 61.03 |
| E002b `g-rgb` /`temporal-small` | 78.92 | 69.70 | 42.95 | 57.58 |
| E002b `g-rgb` /`branches-small` | 75.17 | 67.07 | 41.72 | 56.46 |
| E002b `g-dditr` /`norm` | 78.98 | 73.00 | 40.87 | 60.38 |
| E002b `g-dditr` /`temporal-small` | 80.48 | 73.19 | 44.60 | 65.93 |
| E002b `g-dditr` /`branches-small` | 78.02 | 71.14 | 39.01 | 56.04 |

### LOKI class F1 (%)

| Model | Moving | Stopped | Waiting | Crossing |
|---|---|---|---|---|
| E002a `g` | 91.03 | 53.82 | 63.99 | 46.27 |
| E002a `g-rgb` | 90.55 | 59.13 | 67.87 | 48.19 |
| E002a `g-dditr` | 92.32 | 67.38 | 73.88 | 69.91 |
| E002a `g-rgb-dditr` | 90.01 | 65.00 | 72.62 | 63.06 |
| E002b `g` /`norm` | 90.75 | 55.75 | 59.33 | 47.52 |
| E002b `g` /`temporal-small` | 90.53 | 57.86 | 65.84 | 44.86 |
| E002b `g` /`branches-small` | 89.70 | 58.54 | 62.16 | 43.55 |
| E002b `g-rgb` /`norm` | 89.87 | 61.14 | 69.15 | 47.90 |
| E002b `g-rgb` /`temporal-small` | 90.18 | 57.11 | 70.42 | 52.80 |
| E002b `g-rgb` /`branches-small` | 90.16 | 59.32 | 68.56 | 50.47 |
| E002b `g-dditr` /`norm` | 90.74 | 64.45 | 72.73 | 69.17 |
| E002b `g-dditr` /`temporal-small` | 92.73 | 67.77 | 73.62 | 71.52 |
| E002b `g-dditr` /`branches-small` | 91.45 | 63.16 | 72.49 | 65.32 |

![Class F1](../../outputs/experiments/E002b/results-review-20261027-final/per-class-f1.png)

[SVG](../../outputs/experiments/E002b/results-review-20261027-final/per-class-f1.svg). LOKI E002b LiDAR rows retain the interrupted-training limitation.

ROAD LiDAR temporal-small improves over E002a LiDAR in every class: **Moving +3.11, Stopped +0.82, Waiting +0.41 and Crossing +9.60 pp**. Crossing reaches **65.93% F1**, with 262 supporting tracks; most of the macro-F1 gain is crossing recovery. Geometry's E002b regression is especially pronounced for Waiting, with only 91 supporting ROAD tracks. These changes should not be summarized as uniform improvement across behavior states.

LOKI's smaller-temporal LiDAR changes class F1 by **+0.41 /+0.39 /−0.26 /+1.61 pp** versus original LiDAR. Crossing improves most, while Waiting is nearly unchanged. Stopped recall falls **73.7→69.7%**, despite its slightly higher F1, because precision also changes.

A major persistent confusion is **Crossing→Moving**: ROAD falls **65.5%** with geometry to 43.3% with original LiDAR and 31.0% with smaller-temporal LiDAR; LOKI falls 63.5→29.1→27.7%. For **Stopped→Waiting**, LOKI improves 36.9→14.8% with original LiDAR but rises to 18.7% with smaller-temporal LiDAR. The new model does not repair every class boundary.

| Source | Model | Pooled-frame macro-F1 (%) | Equal-track accuracy (%) | Pooled accuracy (%) |
|---|---|---|---|---|
| ROAD | E002a `g` | 60.17 | 69.83 | 71.47 |
| ROAD | E002a `g-dditr` | 60.43 | 71.44 | 72.03 |
| ROAD | E002b `g-dditr` /`norm` | 64.17 | 72.51 | 73.87 |
| ROAD | E002b `g-dditr` /`temporal-small` | 66.52 | 74.54 | 75.54 |
| ROAD | E002b `g-dditr` /`branches-small` | 61.62 | 70.85 | 72.43 |
| LOKI | E002a `g` | 69.85 | 81.52 | 80.23 |
| LOKI | E002a `g-dditr` | 77.33 | 85.75 | 84.11 |
| LOKI | E002b `g-dditr` /`norm` | 77.63 | 83.84 | 83.21 |
| LOKI | E002b `g-dditr` /`temporal-small` | 77.48 | 86.32 | 84.80 |
| LOKI | E002b `g-dditr` /`branches-small` | 77.75 | 84.22 | 84.26 |

Secondary rankings can differ: LOKI compact-branch LiDAR has **77.75% pooled-frame F1**, above temporal-small 77.48%, while its declared primary F1 is lower 73.11% versus 76.41%. Tracks with more accepted GT frames carry more weight in pooled scoring. Median per-track accuracy is 100% for these ten rows and adds little discrimination; complete mean/median, support-weighted F1, precision and recall remain in the numerical JSON.

Full numerical evidence retains raw-frame and track-weighted confusion matrices, precision, recall, F1 and support. Row-normalized confusions describe the distribution of predictions conditional on GT; their entries are not class precision. Natural class frequencies and semantic differences prevent interpreting cross-source score differences as encoder transfer performance.

## Observation conditions

Strata use each represented track's GT count **inside the slice** to recompute weights. Their macro-F1 averages GT-supported classes only; empty slices are unscored (`null`), and absent-class F1 is `null`. Consequently a two-class slice is not comparable to the primary four-class score. Across models in the same source/bin, GT and condition membership are matched. G/T/F below means supporting groups /tracks /accepted GT frames; M/S/W/C denotes supported classes.

### ROAD selected conditions

| Slice | G /T /F | Classes | a geometry | a +RGB | a +LiDAR | b +LiDAR temporal-small |
|---|---|---|---|---|---|---|
| frame-bbox missing | 0 /0 /0 | — | — | — | — | — |
| frame-bbox valid | 78 /1110 /45,388 | MSWC | 58.61 | 56.97 | 62.57 | 66.05 |
| track-bbox never | 0 /0 /0 | — | — | — | — | — |
| track-bbox partial | 76 /988 /38,158 | MSWC | 57.58 | 58.29 | 63.62 | 66.28 |
| track-bbox complete | 33 /122 /7,230 | MSWC | 62.35 | 46.02 | 55.42 | 63.91 |
| transition near | 38 /187 /850 | MSWC | 39.37 | 34.71 | 35.43 | 36.98 |
| transition steady | 78 /1106 /44,444 | MSWC | 58.88 | 57.25 | 62.92 | 66.60 |
| transition unknown | 26 /75 /94 | MSWC | 56.58 | 51.33 | 52.41 | 46.19 |
| range [0,10) | 9 /16 /101 | MSWC | 64.37 | 51.49 | 74.74 | 57.54 |
| range [10,20) | 43 /206 /1,887 | MSWC | 57.81 | 64.84 | 66.07 | 62.58 |
| range [20,40) | 66 /670 /12,608 | MSWC | 60.06 | 61.71 | 67.19 | 68.53 |
| range [40,inf) | 74 /835 /23,295 | MSWC | 58.13 | 56.36 | 61.54 | 68.08 |
| range unknown | 60 /337 /7,497 | MSWC | 49.72 | 42.75 | 42.37 | 49.07 |

### LOKI selected conditions

| Slice | G /T /F | Classes | a geometry | a +RGB | a +LiDAR | b +LiDAR temporal-small† |
|---|---|---|---|---|---|---|
| frame-bbox missing | 92 /1733 /35,447 | MSWC | 63.38 | 66.14 | 75.47 | 76.22 |
| frame-bbox valid | 90 /1139 /24,784 | MSWC | 64.86 | 69.48 | 77.26 | 76.08 |
| track-bbox never | 73 /717 /17,520 | MSWC | 53.61 | 51.12 | 73.12 | 74.07 |
| track-bbox partial | 89 /1025 /40,846 | MSWC | 64.25 | 68.46 | 75.87 | 75.52 |
| track-bbox complete | 55 /119 /1,865 | MSWC | 56.53 | 52.66 | 70.31 | 68.26 |
| transition near | 42 /286 /1,360 | MSWC | 42.69 | 45.44 | 49.85 | 47.75 |
| transition steady | 93 /1826 /58,836 | MSWC | 64.38 | 66.95 | 76.73 | 76.99 |
| transition unknown | 21 /35 /35 | MSWC | 23.08 | 48.02 | 21.77 | 39.62 |
| range [0,10) | 83 /547 /5,177 | MSWC | 70.01 | 69.08 | 77.53 | 74.91 |
| range [10,20) | 89 /1061 /14,155 | MSWC | 65.14 | 68.15 | 78.82 | 77.00 |
| range [20,40) | 88 /1185 /22,238 | MSWC | 58.99 | 62.47 | 73.64 | 74.10 |
| range [40,inf) | 81 /801 /18,661 | MSWC | 57.79 | 67.20 | 70.97 | 73.36 |
| range unknown | 0 /0 /0 | — | — | — | — | — |

**Interpretation:** LiDAR retains a large LOKI advantage on **717 never-bbox tracks**: E002a geometry 53.61%, RGB 51.12%, LiDAR 73.12%, smaller-temporal LiDAR 74.07%. This supports usefulness without a target RGB crop; it does not establish why the RGB-trained model regresses on that cohort. On ROAD, every accepted GT frame has a valid native bbox/RGB crop; the missing-RGB slice is empty. LOKI has no accepted-GT missing-LiDAR slice, so this review does not establish missing-LiDAR robustness there.

Near-boundary scoring remains weak. ROAD geometry 39.37% exceeds smaller-temporal LiDAR 36.98%; LOKI smaller-temporal LiDAR 47.75% is below original LiDAR 49.85%. Global F1 gains therefore do not establish better transition behavior. ROAD's <10m slice has only 16 tracks /101 GT frames, so its large regression needs much more caution than the 40m+ comparison (835 tracks /23,295 frames).

GT-count and duration results are also mixed: on LOKI, smaller-temporal LiDAR improves the 5–19-GT-frame slice 72.80→75.86%, but lowers 2–4-frame F1 from 63.19→50.70% (127 tracks /376 GT frames). Native duration <1 s falls 57.69→51.58% (191 tracks /577 GT frames), while 1–5 s improves 76.14→79.30%. This is not uniformly better short-track recognition. Only the full table supports each bin-specific statement; these exploratory slices are not independent hypothesis tests.

[GT-count comparison PNG](../../outputs/experiments/E002b/results-review-20261027-final/common-gt-count-strata.png) /[SVG](../../outputs/experiments/E002b/results-review-20261027-final/common-gt-count-strata.svg). The ROAD single-GT bin contains only 4 tracks and two supported classes; do not read it as four-class performance.

Missing bbox annotations, actual RGB crop availability and native 2D-source flags are distinct concepts. A missing annotation is not verified invisibility. Natural available/missing cohorts differ in class mix and difficulty; their score difference is not a causal modality ablation. Keep 3D-only/missing-RGB pedestrians eligible.

Transition `near`, `steady` and `unknown` are **frame-state** slices within labeled runs. They are not transition-event or transition-type F1. Range uses unnormalized saved physical coordinates; duration and gap definitions retain their documented native/grid distinction. Full tables retain all applicable axes and denominators rather than selecting only favorable bins.

## What this supports next

1. **Shortlist smaller-temporal geometry + LiDAR for confirmation**, retaining E002a geometry and the matched normalized LiDAR control. This is the clearest common-source result, not an established final model. The additional cost of repeat seeds should be measured and agreed before launching them.
2. **Keep RGB provisional.** ROAD favors temporal-small as a F1/CE compromise, while LOKI does not give the same loss result. The completed E002a RGB seed repeat confirms a large online train/validation classification gap; the separate [E002c dropout check](../E002c-rgb-regularization/README.md#selected-source-validation-results) gives no consistent F1 gain from stronger dropout. Avoid a broad regularization search on this evidence.
3. **Do not restart the interrupted group merely to make the table look complete.** Report its best-so-far checkpoints and common-epoch comparison. A future confirmation run needs an explicit fresh budget and output directory; it would be a new attempt.
4. **Separate confidence from classification in future diagnostics.** Online train F1 alongside CE is now available for the E002a RGB repeats; consider a fixed-checkpoint train/validation comparison only if needed to resolve the mechanism. ECE, temperature scaling and selective prediction have not been evaluated here.
5. **Keep final generalization evaluation separate.** Declare the confirmation seeds, stopping rule and eventual held-out/transfer access before execution. Current validation-guided choices cannot establish an untouched-test or strict dataset-excluded transfer claim. The current comparison supports a practical reference, not architectural novelty.

No new encoder, scene retrieval module or heterogeneous-supervision design is selected by this report. The evidence supports a narrower confirmation step before committing the remaining project time to one substantive extension.

## Evidence and reproducibility

Canonical artifacts are on `aalto` under `/home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E002b/results-review-20261027-final/`. The review was generated from the isolated evaluation checkout at revision `ddf2faa6fe0267aaaf5ba44fb29a43192c180b49`; the compact numerical/figure bundle is also available locally:

- [Audit and exact numerical results](../../outputs/experiments/E002b/results-review-20261027-final/results.json): source/run/checkpoint/W&B references, full histories, both confusion matrices, class/stratum metrics and 172 unchanged JSON/NPZ input hashes. Selected checkpoint SHA-256 identities are recorded; recovered rows additionally match the original file to its saved lineage digest.
- [All primary/secondary CSV rows](../../outputs/experiments/E002b/results-review-20261027-final/primary-metrics.csv), [per-class CSV](../../outputs/experiments/E002b/results-review-20261027-final/per-class-metrics.csv), [all observation-bin/class CSV rows](../../outputs/experiments/E002b/results-review-20261027-final/strata-metrics.csv). Empty scores stay blank in CSV and `null` in JSON.
- Five PNG/SVG figure pairs: primary F1, class F1, validation-F1 curves, train/validation CE and GT-count strata. Full per-run confusion/strata plots remain in original evaluation folders and linked W&B records; checkpoints and full predictions remain local.
- [Regeneration command](../../README.md#e002ae002b-saved-results-comparison), [collector](../../src/pedestrian_behavior/experiments/e002ab_results.py), [focused acceptance tests](../../tests/test_e002ab_results.py). All metrics reconstruct from saved float32 predictions at `rtol=1e-8`, `atol=1e-10`. Records with different ordering are matched by unique locator; relative/absolute archive references are canonicalized, and GT/time/condition membership matches exactly. Thirteen condition axes are common to all arms; RGB/LiDAR availability axes occur only in enabled arms, giving 14 or 15 axes there.
- The separate RGB repeat uses Aalto `outputs/experiments/E002a/runs/g-rgb-seeds-1-4-20261028-083613-197026/` and original seed0 at `outputs/experiments/E002a/runs/rgb-lidar-20261023-125617/`. Local ignored `outputs/experiments/E002a/rgb-seed-curves-20261028/` contains copied histories/provenance/status and two PNG/SVG pairs; [plot regeneration](../../README.md#e002a-rgb-seed-history-figures) reads histories only. This update did not repeat the 26-checkpoint prediction/confusion audit or evaluate held-out tests.

Original histories, checkpoints, statuses, prediction archives and W&B records stay unchanged. The three recovered LOKI records link back to interrupted training and retain 23 completed epochs. All other models stopped by patience before the 100-epoch cap. E002a LOKI LiDAR already reaches its best F1 at epoch 18, so restricting all four LOKI LiDAR histories to their first 23 epochs preserves the best-so-far ranking (76.41 /75.87 /74.27 /73.11% for temporal-small /E002a /norm /branches-small). This controls the observed epoch window, not the counterfactual effect of completing training; later improvement remains unknown.

No new training, inference, encoder extraction, feature-cache mutation or held-out scoring was performed for this review. No seed standard deviation, confidence interval or significance test is available: one training seed on one frozen split is not repeated evidence. Earlier five-seed E001-family test results remain in their own report and are not pooled with these validation results.
