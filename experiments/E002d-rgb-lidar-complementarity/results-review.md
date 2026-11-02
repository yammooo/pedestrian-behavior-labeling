# E002d — compact results review

All scores below are **seed0 source-validation** results. F1 is four-class track-weighted macro-F1 (%); CE is track-averaged CE at the F1-selected checkpoint. E002d trained and evaluated all four attempts. No held-out or cross-dataset results are included.

## Modality and model comparison

Each metric cell is `F1 / CE`. `g` is geometry, `RGB` is frozen DINOv3, and `LiDAR` is frozen D-DITR. E002a is the original model (no branch LayerNorm, temporal width256). E002b/c/d use branch LayerNorm and temporal-small (width128); E002c changes RGB branch dropout, while E002d adds RGB to the geometry + LiDAR model. Thus E002a is historical context; E002b is the closest LiDAR-only control. LOKI E002b LiDAR checkpoints were selected from runs interrupted at epoch23 (†).

| Source | Input | E002a | E002b temporal-small | E002c RGB branch dropout0.5 | E002d RGB dropout0 | E002d RGB dropout0.5 |
|---|---|---:|---:|---:|---:|---:|
| ROAD-Waymo | Geometry | 58.61 /0.779 | 54.00 /1.160 | — | — | — |
| ROAD-Waymo | Geometry + RGB | 56.97 /2.179 | 62.29 /0.801 | 64.37 /0.805 | — | — |
| ROAD-Waymo | Geometry + LiDAR | 62.57 /0.752 | 66.05 /0.719 | — | — | — |
| ROAD-Waymo | Geometry + RGB + LiDAR | 59.08 /1.419 | — | — | [64.75 /0.780](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/3zvfqock) | [66.07 /0.803](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/9k11kn3b) |
| LOKI | Geometry | 63.78 /0.527 | 64.77 /0.520 | — | — | — |
| LOKI | Geometry + RGB | 66.44 /0.578 | 67.63 /0.755 | 68.20 /0.520 | — | — |
| LOKI | Geometry + LiDAR | 75.87 /0.420 | 76.41† /0.376† | — | — | — |
| LOKI | Geometry + RGB + LiDAR | 72.67 /0.464 | — | — | [72.29 /0.585](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/dt3kntl0) | [74.68 /0.385](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/lj325b3g) |

Earlier values and their qualifications are in the [E002a/E002b review](../E002b-normalization-capacity/results-review.md) and [E002c review](../E002c-rgb-regularization/README.md). The [selected-F1 graph](../../outputs/experiments/E002d/results-review-20261102/figures/selected-f1.png) makes the model-package comparisons easier to scan ([SVG](../../outputs/experiments/E002d/results-review-20261102/figures/selected-f1.svg)).

**What changes with the extra RGB branch:** dropout0.5 raises E002d F1 over dropout0 by1.32pp on ROAD and2.39pp on LOKI. Against the closest geometry + LiDAR control, that is essentially tied on ROAD (+0.02pp) and lower on LOKI (−1.73pp). Its selected CE is higher by0.084 ROAD and0.009 LOKI. So this pilot does not show a consistent improvement from adding RGB to the best LiDAR model. RGB + LiDAR also scores above the E002c RGB-only dropout0.5 result (by1.70pp ROAD and6.48pp LOKI), but this single-seed, cross-experiment comparison does not isolate a causal LiDAR contribution.

## Training fit and validation deterioration

E002d reports online training CE/F1 accumulated while weights change and dropout is active. The last-epoch train values are not a fixed-checkpoint training evaluation. Validation uses each epoch's fixed model in evaluation mode.

| Source | RGB branch dropout | Selected epoch | Selected val F1 / CE | Last epoch | Last val F1 / CE | Last online train F1 / CE |
|---|---:|---:|---:|---:|---:|---:|
| ROAD-Waymo | 0 | 9 | 64.75 /0.780 | 24 | 59.37 /1.362 | 94.29 /0.122 |
| ROAD-Waymo | 0.5 | 17 | 66.07 /0.803 | 32 | 64.25 /1.054 | 89.89 /0.197 |
| LOKI | 0 | 21 | 72.29 /0.585 | 36 | 71.47 /0.815 | 96.99 /0.057 |
| LOKI | 0.5 | 12 | 74.68 /0.385 | 27 | 71.03 /0.548 | 93.70 /0.109 |

All four runs continue to fit their training data after validation quality peaks: validation CE rises and train/validation F1 separates. RGB branch dropout0.5 reduces the final training F1 and limits the CE rise compared with dropout0, especially on ROAD, but it does not remove the gap. F1 and CE also choose different trade-offs: on ROAD the0.5 arm has higher selected F1 but slightly higher selected CE; on LOKI it improves both selected metrics over dropout0.

![E002d training and validation curves](../../outputs/experiments/E002d/results-review-20261102/figures/training-validation-curves.png)

[SVG](../../outputs/experiments/E002d/results-review-20261102/figures/training-validation-curves.svg). Green is the E002b LiDAR control; its LOKI training stopped at epoch23. E002b did not log training F1.

## Run records

All four attempts completed with patience stopping before the100-epoch budget. Configs, histories, checkpoints, predictions and full metrics remain on Aalto at `/home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation/outputs/experiments/E002d/runs/fusion-20261030-140937-{road-waymo,loki}-rgbdrop{0,50}/`. Exact run records: [ROAD RGB dropout0](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/3zvfqock), [ROAD0.5](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/9k11kn3b), [LOKI0](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/dt3kntl0), [LOKI0.5](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/lj325b3g). The metric denominators match the frozen E002 populations: ROAD78 groups /1,110 tracks /45,388 GT frames; LOKI93 /1,861 /60,231. This is validation-guided, single-seed development evidence, not final generalization evidence.
