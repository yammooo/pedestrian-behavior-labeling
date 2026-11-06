# E003c — Head versus temporal adaptation

**Protocol agreed and implemented, 2026-11-06. Full comparisons remain user-launched.** This follows [E003b target-head adaptation](../E003b-target-head-adaptation/README.md#results), not the E003a masking method.

## Question and scope

After target-head adaptation, does updating the source-trained temporal block recover additional target behavior performance beyond further optimization of the same head?

Start with one direction, **ROAD-Waymo → LOKI**, ordinary E002d fusion only, seed0. Two continuation runs; do not add E003a bodies, other modalities, reverse transfer or a hyperparameter sweep in this increment. This direction has the larger remaining matched-reference gap: adapted-head validation F1 **64.16%**, versus **74.68%** for the same-family target-trained model. These previously inspected validation sets support exploratory diagnostics, not held-out claims.

## Shared starting point

Both arms start from the identical E002d ROAD source body and the selected E003b LOKI-adapted head. No parameters are randomly reinitialized. Reset the optimizer in both arms; do not reuse optimizer moments or treat this as resuming the original attempt.

- Source body: `outputs/experiments/E002d/runs/fusion-20261030-140937-road-waymo-rgbdrop50/best.pt` (source-selected epoch17).
- Adapted head: `outputs/experiments/E003b/runs/20261104-092202/e002d-road-waymo-to-loki/best.pt` (target-selected epoch87).
- Retain original ROAD normalization for LOKI training and validation, frozen E002 populations/splits, full tracks, natural modality availability and accepted GT. No input masking, extra observations or test access.

| Component | `head` control | `temporal-head` diagnostic |
|---|---|---|
| Geometry/RGB/LiDAR branches and fusion | Same source weights, frozen | Same source weights, frozen |
| Two temporal Transformer blocks | Same source weights, frozen | Same source weights, trainable |
| Final LayerNorm | Source weights, frozen | Same source weights, trainable |
| Linear head | Selected E003b target weights, trainable | Identical target weights, trainable |

The head-only continuation controls for additional target optimization: the original E003b ROAD→LOKI head reached the100-epoch budget. A gain from `temporal-head` would establish benefit from adapting that block under this recipe, not isolate temporal mismatch, annotation differences or capacity as the cause.

## Continuation recipe

| Setting | Both arms unless stated |
|---|---|
| Data | All LOKI training tracks; full LOKI validation for selection |
| Loss / primary metric | Existing mean-track accepted-GT CE / four-class track-weighted macro-F1 |
| Optimizer | Fresh AdamW; weight decay1e-4, betas(0.9,0.999), epsilon1e-8 |
| Learning rates | Head3e-4 in both; temporal blocks and final LayerNorm3e-5 when trainable |
| Budget | Up to100 additional epochs; patience15; effective batch64 whole tracks, final partial batch retained |
| Schedule / clipping | Existing three-epoch linear warmup, then fixed LR; global gradient clipping1.0 |
| Dropout | **Disabled throughout both arms**, including trainable temporal layers, to separate weight adaptation from reintroducing stochastic dropout |
| Seed / precision | Seed0; existing deterministic float32 settings; same logical track order |
| Selection | Validate the shared starting model as epoch0 and retain it as a candidate; then highest validation F1, lower CE and earlier epoch for exact ties; patience resets only on strict F1 improvement |

The lower temporal LR aims to preserve useful source weights while adapting them; it is a fixed starting recipe, not an established optimum. Turning temporal dropout back on would be a separately declared recipe change.

## Execution, evidence and acceptance

Cache frozen **fusion outputs before the temporal Transformer**, `[T,128]`, once for LOKI training/validation with full context and source normalization. Subsequent temporal training reads these compact arrays; no repeated LiDAR spatial readout or raw encoder inference. The head-only control can reuse the existing E003b post-final-LayerNorm cache. Preserve original checkpoints, feature caches and experiment results; fresh outputs belong under ignored `outputs/experiments/E003c/`.

Before continuation, reproduce the original E003b adapted logits in both paths. Verify identical head initialization, exact trainable/frozen parameter sets, disabled dropout after training-mode changes, complete track/padding/GT semantics, and selected-checkpoint reload. Reuse existing training, metric, checkpoint and logging paths. Log update/epoch CE and online training F1, full validation CE/F1, class-wise results, selected epoch and stopping reason; training metrics still use changing weights. Keep checkpoints/predictions local; use one W&B run per continuation.

Report the shared epoch0 score, both selected continuation scores, their difference, curves and the existing target-trained reference with its normalization/training qualification. No new architecture, neighbor analysis, bias-only adaptation, representation plot or test evaluation is included.

## Launch

On Aalto, from the synchronized checkout:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
bash scripts/run-e003c-sequential.sh
```

The command first extracts pre-temporal LOKI training/validation features once, then runs `head` and `temporal-head` sequentially. Fresh outputs live under `outputs/experiments/E003c/runs/temporal-adaptation-<UTC timestamp>/`; the two arms share its `fusion-features/` directory. The head-only arm reads the existing E003b post-final-LayerNorm caches. Each continuation has its own W&B run, local histories, selected/last checkpoints and validation predictions. Per-arm runtime excludes the one-time feature preparation. No full comparison is launched by acceptance checks.

## Acceptance (2026-11-06)

The implementation was transferred through Git and verified on Aalto. **CPU suite:269 tests with13 expected skips; CUDA suite:all269 tests pass** at revision `d3055aa`. This includes a real64-track ×102-timestep forward/backward/AdamW update for the128-wide, two-block temporal model, plus epoch0 checkpoint retention, parameter/dropout scope, LR-ratio warmup, deterministic shared track order and failed-cache rejection.

Bounded extraction and one-epoch continuation checks completed at revision `1fe65f1` under Aalto `outputs/experiments/E003c/acceptance-20261106/`: **3 training tracks /125 context slots /86 GT slots;2 validation tracks**. Both paths reproduce the parent adapted predictions within float32 tolerance, use identical initial head weights, and retain epoch0 when continuation worsens validation CE at unchanged F1. Trainable parameter counts are **516** (`head`) and **397,316** (`temporal-head`). Both runs have complete status with empty failure lists, initial/epoch histories, best/last checkpoints, validation predictions and PNG/SVG plots; the loss plot was inspected. These are integration fixtures, not comparison results.

| GPU smoke | W&B |
|---|---|
| Head only | [08v3gumy](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/08v3gumy) |
| Temporal + final LayerNorm + head | [ezm6422k](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ezm6422k) |

Launcher shell syntax, CLI help, local documentation links and `git diff --check` pass. Full cache preparation and the two full continuations remain user-launched. No held-out test inference occurred.
