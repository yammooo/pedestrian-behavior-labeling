# E003b — Target-head adaptation

**Four seed0 adaptations complete; results reviewed 2026-11-04.** This diagnostic follows the [E003a masking results](../E003a-structured-observation-masking/README.md#results).

## Question and comparison

Does a source-trained representation retain useful target behavior information that a new linear classifier can recover? Does E003a masking improve that adaptability relative to its matched E002d fusion control, despite its mixed direct-transfer result?

Freeze the complete source-trained network, including its final LayerNorm. Replace only the final `Linear(128,4)` with a fresh head: **516 trainable parameters**. Keep the frozen network in evaluation mode with dropout disabled throughout extraction. Natural target availability is retained; no E003a observation masks are sampled. RGB/LiDAR encoders, spatial readout, geometry branch, fusion and temporal layers all remain frozen.

| Frozen representation | Original source → head-training target | Original selected epoch | Original checkpoint under `outputs/experiments/` |
|---|---|---:|---|
| E002d fusion, RGB descriptor dropout0.5 | ROAD → LOKI | 17 | `E002d/runs/fusion-20261030-140937-road-waymo-rgbdrop50/best.pt` |
| E003a masked fusion | ROAD → LOKI | 24 | `E003a/runs/20261103-105614/road-waymo/best.pt` |
| E002d fusion, RGB descriptor dropout0.5 | LOKI → ROAD | 12 | `E002d/runs/fusion-20261030-140937-loki-rgbdrop50/best.pt` |
| E003a masked fusion | LOKI → ROAD | 12 | `E003a/runs/20261103-105614/loki/best.pt` |

All four original checkpoints were selected using their original source validation. Fix them before target-head fitting; do not reselect the source network using target scores. E002d checkpoint hashes remain in [the transfer manifest](../E002-multimodal-reference/transfer-comparison.json); E003b records all four hashes, source selections and normalization values in its evidence.

## Data and fitting

| Setting | Fixed value |
|---|---|
| Population, groups, GT and grid | Existing frozen E002 train/validation splits and four-state targets; no new split or population |
| Head training | All opposite-dataset **training** tracks, one presentation per track per epoch |
| Head selection | Full opposite-dataset **validation** track-weighted four-class macro-F1; lower CE then earlier epoch breaks exact ties |
| Normalization | Original **source-training** baseline and bbox statistics for target training **and** validation; no target-statistics fitting |
| Head | Fresh affine `128→4`; no dropout, extra hidden layer or trainable normalization |
| Loss | Existing mean accepted-GT CE per track, averaged over actual batch tracks; padding/unlabeled slots excluded |
| Optimizer | AdamW, LR3e-4, weight decay1e-4, betas(0.9,0.999), epsilon1e-8 |
| Budget | Batch64 whole tracks, final partial batch retained, three-epoch warmup, clip1.0, maximum100 epochs, patience15 |
| Seed and precision | Seed0; deterministic float32, existing backend settings |
| Test access | **None in this first diagnostic**; final test evaluation requires a fixed adaptation recipe and separate feature preparation |

Only the adaptation head uses target labels. All original source weights/statistics and existing feature caches are read-only. This is **full-target-label linear adaptation**, not few-shot or zero-shot transfer. Both validation sets have already informed development. Recovery would show that a linear boundary can recover useful information; it would not isolate annotation differences from class priors or feature-distribution differences. Poor recovery would not prove the representation contains no useful information.

## Execution and evidence

For each frozen checkpoint, run the target training and validation tracks through the frozen body once. Save unpadded float32 post-final-LayerNorm features `[T,128]`, targets, accepted-GT masks and track identities/references in pickle-free local files. Reuse existing prepared DINO CLS/indexed D-DITR inputs and the checkpoint's accepted execution limits; no raw encoder rerun or replacement point cache. Preserve full-track temporal context before extracting each frame representation.

Fit the new head from these compact features. Evaluate the original source classifier and the adapted classifier on exactly the same target-validation tracks. Report track-weighted macro-F1, track-averaged CE, class-wise scores and denominators. Save predictions so scores can be reconstructed. Head checkpoint selection never uses test data. No target test features are prepared by the launcher.

Fresh attempts live under ignored `outputs/experiments/E003b/`. Retain configuration, source-checkpoint identity/hash, frozen normalization, feature-extraction scope, histories, best/last head state dictionaries, selected epoch/stopping reason, predictions, numerical comparisons and PNG/SVG curves. A head checkpoint references the original frozen source model; it is not a standalone full labeler. Existing attempts are preserved; no automatic resume/retry.

One W&B run per adaptation uses readable direction/reference names. Keep ordinary step/epoch CE/F1, final validation metrics and plots. Checkpoints, full representations and predictions remain local. Logging failures propagate with available local evidence retained.

The [launch command](../../README.md#e003b-target-head-adaptation) runs all four adaptations sequentially. Feature extraction remains the expensive stage; no full-run runtime claim is made before measurement.

## Acceptance

Verify frozen parameters and evaluation mode, original-logit reproduction from cached representations, source normalization on both target splits, unchanged GT/lengths, equal-track loss, padding/unlabeled exclusion, the516-parameter head and head-only gradients, feature-cache round trips, selection and head-checkpoint reload. Run the CPU suite and bounded CUDA checks through the actual four-checkpoint extraction/training/logging path. Full target-training comparisons remain user-launched; acceptance fixtures are not experimental results.

Implementation revision **`da9b6de`** was transferred through Git and verified on Aalto. Local CPU suite: **259 tests pass,12 expected skips**, including six focused E003b tests. CLI help, launcher shell syntax and `git diff --check` pass. The original sandbox suite could not bind the existing viewer test’s localhost socket; the complete rerun with localhost access passed without changing that test.

All four one-epoch CUDA/W&B smoke attempts completed with empty failure lists under Aalto `outputs/experiments/E003b/acceptance-20261104/smoke-02/`. They exercised existing prepared target features, original-head logit reproduction, compact feature caches, fresh-head fitting, selected/last head checkpoints, predictions and plots. These are bounded integration fixtures, not E003b comparison results. The initial noninteractive SSH launch found no `uv` on PATH and exited before Python; its shell log is preserved as `acceptance-20261104/smoke.log`. Adding Aalto’s existing uv directory resolved it; the user’s interactive launcher is unchanged.

Read-only artifact verification passes for all four cases: each uses3 target-training /2 target-validation tracks and trains only516 parameters. Original/adapted metrics reconstruct from saved predictions; reloading the selected head reproduces cached-feature logits; both feature splits retain source normalization; protected-input changes are empty. Each attempt has best/last checkpoints and12 PNG/SVG plot pairs. Aalto `outputs/experiments/E003b/acceptance-20261104/verification.json` records these checks.

| Bounded smoke | W&B record |
|---|---|
| ROAD→LOKI, E002d representation | [arx82owt](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/arx82owt) |
| ROAD→LOKI, E003a representation | [eh6egi9q](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/eh6egi9q) |
| LOKI→ROAD, E002d representation | [k8uc4eg2](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/k8uc4eg2) |
| LOKI→ROAD, E003a representation | [x7n28ue8](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/x7n28ue8) |

The full adaptations below used [the sequential command](../../README.md#e003b-target-head-adaptation). No target test preparation or inference occurred.

## Results

**A fresh linear head recovers substantial target-validation performance from both frozen representations. E003a masking gives no consistent adaptation advantage over E002d.** Target training labels and target-validation checkpoint selection are used here: these are full-label adaptation results, not improved zero-shot transfer.

All four attempts under Aalto `outputs/experiments/E003b/runs/20261104-092202/` completed with empty failure lists at revision `9e09e7f`. The complete body stayed frozen, including final LayerNorm; only the516-parameter head learned. Both target splits retained original source normalization. Target training contains8,574 LOKI or4,380 ROAD tracks. Validation support is **LOKI:93 groups /1,861 tracks /60,231 GT frames; ROAD:78 /1,110 /45,388**. Tests remain closed.

### Main comparison: adaptation versus a model trained on the target

The relevant reference for the remaining transfer gap is the **same experiment family trained directly on the target**, evaluated on the same target-validation population. Similar source-validation and adapted-target F1 values do not establish little degradation: they measure different datasets.

| Frozen representation → target | Adapted head F1 (%) | Same-family target-trained model F1 (%) | Remaining gap (pp) |
|---|---:|---:|---:|
| E002d ROAD→LOKI | 64.16 | 74.68 | 10.52 |
| E003a ROAD→LOKI | 62.77 | 76.08 | 13.32 |
| E002d LOKI→ROAD | 59.49 | 66.07 | 6.58 |
| E003a LOKI→ROAD | 60.49 | 65.46 | 4.97 |

Against these references, **LOKI→ROAD has the smaller remaining gap**, despite lower absolute F1. This comparison holds the target population and experiment family fixed, but changes task-network training and normalization: target-trained references learn branches/fusion/temporal layers/head on target data and use target-training statistics; adaptation freezes the source-trained body and retains source statistics. The pretrained DINO/D-DITR encoders stay frozen in both. These references are not an upper bound or a perfectly isolated head comparison.

![Direct transfer and adapted heads versus target-trained references on the same target validation](../../outputs/experiments/E003b/results-review-20261104/figures/head-adaptation-comparison.png) ([SVG](../../outputs/experiments/E003b/results-review-20261104/figures/head-adaptation-comparison.svg))

### Original source performance and adaptation evidence

Cells show track-weighted four-class macro-F1 (%) / mean track CE. Source validation is the original model's performance on its own dataset, at the original source-selected checkpoint. The direct/adapted target columns share a frozen body and target-validation population; only the adapted head has target-label access. Source and target columns have different populations and class distributions, so their differences are descriptive, not matched-population adaptation effects. Deltas use unrounded target scores.

| Frozen source → target | Representation | Original source-val F1 / CE | Direct target-val F1 / CE | Adapted target-val F1 / CE | Target ΔF1 (pp) | Selected / final epoch; stop |
|---|---|---:|---:|---:|---:|---|
| ROAD → LOKI | E002d fusion | 66.07 /0.8027 | 46.50 /1.2573 | **64.16 /0.5162** | +17.66 | 87 /100; budget |
| ROAD → LOKI | E003a masking | 65.46 /0.8002 | 40.26 /1.1923 | 62.77 /0.5260 | +22.51 | 68 /83; patience |
| LOKI → ROAD | E002d fusion | 74.68 /0.3853 | 45.02 /1.3485 | 59.49 /0.7134 | +14.46 | 61 /76; patience |
| LOKI → ROAD | E003a masking | 76.08 /0.3859 | 45.28 /1.2708 | **60.49 /0.7042** | +15.22 | 95 /100; budget |

After adaptation, E003a differs from E002d by **−1.39pp ROAD→LOKI /+1.01pp LOKI→ROAD**. Its larger ROAD→LOKI recovery starts from a worse original head and does not yield a better adapted score.

Recovery establishes that useful target information survives in the source-trained full-track representation. It does **not** separate annotation rules, class priors, feature shift or source-specific decision boundaries. The remaining gap to target-trained models also does not establish that the frozen pretrained RGB/LiDAR encoders are the bottleneck: the source-trained branches and temporal body are frozen too.

### Head training and class-wise recovery

![Head training and target-validation CE and F1](../../outputs/experiments/E003b/results-review-20261104/figures/head-training-validation-curves.png) ([SVG](../../outputs/experiments/E003b/results-review-20261104/figures/head-training-validation-curves.svg))

At selection, online training F1 /validation F1 is63.18 /64.16% and61.84 /62.77% for the two LOKI-target heads;60.83 /59.49% and62.50 /60.49% for the ROAD-target heads. Selected training/validation CE is0.502 /0.516,0.511 /0.526,0.677 /0.713 and0.662 /0.704, respectively. These fresh heads have small gaps compared with earlier full-body training. Training metrics still accumulate with changing head weights; **all dropout is disabled**, and there is no extra end-of-epoch training evaluation. A small head-fitting gap cannot rule out limitations already embedded in the frozen body.

Two attempts hit the100-epoch budget, so this does not establish optimal linear recoverability or convergence. The other two stop after15 epochs without a strict F1 improvement; CE can continue falling while F1 fluctuates.

Per-class track-weighted F1 (%), original→adapted:

| Direction / representation | Moving | Stopped | Waiting to cross | Crossing |
|---|---:|---:|---:|---:|
| ROAD→LOKI /E002d | 67.47→89.28 | 32.29→54.05 | 45.75→68.41 | 40.47→44.88 |
| ROAD→LOKI /E003a | 70.81→88.90 | 43.54→53.59 | 10.49→64.94 | 36.20→43.64 |
| LOKI→ROAD /E002d | 76.97→79.04 | 62.62→73.43 | 30.14→31.79 | 10.36→53.69 |
| LOKI→ROAD /E003a | 76.42→79.98 | 56.66→73.13 | 30.71→34.19 | 17.33→54.68 |

Crossing accounts for much of the ROAD-target recovery; waiting-to-cross remains weak there. On LOKI, moving and waiting-to-cross improve strongly, while crossing improves less. Frame class support is LOKI `[37410,4206,8320,10295]` /ROAD `[24018,11232,3317,6821]` in the table's class order; primary F1 remains equal-track weighted.

### Evidence and reproduction

| Attempt directory under `runs/20261104-092202/` | W&B |
|---|---|
| `e002d-road-waymo-to-loki` | [lp7rjhja](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/lp7rjhja) |
| `e003a-road-waymo-to-loki` | [uc0enhvz](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/uc0enhvz) |
| `e002d-loki-to-road-waymo` | [h6k42uld](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/h6k42uld) |
| `e003a-loki-to-road-waymo` | [iheyyj7p](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/iheyyj7p) |

Each attempt retains `validation-comparison.json`, histories, provenance, selected/last head checkpoints, original/adapted predictions and both compact feature archives. Small JSON evidence was copied to ignored `outputs/experiments/E003b/results-review-20261104/evidence/`; large representations and predictions remain on Aalto. Rebuild the two PNG/SVG pairs and numerical summary with `UV_CACHE_DIR=/tmp/uv-e003b-results-review MPLCONFIGDIR=/tmp/mpl-e003b-results-review uv run --locked --extra cpu python scripts/summarize-e003b-results.py`. No new training or inference was needed for this review. These are exploratory seed0 results on previously inspected validation sets.
