# E003a — Structured observation masking

**Specification agreed and implemented 2026-11-03; CPU/CUDA and bounded GPU/W&B acceptance complete.** Two new seed0 training attempts, one per source, will compare against the completed E002d controls. Full training remains user-launched. No E003a performance results exist yet.

## First comparison

Does training with varied branch availability and temporal persistence improve **natural-availability cross-dataset validation F1**, while retaining useful source performance? Source changes are reported alongside transfer, without a numerical pass/fail cutoff. Missing-input diagnostics are secondary. These validation populations already informed development; this is **target-informed exploratory development**, not held-out or strict zero-shot evidence. Test sets remain closed.

Compare the E002d `g-rgb-dditr / temporal-small / rgbdrop50` reference with the same model trained using observation masking. Present each original training track **once per epoch** in both conditions. No doubled population, extra masked samples, reconstruction objective, teacher or auxiliary labels. Whole-modality removal has established [ModDrop/ActionMAE precedents](../../docs/literature/README.md#observation-masking-precedents); this experiment does not claim architectural novelty.

## Fixed reference

Retain frozen E002 populations/splits, accepted four-state GT, source-only normalization, E001b geometry, 3× DINOv3 CLS crops and indexed D-DITR features. Preserve missing-RGB/3D-only pedestrians. Initialize trainable layers freshly at seed0; encoders remain frozen. The source controls are:

| Training source | E002d attempt under `outputs/experiments/E002d/runs/` | Selected epoch | Source F1 | Source CE |
|---|---|---:|---:|---:|
| ROAD-Waymo | `fusion-20261030-140937-road-waymo-rgbdrop50` | 17 | 66.07% | 0.8027 |
| LOKI | `fusion-20261030-140937-loki-rgbdrop50` | 12 | 74.68% | 0.3853 |

Their checkpoint hashes and original transfer metrics are frozen in [the E002 transfer manifest](../E002-multimodal-reference/transfer-comparison.json). Keep the following settings unchanged:

| Component | Setting |
|---|---|
| Branch descriptors | Geometry64, RGB128, LiDAR128; affine LayerNorm per branch |
| LiDAR readout | One learned cross-attention query, 20m target-centered point features, width128, four heads, one spatial block |
| Fusion / temporal | 323→128; two bidirectional RoPE blocks, four heads, FFN512 |
| Dropout | Spatial/temporal0.1; normalized RGB descriptor0.5; LiDAR/geometry descriptor0 |
| Optimizer | AdamW, LR3e-4, weight decay1e-4, betas(0.9,0.999), epsilon1e-8 |
| Training | Float32, whole-track effective batch64, three-epoch warmup, clip1.0, maximum100 epochs, patience15, seed0 |
| Selection | Full natural source-validation four-class track-weighted macro-F1; exact ties use lower CE, then earlier epoch |

Patience and aggregation retain the existing shared implementation. Different packing may change stochastic dropout draws, so this is not a promise of identical control training trajectories. E002e does not silently change this reference recipe.

## Mask sampler

Natural availability is `A[t,m]`, with branches geometry G, RGB R and LiDAR L. G is the OR of the six existing geometry validity groups, not a guarantee that every geometry field exists.

| Choice | Fixed value |
|---|---|
| Clean versus augmented | 0.5 / 0.5 per track presentation |
| Full-track persistence | Probability0.25 conditional on augmentation |
| Other persistence lengths | Remaining0.75 uniformly over `{1,5,10,20}` strictly less than track length T |
| Shortest track | If no finite length exists, use full-track persistence only |
| Retained branch count per block | `P(k=0,1,2,3)=(0.1,0.3,0.3,0.3)` |
| Branch subset | Uniform among subsets of the selected size; independent per block |
| Blocks | Start at first slot; truncate final block; no random phase |
| Candidate attempts | At most3 complete draws; then restore the original view of the same track |

Sample one persistence length per augmented track. The chosen branch subset is constant within a block, including full-track blocks. Define effective availability `E[t,m] = A[t,m] AND S[block(t),m]`. Never invent observations or switch to an available branch when a selected branch is absent. k=0 deliberately permits short complete outages; k=3 and intersection with natural missingness may yield unchanged views. Accept unchanged candidates and record them.

Training masks use an independent seed0 RNG keyed by source, epoch and track identity. They do not consume shuffle/model-dropout RNG or depend on cache hits, loading order or worker scheduling. Resampling redraws length/subsets, without redrawing the clean/augmented coin or adding presentations/updates.

## Severity checks

Use the full unpadded context, **never GT**, to accept a candidate. Let `O[t]=any(A[t,:])` and `V[t]=any(E[t,:])`.

1. Require `sum(V) >= ceil(0.25 * sum(O))`.
2. For each contiguous interval where V is false, count the originally observable slots O within that interval. This newly hidden count must be at most `min(10, floor(T/4))`. Natural gaps consume no budget but can connect hidden observations into one resulting gap. T<4 permits no newly fully hidden timestep.
3. After three rejected candidates, restore E=A for that same track; do not relax limits.

The ten-slot cap is2 seconds of grid coverage, not a bound on physical empty-interval duration: existing natural gaps can extend the interval. Record both the resulting interval length and newly hidden count. Retaining25% of observable context cannot guarantee behavioral identifiability. A candidate may retain unlabeled context while hiding every accepted-GT slot; do not use GT to reject it.

The preceding [training-only feasibility audit](../E002-multimodal-reference/README.md#training-gt-runs-and-temporal-mask-feasibility-2026-11-02) motivated finite lengths. A disposable five-presentation NumPy simulation with these agreed limits changed44.8% ROAD /42.6% LOKI presentations and fell back on about0.6% /0.8% of augmented draws. It retained roughly82.6% /82.9% of naturally available LiDAR slots overall. These approximate metadata-only draws are feasibility signals, not the final sampler audit, runtime estimates or training results.

## Input semantics and loading

Choose the accepted mask before packing/loading. Reuse it through forward/backward recomputation. Pack using retained LiDAR point counts, skip masked point-feature reads/readout, and preserve track lengths, timestamps/RoPE positions, padding, targets and GT masks. Existing caches stay immutable.

Mask G by zeroing all20 inputs, including its six internal validity flags. Masked branch descriptors remain zero after projections, affine normalization and existing dropout; effective branch validity is0. Retained branches keep their original partial-validity structure. Observation masking has no inverse-keep scaling. Existing descriptor/attention dropout keeps its usual implementation.

This masks **model branches**, not independent physical sensors. Geometry contains 2D bbox fields; retained RGB/LiDAR still use supplied target localization. Cached velocities may depend on adjacent observations. Do not recompute velocities or claim a raw-sensor outage. Keep natural/effective masks separately in local evidence; the model sees only effective validity, without an artificial-mask indicator.

## Evaluation and logging

Every epoch uses existing online training CE/F1 and **natural full source-validation** CE/F1. No new validation metric or additional epoch-level inference pass. Training scores remain changing-weight, dropout-active scores, now on clean/masked observations; they are not fixed-checkpoint train scores directly matched to natural validation difficulty.

Add compact epoch masking statistics: clean/changed/fallback shares, candidate/rejection counts, accepted lengths/subsets, branch coverage, observable-time retention, and resulting empty-gap/newly-hidden summaries. Save detailed masks and post-hoc retained/hidden GT support locally. GT/class/transition summaries never enter the sampler. Keep W&B tidy: existing main curves, compact masking scalars and final tables; checkpoints and full predictions stay local.

Reload `best.pt` and evaluate:

| Evaluation | Models | Availability | Purpose |
|---|---|---|---|
| Source validation | Masked model; existing control metrics | Natural | Source trade-off |
| Opposite validation | Masked model; existing frozen control transfer | Natural, source normalization | Primary transfer comparison |
| Source diagnostics | Masked model and matched E002d control | G-only, RGB-only, LiDAR-only, fixed mixed masks | Secondary missing-input robustness |

The three single-branch diagnostics gate directly, **without fallback or severity repair**. Keep all tracks and GT, including tracks with no remaining input; report the no-input cohort separately. The mixed diagnostic forces the augmented path of the exact training sampler, with the same three-draw checks/fallback, but diagnostic seed10000. Hold these masks fixed across models. Evaluation mode disables ordinary dropout. No diagnostic or target result participates in checkpoint selection.

Retain existing per-class/confusion/observation-stratum outputs for natural evaluations and a compact diagnostic comparison table. Save predictions and mask evidence so numerical results can be reconstructed without inference. Do not modify historical run artifacts.

## Implementation and acceptance

Use thin E003a assembly/launcher code over existing preparation, model, accumulation, lifecycle and evaluation operations. Fresh attempts live under ignored `outputs/experiments/E003a/`; one W&B run per new attempt. Do not resume or automatically retry training. Exact-code CUDA resource verification remains required; masking is not permission to bypass calibrated limits.

Acceptance covers deterministic source/epoch/track draws; no invented availability; finite/full-track boundaries; short tracks; bridged natural gaps; three-failure fallback; unchanged accepted GT/times/context; zero descriptors after affine normalization; consistent point counts; skipped masked point reads; immutable caches; shared diagnostic masks; and prediction-to-metric reconstruction. Run focused tests, the required full CPU suite, bounded checks on both CUDA sources, dependency/link checks where affected, and `git diff --check`. Measure masked/unmasked throughput on matched inputs; do not infer speedup from retained-token ratios.

### Acceptance evidence (2026-11-03)

Implementation revision `47170a4` was transferred through Git and verified on Aalto. Local CPU suite: **253 tests pass, 12 expected skips**; Aalto CUDA suite: **all253 pass**. Dependency lock/installed-package checks, local documentation targets/anchors and `git diff --check` pass. The [metadata-only sampler audit](../../outputs/experiments/E003a/sampler-audit-20261103/summary.json) covers all4,380 ROAD /8,574 LOKI training tracks at mask seed0/epoch1, with repeated-draw equality and no invented-availability, retained-coverage or added-gap violations. Changed views:1,922 /3,651; fallbacks:15 /36; natural LiDAR slots retained:82.80% /82.86%. Counts are realized draws, not assumed sampling probabilities; the input hash and regeneration script remain beside the summary.

Aalto calibration: `outputs/experiments/E003a/calibration/profile.json`, completed on the RTX4080 Laptop GPU using the existing E002 feature environment (PyTorch2.7.1+cu126 /CUDA12.6). Both-source full-track fixture and64-track/102-slot temporal-bound updates pass; peak allocations were312,635,392 ROAD /2,785,686,528 LOKI /325,294,080 shape bytes. These are bounded capacity checks, not full-batch/epoch memory or throughput claims. Packing retains the reference's two-frame spatial chunks and existing token/temporal ceilings. No model, feature or test-set change.

Two separately labeled one-epoch smoke attempts completed under `outputs/experiments/E003a/acceptance-20261103/smoke/`: three source training tracks, two source validation tracks and two opposite-validation tracks per attempt. [ROAD W&B smoke](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/6p4sp4vi) / [LOKI W&B smoke](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/wv6qarsa). Each produced best/last checkpoints,42 PNG/SVG plot pairs, natural source/opposite predictions and all eight source diagnostic model/view cells. Ten prediction archives per source reconstruct saved confusion/CE/accuracy, and control/model effective masks match in all diagnostics. Source/target protected-file checks report no changes. `acceptance-20261103/selection.json` and `verification.json` pin scope and checks. These are integration artifacts, not comparison scores. The ROAD training draws happened to be clean for all three tracks; LOKI changed one. Both sources exercised masked diagnostic inference.

The original CUDA-suite run exposed a pre-existing E001 test that set the RNG but omitted experiment FP32/TF32 backend settings: batch/single logits differed by about4e-6 against its1e-6 tolerance. Applying existing `seed_run(0)` made the test and complete CUDA suite pass; no tolerance or model was changed. Revision `8bfdb79` adds that test setup and the bounded timing script; calibrated training source remains identical to `47170a4`.

### Matched bounded timing

At revision `8bfdb79`, [the disposable timing script](../../scripts/benchmark-e003a.py) opened the full source/opposite/control-transfer assembly, verified existing control evidence, then used the same three short/typical/long **training** fixtures per source. One warmup and two measured updates per natural/masked view; fresh seed0 model/optimizer each time, weights discarded. Each timed update includes warm-cache reads, CPU preparation/packing, transfers, forward/backward and AdamW; model/optimizer creation and initial assembly are reported separately. No W&B or validation/test inference.

| Source | Natural→effective point tokens | Changed tracks | Natural median update | E003a median update | Initial assembly |
|---|---:|---:|---:|---:|---:|
| ROAD | 652,968→652,968 | 0/3 | 0.252s | 0.249s | 30.85s |
| LOKI | 712,568→573,892 | 1/3 | 0.205s | 0.180s | 26.98s |

ROAD happened to draw three clean views at epoch1, so it measures adapter overhead, not a masking speedup. LOKI's measured update is about12% shorter here. These tiny warm-cache single-physical-batch results **do not estimate full-epoch time, cold disk throughput or worst-case speedup**. Two repeats are descriptive only. Raw timings, cache counters, masks and allocation peaks are in Aalto `acceptance-20261103/benchmark-road-waymo/benchmark.json` and `benchmark-loki/benchmark.json`. The metadata audit's roughly83% retained LiDAR slots is not a runtime prediction.

Full comparison training is not part of implementation acceptance. Single-seed evidence will remain exploratory. The [manual launch command](../../README.md#e003a-structured-observation-masking) runs ROAD then LOKI using fresh timestamped attempts; calibration must match the current code.
