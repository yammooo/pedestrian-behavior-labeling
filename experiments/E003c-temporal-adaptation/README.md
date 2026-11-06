# E003c — Head versus temporal adaptation

**Protocol: original-source initialization, 2026-11-06. Corrected comparison not run yet.**

## Question and scope

Starting directly from the **original source-trained E002d network, including its original head**, does adapting the temporal block improve target behavior labeling over fine-tuning only that head?

Two seed0 ROAD-Waymo → LOKI fits use the E002d geometry + RGB + D-DITR model with branch LayerNorm and two128-wide temporal blocks. Both begin at the original direct-transfer **46.50% LOKI validation F1**, not an already target-adapted checkpoint. There is no random head replacement or preliminary target-head training. E003b remains the separate fresh-head diagnostic.

## Shared initialization

Source checkpoint: `outputs/experiments/E002d/runs/fusion-20261030-140937-road-waymo-rgbdrop50/best.pt`, selected source epoch17, SHA256 `d1f83183dcff61f8d9925a74c2c76811d1eca225119e588df8aa3770720e42a8`.

| Component | `head` | `temporal-head` |
|---|---|---|
| Geometry/RGB/LiDAR branches and fusion | Original source weights, frozen | Identical source weights, frozen |
| Two temporal Transformer blocks | Original source weights, frozen | Identical source weights, trainable |
| Final LayerNorm | Original source weights, frozen | Identical source weights, trainable |
| Linear classifier | Original source head, trainable | Identical original source head, trainable |

No E003b target-trained head checkpoint is loaded. Both optimizers start fresh. Initialization is recorded as `source-checkpoint`; epoch0 logits must reproduce the original E002 direct-transfer predictions on the same target population.

## Fixed recipe

| Setting | Both arms unless stated |
|---|---|
| Data | Frozen E002 LOKI training/validation populations:8,574 /1,861 tracks; complete tracks and natural availability |
| Input normalization | Original ROAD training statistics on both LOKI splits |
| Supervision / metric | Existing accepted-GT mean-track CE / four-class track-weighted macro-F1 |
| Optimizer | AdamW, weight decay1e-4, betas(0.9,0.999), epsilon1e-8 |
| Learning rates | Head3e-4 in both; temporal blocks and final LayerNorm3e-5 when trainable |
| Budget | Maximum100 target-training epochs; patience15; batch64 whole tracks with final partial batch retained |
| Schedule / clipping | Three-epoch linear warmup, then fixed LR; global gradient clipping1.0 |
| Dropout | Disabled throughout both arms, including trainable temporal layers |
| Seed / precision | Seed0; existing deterministic float32 settings; identical canonical track order and persistent seeded shuffle |
| Selection | Epoch0 is a candidate; highest target-validation F1, lower CE then earlier epoch for exact ties; patience resets only on strict F1 improvement |

Full target-training labels and target-validation model selection are used. Both validation sets informed earlier development; this is exploratory supervised adaptation, not zero-shot or held-out evidence. Tests stay closed. A gain would establish benefit from adapting this block under the recipe, not isolate temporal mismatch, annotation differences or capacity as the cause.

## Features and launch

Both feature stages depend only on the frozen original source network. Reuse the source-only post-final-LayerNorm128-D caches produced during E003b for `head`; they contain no target-head training. Reuse the original source model's pre-temporal fusion128-D cache for `temporal-head`. No encoder inference, point-cache rebuild or native-data reprocessing is needed when these caches are complete.

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
bash scripts/run-e003c-sequential.sh
```

The launcher uses `outputs/experiments/E003c/fusion-features-road-waymo/` by default, reusing a completed source cache or preparing it once if absent. Fresh model outputs live under `outputs/experiments/E003c/runs/source-adaptation-<UTC timestamp>/`, with one W&B run per arm, update/epoch CE, online training F1, target-validation metrics, epoch0 history, selected/last checkpoints and predictions. Training metrics use changing weights; dropout is off. Models/predictions remain local. Existing checkpoints/data are read-only; no automatic retry or resume.

## Acceptance and results

Verify exact original-head initialization in both arms, original direct-transfer epoch0 logits/F1, trainable parameter sets (516 /397,316), disabled dropout, LR warmup ratio, shared loader order, GT/padding, checkpoint reload and a64×102 CUDA update. Check both actual cache paths with short GPU/W&B fixtures before the user launches full training.

No results from a target-adapted-head initialization belong to this comparison. Non-comparison artifacts remain outside active run/review paths under ignored `outputs/experiments/E003c/superseded/`.
