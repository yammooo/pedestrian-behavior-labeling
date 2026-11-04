# E003b — Target-head adaptation

**Agreed 2026-11-04; implementation and bounded acceptance in progress.** Full training remains user-launched. This diagnostic follows the [E003a masking results](../E003a-structured-observation-masking/README.md#results).

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

Acceptance status and exact runnable revision: **TBD**. No E003b comparison result exists yet.
