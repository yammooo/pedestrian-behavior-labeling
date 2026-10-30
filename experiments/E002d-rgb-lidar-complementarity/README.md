# E002d — RGB–LiDAR complementarity

**Prepared 2026-10-30; CUDA calibration complete, no E002d training or test evaluation yet.** This is a bounded source-validation comparison within E002, not a new architecture search.

## Question and comparison

Does RGB add useful behavior evidence to the best E002b geometry + LiDAR development model when branch normalization and temporal capacity are matched? Does regularizing the RGB descriptor change that answer?

| Arm | RGB branch dropout | New seed0 attempts |
|---|---:|---:|
| `g-dditr / temporal-small` | No RGB branch | Existing E002b reference, ROAD and LOKI; no new training |
| `g-rgb-dditr / temporal-small` | 0.0 | ROAD and LOKI |
| `g-rgb-dditr / temporal-small` | 0.5 | ROAD and LOKI |

Run the **four new attempts sequentially**, one model at a time. The shared-multiple-model E002b trainer is not used: its measured benefit was too small for this comparison and its interruption cause is unresolved. Preserve the earlier attempts and failure evidence.

The `0.0` fusion comparison against `g-dditr` tests the added RGB descriptor **and its availability flag together**. The `0.5` versus `0.0` fusion comparison isolates RGB branch dropout. There is no LiDAR branch dropout or broader dropout grid. E002c found only a single-seed RGB-branch-dropout signal; it does not establish that 0.5 is optimal or that it will help fusion.

## Fixed protocol

Use the frozen E002a population, split, four-class GT, E001b geometry, 3× DINOv3 CLS cache and indexed D-DITR feature cache, with source-training normalization only. Keep 3D-only and missing-RGB tracks. E002b `temporal-small` supplies final affine LayerNorm per enabled branch (mask after normalization), geometry width64, RGB width128, LiDAR spatial width128, fusion/temporal width128, two bidirectional temporal blocks, four heads and FFN512. The fusion input is 323 values: three descriptors plus their three validity flags. The LiDAR readout remains one learned query over selected encoded points within20 m; no encoder fine-tuning or new cache is involved.

Fix temporal and spatial dropout at0.1. For the 0.5 arm, apply track-consistent dropout only to the normalized RGB descriptor before its validity mask and fusion; leave its availability flag intact. Reuse the E002a/E002b recipe: seed0, float32, complete-track effective batch64, AdamW LR3e-4 and weight decay1e-4, three-epoch warmup, clip1.0, maximum100 epochs, patience15. Select best checkpoint by source-validation four-class track-weighted macro-F1, with CE then earlier epoch breaking exact ties. Validation uses the full source-validation split. Test sets and opposite-source results do not select models or variants.

Primary development comparison: selected source-validation track-weighted macro-F1. Also report selected CE, per-class scores, GT/track denominators, RGB-availability strata and runtime. Compare source-matched runs only. The E002b LOKI `g-dditr` reference was interrupted after epoch23, although its selected checkpoint was evaluated; disclose this limitation when comparing. If code or input semantics differ from that control, repeat the control under matched settings before attributing a difference to RGB.

## Decision after the pilot

Inspect whether fusion improves F1 over LiDAR on **both** sources and whether the gain survives missing-RGB and class-level inspection; compare 0.5 with 0.0 separately. Seed0 validation is exploratory. A promising arm needs matched repeat seeds before a final within-dataset or cross-dataset claim. If neither fusion arm helps convincingly, retain `g-dditr / temporal-small` as the working candidate and spend no more runs on RGB dropout here. The later temporal masking test, representation/efficiency work and heterogeneous supervision remain separate decisions. No held-out or cross-dataset inference is part of this pilot.

The [shared E002 safeguards](../README.md#strict-zero-shot-access), [E002b control](../E002b-normalization-capacity/README.md) and [E002c RGB evidence](../E002c-rgb-regularization/README.md) retain their original access and pretrained-exposure qualifications. E002a's DINO and D-DITR checkpoint provenance carries into E002d.

## Execution and evidence

Use the [sequential launch command](../../README.md#e002d-rgb--lidar-complementarity) on Aalto revision `d81a97a`. Each attempt gets a fresh directory under ignored `outputs/experiments/E002d/runs/`, its own W&B run, history, best/last checkpoints, source-validation predictions, metrics and plots. No checkpoint or full prediction upload. The feature caches are read-only.

The completed CUDA calibration is `outputs/experiments/E002d/calibration-20261030-sequential-ready/profile.json` on Aalto, SHA-256 `c1d923c894f56c23f7bd7d3d831b0d8633f30e933b9a7380f63719aad4c62bd8`. It covers one real full-track combined-arm forward/backward per source plus a64-track temporal shape check with empty LiDAR inputs. Maximum allocation **in those bounded checks** was2,785,686,016 bytes against the9,711,719,546-byte ceiling; this is not the peak of a production training batch or epoch. It uses the completed E002a indexed profile for the existing point-token/packing limits and performs no behavior training or W&B logging. The earlier failed profiles, `calibration-20261030-sequential` and `calibration-20261030-sequential-fixed`, remain: the first incorrectly extrapolated a single-frame reference memory peak by squared track length; the second discarded the LiDAR flag while selecting a fixture. Both failed before any E002d training. The corrected code and focused regression tests are in revision `d81a97a`. If execution code or backend changes, rerun calibration in a fresh directory before training.

Record attempted runs, failures and outcomes here after execution.
