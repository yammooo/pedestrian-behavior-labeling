# E002d — RGB–LiDAR complementarity

**Status: the four-run fusion comparison is complete; two LiDAR-dropout controls are calibrated and ready for manual launch.** The original comparison used source validation for selection. The later [E002 umbrella cross-dataset validation](../E002-multimodal-reference/transfer-results.md) is exploratory; no held-out test inference was run.

## Question and comparison

Does RGB add useful behavior evidence to the best E002b geometry + LiDAR development model when branch normalization and temporal capacity are matched? Does regularizing the RGB descriptor change that answer?

| Arm | RGB branch dropout | New seed0 attempts |
|---|---:|---:|
| `g-dditr / temporal-small` | No RGB branch | Existing E002b reference, ROAD and LOKI; no new training |
| `g-rgb-dditr / temporal-small` | 0.0 | ROAD and LOKI |
| `g-rgb-dditr / temporal-small` | 0.5 | ROAD and LOKI |

Run the **four new attempts sequentially**, one model at a time. The shared-multiple-model E002b trainer is not used: its measured benefit was too small for this comparison and its interruption cause is unresolved. Preserve the earlier attempts and failure evidence.

The `0.0` fusion comparison against `g-dditr` tests the added RGB descriptor **and its availability flag together**. The `0.5` versus `0.0` fusion comparison isolates RGB branch dropout. The four-run fusion comparison had no LiDAR branch dropout or broader dropout grid. E002c found only a single-seed RGB-branch-dropout signal; it does not establish that 0.5 is optimal or that it will help fusion.

## Fixed protocol

Use the frozen E002a population, split, four-class GT, E001b geometry, 3× DINOv3 CLS cache and indexed D-DITR feature cache, with source-training normalization only. Keep 3D-only and missing-RGB tracks. E002b `temporal-small` supplies final affine LayerNorm per enabled branch (mask after normalization), geometry width64, RGB width128, LiDAR spatial width128, fusion/temporal width128, two bidirectional temporal blocks, four heads and FFN512. The fusion input is 323 values: three descriptors plus their three validity flags. The LiDAR readout remains one learned query over selected encoded points within20 m; no encoder fine-tuning or new cache is involved.

Fix temporal and spatial dropout at0.1. For the 0.5 arm, apply track-consistent dropout only to the normalized RGB descriptor before its validity mask and fusion; leave its availability flag intact. Reuse the E002a/E002b recipe: seed0, float32, complete-track effective batch64, AdamW LR3e-4 and weight decay1e-4, three-epoch warmup, clip1.0, maximum100 epochs, patience15. Select best checkpoint by source-validation four-class track-weighted macro-F1, with CE then earlier epoch breaking exact ties. Validation uses the full source-validation split. Test sets and opposite-source results do not select models or variants.

Primary development comparison: selected source-validation track-weighted macro-F1. Also report selected CE, per-class scores, GT/track denominators, RGB-availability strata and runtime. Compare source-matched runs only. The E002b LOKI `g-dditr` reference was interrupted after epoch23, although its selected checkpoint was evaluated; disclose this limitation when comparing. If code or input semantics differ from that control, repeat the control under matched settings before attributing a difference to RGB.

## Decision after the pilot

Inspect whether fusion improves F1 over LiDAR on **both** sources and whether the gain survives missing-RGB and class-level inspection; compare 0.5 with 0.0 separately. Seed0 validation is exploratory. A promising arm needs matched repeat seeds before a final within-dataset or cross-dataset claim. If neither fusion arm helps convincingly, retain `g-dditr / temporal-small` as the working candidate and spend no more runs on RGB dropout here. The later temporal masking test, representation/efficiency work and heterogeneous supervision remain separate decisions. No held-out or cross-dataset inference is part of this pilot.

The [shared E002 safeguards](../README.md#strict-zero-shot-access), [E002b control](../E002b-normalization-capacity/README.md) and [E002c RGB evidence](../E002c-rgb-regularization/README.md) retain their original access and pretrained-exposure qualifications. E002a's DINO and D-DITR checkpoint provenance carries into E002d.

## Execution and evidence

Use the [sequential launch command](../../README.md#e002d-rgb--lidar-complementarity) on Aalto revision `d81a97a`. Each attempt gets a fresh directory under ignored `outputs/experiments/E002d/runs/`, its own W&B run, history, best/last checkpoints, source-validation predictions, metrics and plots. No checkpoint or full prediction upload. The feature caches are read-only.

The completed CUDA calibration is `outputs/experiments/E002d/calibration-20261030-sequential-ready/profile.json` on Aalto, SHA-256 `c1d923c894f56c23f7bd7d3d831b0d8633f30e933b9a7380f63719aad4c62bd8`. It covers one real full-track combined-arm forward/backward per source plus a64-track temporal shape check with empty LiDAR inputs. Maximum allocation **in those bounded checks** was2,785,686,016 bytes against the9,711,719,546-byte ceiling; this is not the peak of a production training batch or epoch. It uses the earlier E002a indexed execution profile with two-frame spatial chunks; the later four-frame optimized profile was not used for these runs. Point-feature and model semantics remain the same, but this limits throughput conclusions. Calibration performed no behavior training or W&B logging. The earlier failed profiles, `calibration-20261030-sequential` and `calibration-20261030-sequential-fixed`, remain: the first incorrectly extrapolated a single-frame reference memory peak by squared track length; the second discarded the LiDAR flag while selecting a fixture. Both failed before any E002d training. The corrected code and focused regression tests are in revision `d81a97a`. If execution code or backend changes, rerun calibration in a fresh directory before training.

## Results

See the [concise comparison and overfitting review](results-review.md) for the modality/model table, two summary graphs, W&B links and Aalto artifact paths. The RGB-dropout0.5 fusion arm gains over dropout0 on both sources, but does not clearly beat the matched geometry + LiDAR reference: it is effectively tied on ROAD F1 and lower on LOKI. Validation CE still worsens late in every E002d run.

## Follow-up: LiDAR branch regularization

The four fusion attempts above are complete. The two-run control is now prepared and calibrated. It tests whether dropout on the **normalized LiDAR descriptor** changes the geometry + LiDAR model's generalization: `g-dditr / temporal-small`, LiDAR descriptor dropout0.5, ROAD-Waymo and LOKI, seed0. No RGB branch is present. This is motivated by the remaining train/validation gap; it is not a broader dropout search or a replacement for the completed fusion comparison.

Apply `Dropout1d(0.5)` after LayerNorm of the LiDAR descriptor, with one channel mask shared across all time steps in each track. The LiDAR validity flag is appended unchanged; missing/padded descriptors are zeroed after dropout. Geometry receives no descriptor dropout. Spatial and temporal dropout stay 0.1. Keep the E002d 100-epoch, patience-15 source-validation recipe, seed, model widths, normalization, optimizer and caches unchanged. Names end in `lidardrop50`.

After the two runs, compare their best source-selected checkpoints on the opposite dataset's **validation** split, labeled exploratory/target-informed development. Do not run held-out test evaluation as part of this follow-up.

The completed CUDA calibration uses the accepted four-frame E002a profile at `/home/user20/projects/e002a-execution-optimizations-20261023/calibration-chunk4-final/profile.json` only as its packing-limit reference. The calibrated profile is `/home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation/outputs/experiments/E002d/calibration-lidar-dropout-20261102/profile.json`, SHA-256 `d7ffef873d6947612af3f3a29999d122ccbe31abb7b296b2dd4fc4f4bd6cc2a5`. It covers only `g-dditr / temporal-small / LiDAR dropout 0.5` on the RTX 4080 at code revision `89b38e2`. The bounded one-track forward/backward updates peaked at 317,553,664 bytes for the 100-slot ROAD fixture and 2,997,184,512 bytes for the 102-slot LOKI fixture; the 64-track/102-slot shape check peaked at 289,374,720 bytes. The conservative packing preflight was 5,300,867,072 bytes against the 9,711,719,546-byte ceiling. These are calibration checks, not full-batch training peaks or epoch-time estimates. The exact single-run trainer accepts this profile. The older two-frame E002d profile is not valid for this follow-up.
