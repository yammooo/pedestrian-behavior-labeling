# E005 — Spatial representation upgrades

**Status: LOKI pilot complete, 2026-11-19.** The user approved retiring the old ROAD D-DITR derived cache to free space. E004c remains stopped with its outputs preserved.

## Question and scope

Can task-trained spatial readouts extract more useful pedestrian and scene information than the current RGB CLS / decoded D-DITR representations? Improvement is a hypothesis. E005a owns the DINO spatial change; E005b owns the proposed Tiny-Utonia replacement. Final comparison runs and matched controls must be pinned before training; the new combined model alone cannot isolate the two changes.

Use the frozen E001/E001b populations, 5 Hz full tracks, geometry descriptor, labels, source-only normalization and missing-observation rules. The feature cache covers **LOKI training and validation** (8,574 / 1,861 tracks). The separate transfer check uses [ROAD-Waymo validation](#loki-to-road-waymo-validation). Historical validation and E001 test exposure remain disclosed. No new official test access.

## Tensor flow

The reference temporal path is E002f's one-layer packed **BiLSTM-96**: fusion128 → bidirectional hidden96 per direction → output dropout0.1 → projection192→128 → final LayerNorm128 → classifier128→4. Geometry retains its20→64→64 GELU MLP and branch LayerNorm. Whole tracks remain intact, including unlabeled context; padding is excluded from recurrent processing.

### E005a: DINO spatial RGB

Keep the existing native-only 3× max-box-side square crop, centered with padding, resized to256², and the frozen original DINOv3 ViT-B/16 checkpoint. Use its256 patch tokens ×768 channels; no CLS token enters this variant.

For each patch, compute center offsets from the target box center, divided by the **full** crop width/height: `dx`, `dy`; append `dx²+dy²` and the patch-area fraction overlapped by the target2D box. These four values are geometric cues, not standardized by dataset statistics or LayerNorm. Box overlap is not segmentation. Coordinates follow the actual crop transform, including padding.

`concatenate(LayerNorm768(patch), four geometric cues)` → Linear772→128 → LayerNorm128 → spatial readout. The resulting RGB descriptor is128D, followed by branch LayerNorm and the reference channel dropout0.5. Missing RGB produces a zero descriptor and its own validity flag.

### E005b: Tiny-Utonia LiDAR

Frozen official `utonia_tiny.pth`, SHA256 `e49a1944403d4ec3df459537fb7d5b132326736d0ebdb7c5a74868f3fa50d137`, HF revision `9457f025e36c7001ee962a542f0d27a5b600e1e7`. The [bounded probe](../../outputs/experiments/E005/online-probe-20261117/timing.json) verifies channels18/36/72/144/216 and four stride-2 pools. Author inference code is pinned at `de1f9deef4ee65e508d74347f8d0eb848d878abb`. [Pretraining exposure limitations](../../docs/literature/papers/utonia.md#downloadable-checkpoints-2026-10-16) remain applicable; Tiny is not established as Waymo-free.

Encode the full valid native sweep with the existing outdoor frame convention, coordinate scale0.2 and scaled grid0.01 (5 cm physical); absent color/normals remain zero. Retain stage3 (72 channels, nominal20 cm grid) and stage5 (216 channels, nominal80 cm grid) at native resolutions. Grid size is not receptive-field size, and pooling does not restore missing returns.

After encoding, select a **20 m world-horizontal cylinder**, without height restriction, centered on the pedestrian. Each level contributes its feature plus five cues: relative XYZ/20, squared horizontal distance/400, and whether the representative coordinate lies inside the target3D box. The last flag is a location cue, not pure target membership of the pooled feature.

Per level: feature LayerNorm → concatenate cues → Linear77→128 or221→128 → learned level embedding. Combine the two levels along the **token axis**, apply LayerNorm128, then spatial readout. Output128 → branch LayerNorm; LiDAR descriptor dropout remains0. Missing LiDAR produces zero and its own flag. Do not duplicate coarse features at fine points.

### Shared readout and fusion

One learned128D query; four-head cross-attention to spatial keys/values; output projection and query residual; pre-LayerNorm FFN128→512→128 with GELU and residual. Dropout0.1 acts on attention probabilities, projected attention output, FFN hidden activation and FFN output. No spatial token-to-token trainable self-attention is added. Geometry/RGB/LiDAR descriptors are64/128/128; all enabled descriptors and their availability flags are concatenated then projected to128. Geometry+RGB therefore uses194→128; all three use323→128. Disabled branches contribute neither dummy features nor flags.

## Cache and execution contract

- New caches live under ignored `outputs/experiments/E005/`. Native data, results and the old LOKI cache are preserved; the explicitly approved ROAD D-DITR cache retirement is recorded below.
- Store frozen features as FP16, LiDAR coordinates as FP32. Trainable readouts use FP32. Cache features precede all task-trained projections.
- RGB stores only valid256×768 patch arrays and the input geometry needed to reconstruct positional cues. Existing native associations/crop semantics remain authoritative.
- LiDAR stores each stage once per unique sweep, retaining the union required by all eligible target20 m neighborhoods. Reader reconstructs each target crop; no per-pedestrian feature duplication, dense inverse expansion or bounded-LRU system.
- Cached and future online extraction must share the same transformation and feature definitions. Native roots/provenance remain available; no derived GT enters features.
- The model consumes ragged CPU tokens and bounds each spatial chunk by frame/token limits. Above the profile's `ordinary_tokens` threshold, training recomputes spatial activations from CPU inputs, preserving dropout RNG. Evaluation and smaller training batches use ordinary execution. These execution limits are not a sampling policy.
- Check actual encoded sample sizes, cache/online agreement, finite outputs and a forward/backward batch before population preparation. Keep at least20 GB free during preparation; stop with evidence if the cache cannot fit. Never silently drop tracks/tokens to meet disk limits.
- Prepare source training/validation only; fit no normalization from validation. No automatic retries or continuation from incomplete outputs.

The earlier [storage/online estimate](../../outputs/experiments/E005/cache-online-feasibility-20261117/report.md) projected about136 GB Tiny +59 GB DINO for LOKI before indices/metadata. The four-scenario native build gives a larger estimate: **200.16 GB LiDAR including indices +71.87 GB RGB =272.04 GB**, versus 232 GB free after the bounded build. These remain sample projections, not full-cache totals. This initially blocked the full build; the approved retirement below resolves that shortage. Online ROAD is a later inference-only execution choice, not an authorization to evaluate now.

## Acceptance and commands

Implemented [model](../../src/pedestrian_behavior/models/e005.py): 528,516 trainable parameters for `g-rgb`, 782,660 for `g-rgb-utonia`. Four focused [model tests](../../tests/test_e005_models.py) cover missing branches, padding, FP16 cache inputs, gradients and ordinary/recompute equivalence with identical RNG. The [cache implementation](../../src/pedestrian_behavior/experiments/e005_cache.py) stores scene-shared arrays and uint32 target indices; [cache tests](../../tests/test_e005_cache.py) cover round trips, positional cues, missing slots and explicit integrity verification. Full CPU suite: **308 tests pass, 13 skips** (52.6 s). Full CUDA suite in the existing extractor environment: **308 tests pass** (58.1 s), revision `f48f3f7`. Native-data integration checks pass; details below.

[Preparation command](../../README.md#e005-spatial-feature-preparation). Bounded output on Aalto: `outputs/experiments/E005/loki-cache-bounded-20261117/`. The four scenarios (`193`, `339`, `360`, `572`) cover 97 tracks and 225 sweeps. Preparation took **110.6 s**, storing 1,235 RGB crops, 5,577,304 stage3 tokens and 1,074,069 stage5 tokens. Per-target indices add 19,397,175 / 3,696,192 rows. Total array payload is **1.925 GB**; this is a verification sample, not the full training cache.

[Native acceptance evidence](../../outputs/experiments/E005/bounded-cache-evidence-20261117.json):

- Explicit array/scene integrity verification passes. RGB validity matches the existing E002a cache on all 2,245 sampled slots (1,235 valid).
- Fresh DINO extraction matches eight cached crops exactly. Fresh Tiny extraction matches both selected levels exactly (21,629 /2,584 tokens).
- A complete 51-frame track passes GPU forward/backward for both arms and forced spatial recomputation. This checks wiring, not trained accuracy or production throughput.
- Using the existing 151,040 valid RGB crops refines RGB storage to 59.39 GB; keeping the sampled LiDAR projection gives **259.55 GB total**. With 232.01 GB free, the full build is **47.55 GB short including the 20 GB floor**. LiDAR remains a four-scenario extrapolation; the guard also remains active during writes.

[Build status/timing](../../outputs/experiments/E005/loki-cache-bounded-20261117/cache.json) · [array/count/provenance manifest](../../outputs/experiments/E005/loki-cache-bounded-20261117/manifest.json). The user subsequently authorized deleting only `outputs/experiments/E002a/indexed-road-waymo`. Removal freed **156.23 GB**; its three top-level JSON provenance files were retained separately (27.76 MB). Native data, old LOKI/RGB caches, weights, checkpoints and predictions remain intact. Older ROAD D-DITR inference/training now requires rebuilding that cache.

## Full preparation launch

Full output: `outputs/experiments/E005/loki-cache-full-20261117/` on Aalto; revision `be7a540`, PID `62329`. Preflight free space was **388.23 GB**, versus the conservative 272.04 GB projection plus 20 GB floor. This is a detached cache build, not behavior training. First durable check: 6/523 scenarios, 261/31,290 sweeps and 1,162 valid RGB crops after 109.7 s; 386 GB remained free. Early runtime estimate was 3–4 hours. The build finished in **3 h 31 min**, covering **523 scenarios**, **8,574 training / 1,861 validation tracks**, **151,040 RGB crops** and **31,290 LiDAR sweeps**. Array payload was **224.57 GB**, with **163.47 GB** free near completion. The final cache status and manifest both report complete. [Status](../../outputs/experiments/E005/loki-cache-full-20261117/cache.json) · [Manifest](../../outputs/experiments/E005/loki-cache-full-20261117/manifest.json).

[Deletion receipt](../../outputs/experiments/E005/loki-cache-full-20261117-road-cache-provenance/deletion-receipt.json). The bounded verification and earlier failed storage gate remain recorded above.

## LOKI training pilot

Train `g-rgb-utonia` with geometry, DINO patches and both Tiny-Utonia levels. The same launcher can fit `g-rgb` as a separate spatial RGB pilot. Both start from freshly initialized readouts and BiLSTM-96 with seed 0. Use the frozen LOKI training split and its E001b normalization. Select by full LOKI validation track-weighted macro-F1, then lower CE and earlier epoch. ROAD and held-out tests are outside this pilot. Matched controls for attributing a change to either spatial encoder remain to be fixed before drawing comparison conclusions.

Reuse the E002a source recipe: equal-track cross-entropy, AdamW at 3e-4 with three epochs of linear warmup, 1e-4 weight decay, gradient clipping at 1.0, and 100 epochs with patience 15. Each update covers up to 64 complete tracks via gradient accumulation. The original pilot used one track per LiDAR microbatch and prefetched one batch. Its spatial chunks were bounded at two frames and 65,536 padded tokens. The largest cached LiDAR frame has 31,497 tokens. Later `g-rgb-utonia` launches on Aalto's RTX 4080 Laptop GPU default to the calibrated profile below. The geometry plus RGB arm uses up to four tracks per physical batch. [Training command](../../README.md#e005-loki-training).

On Aalto, launcher revision `36b5a2a` passed a CUDA smoke with the full cache: two 102-frame training tracks in one optimizer update, two 101-frame validation tracks, checkpoint reload, prediction export and offline W&B logging. Peak GPU allocation was 244 MB in this bounded run. Evidence: `outputs/experiments/E005/acceptance/20261118-053924Z-trainer-smoke/` on Aalto. Full pilot training remains user-launched.

The user-launched `outputs/experiments/E005/runs/loki-g-rgb-utonia-s0-20261118-074440` completed 29 epochs in **24.21 h**. `best.pt` selected epoch 14 at **79.36% LOKI validation F1 / 0.361 CE**. Training took **21.67 h**, averaging about **44.8 min per epoch** and **20.1 s per optimizer update**. The 15-epoch patience window ran after the last improvement at epoch 14. Peak allocated GPU memory was 388 MB. The old two-layer Transformer fusion reference reached 74.68% F1. A matched old-feature BiLSTM control remains pending.

### Training throughput benchmark (2026-11-19)

On the Aalto RTX 4080, [the benchmark](../../scripts/benchmark-e005-training.py) replayed the same 64 seed-0 LOKI training tracks and initial weights, with one prefetch and two timed updates per profile in forward and reverse order. File pages for this subset were warmed. The source pilot and training defaults were untouched. Reported memory is peak CUDA allocated/reserved decimal GB.

| Profile | Median update (s) | Exposed loader wait (s) | Peak allocated / reserved (GB) |
|---|---:|---:|---:|
| Baseline: two frames, one track, 65,536 ordinary tokens | 14.62 | 4.51 | 0.34 / 0.96 |
| One million ordinary tokens, otherwise baseline | 11.36 | 4.53 | 3.09 / 3.69 |
| Also use eight frames and 262,144 tokens per spatial chunk | 11.06 | 4.90 | 3.17 / 3.64 |
| Also use four tracks per physical batch | 10.97 | 3.18 | 2.80 / 3.00 |

The higher ordinary threshold saves **22%** end to end with identical measured training CE and F1 on this update. Peak speedup is **1.33×**, or **25% less time**. Adding larger chunks and batches saves only another 3.5% versus the threshold change, within run-to-run variation. With the same inputs preloaded in CPU RAM, baseline and four-track updates take 8.71 and 7.40 s, including host-to-device transfer and training compute. The four-track profile passes both the densest grouped recomputation case (4.69 million padded tokens) and a near-threshold ordinary case (998,728 tokens, 3.14 GB peak allocation).

Raise the ordinary threshold first if the next training run needs this speedup. These are warmed, bounded measurements and do not establish sustained epoch throughput. CPU suite: 310 tests passed, 13 skipped. [Timing and memory evidence](../../outputs/experiments/E005/benchmarks/e005-chunks-20261119/benchmark.json).

The subsequent calibration used measured bytes per padded token to find dense real batches near the RTX 4080's allocator limit. Ordinary execution fit the densest single track (2.116M tokens, 6.72 GB reserved), a 3.275M-token two-track group (9.80 GB), and a 3.405M-token four-track group (10.23 GB). Two and four tracks were nearly tied on the warmed updates. Two tracks reduced padding and allocator reserve and were selected for deployment with two reader workers. The reader now assembles the same FP16 tokens directly, skipping the FP32 intermediate array.

On two held 64-track cohorts, the original pipeline took **18.17 / 14.82 s** and the tuned layout and reader took **5.99 / 5.59 s**. Median **16.50 → 5.79 s** is **2.85× faster**. The fast profile uses two whole tracks per microbatch, eight frames and at most 262,144 padded tokens per spatial chunk, recomputation above 3,274,888 padded tokens per microbatch, and two bounded reader workers. This is the default for `g-rgb-utonia` on Aalto's RTX 4080 Laptop GPU. Use `--no-aalto-fast` for the original execution profile. Batch membership, FP32 model precision and input features are preserved. Changing physical batch shapes can change the order of dropout draws. These short benchmarks do not establish a full-epoch speedup. The calibration run saved its layout and stress measurements before a benchmark wrapper failed in the final paired checks. A separate replay completed those checks. [Calibration](../../outputs/experiments/E005/benchmarks/e005-tune-20261119/benchmark.json) · [Paired replay](../../outputs/experiments/E005/benchmarks/e005-tune-pairs-20261119/benchmark.json).

The fast profile also avoids materializing per-token keys and values. With one query per head, the key projection can be applied to the query before scoring, and the value projection can be applied after the weighted sum. The mask, dropout, trainable parameters and model definition are preserved. On a warmed matched replay this brought a 5.34 s update to 4.72 s and reduced peak reserve from 10.28 GB to 5.46 GB. Seeded CUDA parity checks found a maximum attention output difference of 2.84e-7 and gradient difference of 3.36e-8. [Persistent-update and attention evidence](../../outputs/experiments/E005/benchmarks/e005-fast-verify-20261119/benchmark.json).

The deployed fast profile at `3c77bd4` completed three consecutive 64-track updates using the same model and optimizer, including 3.275M padded tokens in ordinary mode and 4.312M in recompute mode. It peaked at **6.23 GB allocated / 6.99 GB reserved**. Four-track folded attention also fit a 4.691M-token ordinary stress batch at **10.03 GB reserved**. Warm comparisons had similar model time near 4.44 s after subtracting exposed loader wait. The deployed two-track profile keeps the lower memory and padding load. [Final verification](../../outputs/experiments/E005/benchmarks/e005-fast-folded-20261119/benchmark.json).

## LOKI to ROAD-Waymo validation

Evaluate the completed `g-rgb-utonia` run's LOKI-selected `best.pt` once on the frozen ROAD-Waymo validation split (1,110 tracks across 78 clips). Use the checkpoint's LOKI geometry statistics throughout. Compute DINO patch features and Tiny-Utonia stage features from native ROAD observations online. Reuse each encoded sweep for the eligible pedestrians in its clip, then release that clip's tensors. Only predictions, metrics and run evidence are saved to disk. [Command](../../README.md#e005-road-waymo-transfer).

Compare track-weighted macro-F1, CE and class precision, recall and F1 against the historical [E002 transfer results](../E002-multimodal-reference/transfer-results.md). The old LOKI Transformer changes both the backbone and feature representation. The matched LOKI BiLSTM control is still needed to attribute a change to the combined spatial upgrade. This is exploratory transfer to an already studied validation split. Tiny-Utonia's exact Waymo pretraining exposure remains unknown.

The CPU integration check uses synthetic native observations to test sweep reuse, missing modalities, both Utonia stages and complete-track prediction. The validation index was checked against the frozen preparation. Native ROAD and CUDA execution await the end of source training.
