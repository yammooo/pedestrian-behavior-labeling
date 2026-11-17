# E005 — Spatial representation upgrades

**Status: full LOKI cache preparation running, 2026-11-17.** The user approved retiring the old ROAD D-DITR derived cache to free space. E004c remains stopped with its outputs preserved. No behavior training or ROAD evaluation is launched by this increment.

## Question and scope

Can task-trained spatial readouts extract more useful pedestrian and scene information than the current RGB CLS / decoded D-DITR representations? Improvement is a hypothesis. E005a owns the DINO spatial change; E005b owns the proposed Tiny-Utonia replacement. Final comparison runs and matched controls must be pinned before training; the new combined model alone cannot isolate the two changes.

Use the frozen E001/E001b populations, 5 Hz full tracks, geometry descriptor, labels, source-only normalization and missing-observation rules. Prepare **LOKI training and validation only** (8,574 / 1,861 tracks). ROAD features will be computed online for a later transfer evaluation; its evaluation split and command remain TBD. Historical validation and E001 test exposure remain disclosed. No new official test access.

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

Full output: `outputs/experiments/E005/loki-cache-full-20261117/` on Aalto; revision `be7a540`, PID `62329`. Preflight free space was **388.23 GB**, versus the conservative 272.04 GB projection plus 20 GB floor. This is a detached cache build, not behavior training. First durable check: 6/523 scenarios, 261/31,290 sweeps and 1,162 valid RGB crops after 109.7 s; 386 GB remained free. Early runtime estimate is 3–4 hours, subject to scene variation. Completion counts and final footprint remain pending.

[Deletion receipt](../../outputs/experiments/E005/loki-cache-full-20261117-road-cache-provenance/deletion-receipt.json). The bounded verification and earlier failed storage gate remain recorded above.
