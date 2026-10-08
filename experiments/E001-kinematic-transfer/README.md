# E001 — Kinematic transfer

Status: **Planned** for training. Created 2026-10-05 from the 2026-10-02 proposal. Protocol agreed 2026-10-06; input/model comparison revised 2026-10-08. Data preparation/setup, ROAD association acceptance, both models and training/evaluation are implemented with acceptance checks below. No comparison training or comparison results.

## Question and controls

Which kinematic information helps frame-state labeling and cross-dataset transfer, and does full-track learned temporal context improve it? Compare motion, added trajectory, added ego-relative interaction and a raw-state control within ROAD-Waymo and LOKI and in both transfer directions. This is the first RQ1 evidence/temporal-context diagnostic and the single-dataset end-task reference for RQ2. It uses a kinematic subset of the [study contract](../../docs/datasets/README.md), without settling the final architecture or contribution. Follow the [shared evaluation rules](../README.md) and [research questions](../../docs/research.md).

For each feature set, hold inputs, projection, samples, splits, normalization, classifier design, loss, optimizer, selection rule and budget fixed between A and B. Across feature sets, change only inputs and the first encoder layer's input dimension; retain the same eligible population and protocol. Parameters are independently trained; B adds recurrent capacity. No RGB, raw LiDAR, scene encoder, pedestrian yaw/dimensions, acceleration, factorized heads, extra datasets, distillation, modality dropout or adaptation enters E001.

## First two-dataset diagnostic

| Train dataset | Evaluate dataset | Purpose |
|---|---|---|
| ROAD-Waymo | Held-out ROAD-Waymo test | Within-dataset learnability |
| LOKI | Held-out LOKI test | Within-dataset learnability in the reverse direction |
| ROAD-Waymo | Held-out LOKI test | Camera-selected supervision toward a 3D-first population |
| LOKI | Held-out ROAD-Waymo test | Reverse transfer and possible asymmetry |

Each source-trained model is selected only on its source validation split, then evaluated on both datasets. Four feature sets × two models × two sources give **16 training runs and 32 evaluation cells**. Target training data, including unlabeled adaptation, and target-based model selection are excluded. Source normalization is reused unchanged at target inference.

## Samples and eligibility

One sample is one verified pedestrian identity over its **complete available temporal extent within a scenario**, from its first to last native 2D or 3D observation. Keep internal gaps and unlabeled extensions. Add no context before/after that extent and do not stitch identities across scenarios.

- **LOKI:** join `(scenario, track_id)` across native 2D and 3D observations.
- **ROAD-Waymo:** retain the ROAD tube and its original annotations. Extend context with native FRONT camera observations of the same camera ID and native 3D observations of its unique, officially associated LiDAR ID. Require scene-scoped identity consistency; ambiguous/conflicting associations fail verification. No IoU or inferred identity links. Native extensions add observations, never ROAD behavior GT.

Preserve observation timestamps, source/frame/annotation IDs, native geometry/types, identity/association provenance and original same-frame pairing masks. ROAD CSV `has_3d_box` continues to describe export pairs; a separate native-observation mask describes extended geometry. Identical repeated observations become one timestep with all annotation IDs retained; conflicting duplicates fail explicitly. Annotated 2D availability includes native boxes, with ROAD and Waymo geometry/provenance distinguished.

After the [5 Hz selection](#first-baseline-temporal-representation), an eligible sample has at least one usable pedestrian 3D position, at least one accepted behavior label anywhere in the sample, and a verified ego anchor pose. Position and GT need not occur together. Include 3D-only LOKI, single-observation and single-label tracks; there is no minimum duration or paired-coverage threshold. Exclude tracks lacking usable input, accepted GT or an anchor, and report separate/overlapping reasons and counts. An unexpectedly empty-GT training sample is an error, not a silent skip.

## Accepted four-state projection

This mapping is accepted for E001, not asserted as a universal equivalence of native taxonomies. Native meanings and limitations remain in [LOKI](../../docs/datasets/loki.md#semantic-limits) and [ROAD-Waymo](../../docs/datasets/road-waymo.md#documented-annotation-form).

Prepared tracks retain unchanged native supervision; E001's downstream target policy derives the labels/GT mask and eligible view, rather than saving this temporary projection into the common track record. The [reader/track/provenance schema](../../docs/datasets/README.md#reader-and-prepared-track-schema) is implemented for the combined reader/preparation increment. The complete native conflict audit and declared observation overrides below still precede frame filtering; sampling cannot hide unresolved native conflicts. Target policy, eligibility/split manifests and source-training normalization must be versioned together for reproducibility.

| ROAD-Waymo native action | LOKI original current-frame action | Output / class index |
|---|---|---|
| `Mov`, `MovAway`, `MovTow` | `Moving` | `MOVING` / 0 |
| `Stop` | `Stopped` | `STOPPED` / 1 |
| `Wait2X` | `Waiting to cross` | `WAITING_TO_CROSS` / 2 |
| `Xing`, `XingFmLft`, `XingFmRht` | `Crossing the road` | `CROSSING` / 3 |

Discard `PushObj` from the E001 action projection, while retaining the pedestrian, its context and other action labels. Map the remaining native action list to a set of output states: direction variants and simultaneous movement labels collapse to MOVING, and a unique mapped state is retained alongside any unmapped co-label. If the mapped set is exactly `{CROSSING, MOVING}`, use `CROSSING`. Unmapped-only actions (including `PushObj` alone), no native action and missing annotations have no four-state GT. Preserve original action lists and mapping/correction provenance; report exclusions and class/track denominators after resampling. Never use location `xing` as action `Xing`.

The user accepted the [two native conflict corrections](../../docs/datasets/road-waymo.md#movementstop-conflict-inspection-2026-10-06) on 2026-10-08: `train_00383` tube `7fc2c418-f760-496a-927c-717d8df6ad06`, frames 81–129 → MOVING; `train_00425` tube `e1dce432-b655-4627-bea9-8f7e0a5f17be`, frames 1–10 → STOPPED. The versioned [override artifact](behavior-overrides.json) scopes these 59 native observations by source checksum, identity, frame/timestamp bounds and expected actions. Audit overrides before resampling, rather than applying a universal priority or masking conflicts; native supervision stays unchanged and no GT is added to native-only extensions. Any additional unresolved mapped-state conflict must pass an explicit semantic audit before execution. Do not infer corrective GT from an automatic kinematic threshold.

LOKI uses original current-frame actions, without its paper's four-frame/0.8 s future-target shift. Its `intended_actions` column is read only as GT, never as an input. Timeline/boundary inspection remains useful for understanding semantic mismatch, particularly Stop/Wait2X; it does not silently redefine this projection after scores are seen.

## First-baseline temporal representation

Use a **track-aligned 5 Hz grid**: `t_j = t_0 + j × 200 ms`, where `t_0` is the first native union observation and the last tick is at or before the native endpoint. A final fractional interval does not add an endpoint tick. Complete extent means retaining the union span rather than cropping to consecutive labeled/paired runs.

At each tick choose the timestamp-nearest recorded scene frame within **75 ms**, restricted to the native extent; ties choose the earlier frame. Read available modalities and GT at that selected frame. Do not independently search for a nearby labeled pedestrian observation. With no qualifying scene frame, keep the grid position as missing. No geometry or GT interpolation. Retain nominal grid time, actual selected source time and source frame ID separately.

ROAD-Waymo uses verified physical timestamps, not alternate frame indices. LOKI uses filename ordinal `suffix / 2` at documented nominal 5 Hz until physical timestamps are verified; its elapsed times and derivatives retain that qualification. [Native extent statistics](../../docs/datasets/README.md#outstanding-dataset-checks) predate this eligibility/grid/projection and are sizing evidence, not final sample counts.

### Velocity derivation

Derive both pedestrian and ego velocity from resampled canonical positions and **actual source timestamps** (nominal for LOKI). For a valid current position, use only immediately adjacent grid positions: both valid neighbors give the time-weighted centered derivative, one valid neighbor gives a one-sided derivative, and neither gives missing velocity. A missing current position has missing velocity. Do not bridge invalid positions, search farther neighbors or use observations outside the extent.

With `h_minus = t - t_prev` and `h_plus = t_next - t`:

```text
v = h_plus / (h_minus + h_plus) * (p - p_prev) / h_minus
  + h_minus / (h_minus + h_plus) * (p_next - p) / h_plus
```

On a uniform grid this is the centered secant. Positive, ordered source times are required. This identical preprocessing gives Model A short temporal support; it is frame-independent learned processing, not strictly single-observation inference. Model B additionally learns from the whole available track.

## Coordinates, features and missing inputs

First apply the verified **full native 3D-to-world transform**. Then express horizontal positions in one fixed frame per track: origin at the ego world position at the first timestep, +x along initial ego heading and +y left. Rotate world xy by negative initial ego yaw. Pedestrian/ego positions and world-motion velocities use this same frame; it does not rotate with the moving ego. Ego heading is `delta_ego_yaw = yaw(t) - yaw(t_0)`.

Use native z/roll/pitch where required by the transform, then retain xy model features. Transform in float64, normalize, and cast model inputs to float32. Units are metres, seconds and radians after verification. A pedestrian position is usable only when the required geometry, pose and transforms are valid; ego pose validity covers position and heading. An invalid initial anchor excludes the track.

### Four input configurations

All vectors below have two components in the fixed ground-plane frame. Derive them from the saved, unnormalized track arrays when constructing model inputs.

| Feature set | Numeric inputs, in order | Purpose |
|---|---|---|
| **K — Motion** | `v_ped` | Pedestrian motion alone |
| **K+T — Trajectory** | `v_ped`, `delta_p_ped` | Added explicit trajectory geometry |
| **K+T+R — Interaction** | `v_ped`, `delta_p_ped`, `r`, `v_rel` | Added ego–pedestrian interaction |
| **RAW — Raw state** | `p_ped`, `v_ped`, `p_ego`, `v_ego` | Control for information lost by structured features |

- `delta_p_ped(t) = p_ped(t) - p_ped(first valid)`, using the first valid pedestrian position in the selected full track.
- `r(t) = p_ped(t) - p_ego(t)`.
- `v_rel(t) = v_ped(t) - v_ego(t)`.

Append one unscaled validity flag per vector after all numeric columns, in the same vector order. Total dimensions are **3, 6, 12 and 12** for K, K+T, K+T+R and RAW. Displacement requires the current and reference pedestrian positions; relative position/velocity require both corresponding pedestrian and ego measurements. RAW uses the saved position/velocity masks (`ego_pose_valid` for ego position). Missing inputs remain missing before normalization; no gap filling. None of the four sets includes ego yaw or its sine/cosine. Initial ego alignment still supplies an ego-related orientation cue even in K; ego poses remain necessary for native coordinate transforms.

For each source and feature set, compute each numeric column's mean and population standard deviation over its valid source-training timesteps only; reuse those statistics for A/B, validation, test and target. Use mean 0/scale 1 if a feature has no valid training values and scale 1 for a zero/near-zero standard deviation; record the condition and numerical threshold in the effective config (threshold **1e−8**, inclusive). Fill missing numeric values with **0 after normalization**. Validity flags preserve partial missingness. There are no learned missing vectors or separate position/velocity encoders.

Keep padding, 2D/3D/RGB availability, GT and provenance masks separately as metadata. GT validity is not an input. RGB files do not imply an annotated box or a usable crop. Dataset-native `stationary`, behavior/intention fields, vehicle-state labels, destinations and other semantic annotations are excluded from input; velocity is derived by the common rule, not supplied by native semantic fields.

## First diagnostic baseline

```text
Model A: selected features + masks → joint MLP → classifier → framewise states
Model B: selected features + masks → joint MLP → whole-track BiLSTM → classifier → states
```

| Component | Agreed architecture |
|---|---|
| Joint encoder, A and B | Linear D→64, ReLU, dropout 0.1, linear 64→64, ReLU; D is the feature-set dimension including masks |
| BiLSTM, B only | One layer, input 64, hidden 32 **per direction**; concatenate outputs to 64 |
| Classifier, A and B | Dropout 0.1, linear 64→4; four unnormalized per-timestep logits |

Each recurrent direction receives the complete 64-feature embedding; it is not split into two input halves. No extra recurrent layers, internal recurrent dropout or hidden-state carry between tracks. No pooling or final-state-only readout. Encoder/head structures match across A/B, with independent weights. Future Transformers/token layouts are not fixed by E001.

On 2026-10-08 the user authorized this implementation after reconsidering depth, widths and regularization. The linear head replaces the former 64→32→4 head; both encoder nonlinearities remain. One recurrent layer introduces full-track learned context without assuming stacked temporal processing is needed. Width 64 / hidden 32 and dropout 0.1 are practical starting choices, not experimentally validated optima. Dropout applies to hidden activations between encoder layers and before the readout, never directly to input features/flags or timestep/GT masks. It is disabled during evaluation. B adds capacity as well as recurrence, so an A/B gain cannot isolate temporal access from parameter count.

| Feature set | Input dimension | A parameters | B parameters |
|---|---:|---:|---:|
| K | 3 | 4,676 | 29,764 |
| K+T | 6 | 4,868 | 29,956 |
| K+T+R | 12 | 5,252 | 30,340 |
| RAW | 12 | 5,252 | 30,340 |

### Model implementation and acceptance (2026-10-08)

[Shared model](../../src/pedestrian_behavior/models/kinematic.py) implements framewise and one-layer bidirectional processing; [E001's `build_model(configuration, variant)`](../../src/pedestrian_behavior/experiments/e001.py) owns feature dimensions, classes and model settings. Inputs are `[B,T,D]` float32 tensors and CPU int64 `lengths[B]`; outputs are `[B,T,4]` logits on the input/model device. Padding is masked before the encoder and excluded from recurrence with unsorted length-based packing. Padding logits are zero and unscored; internal missing slots receive predictions and remain recurrent updates. Every call starts with zero recurrent state. No GT or native sensor files enter the model.

[Model acceptance tests](../../tests/test_e001_models.py) verify hand-set two-ReLU outputs and an independently calculated bidirectional cell recurrence (`i=f=o=1/2`), including a singleton, unsorted lengths and a decaying internal zero-input slot. They compare per-track logits alone/in mixed batches/with extra NaN padding at tolerance 1e−6, check no state carry, finite gradients, zero padding gradients and an optimizer update. Counts/output dimensions are checked for all eight feature/model combinations. A CUDA-specific check verifies all eight combinations on the GPU when available. These are numerical/differentiability acceptance checks, not comparison training or evidence of predictive performance. Equal-track loss/F1 and checkpoint/training acceptance are recorded in the training/evaluation section below.

The [existing report command](../../README.md#e001-models) now includes data and model expected/actual checks; local evidence lives under ignored `outputs/experiments/E001/models/`. Native archives, eligible populations, splits, normalization and target mapping are unchanged.

Validation at implementation revision `a76b19959c0d22bb70494e584799f0ce32448e47`: the full locked-uv suite completed **29 tests** on each laptop, with **28 passed / one CUDA-only skip locally**, and **29 passed on `aalto`**, including actual GPU forwards/backwards and padding checks for all eight feature/model combinations. Expected/actual reports: `models/acceptance-local/index.html` (**93 checks**) and `models/acceptance-aalto/index.html` (**101 checks**, copied locally). `models/real-batches-local.json` records **16** finite CPU forward checks on seeded 64-track training batches: two real collections × four feature sets × A/B, zero padding logits and first-track-alone agreement within 1e−6. Documentation links/anchors and `git diff --check` passed. No comparison training, predictive-performance measurement or throughput benchmark was run.

## Batch construction

**Random batches with dynamic padding**, batch size **64 tracks**, for A and B. Shuffle eligible whole tracks each epoch with training seed 0, retain the incomplete final batch, and pad only to that batch's longest sequence. No length bucketing, cropping, fixed context limit or concatenation of independent tracks. Internal missing-input slots remain sequence positions.

Exclude padding from recurrent processing as well as loss: loss masking alone would let backward recurrence consume padded timesteps. Length-based recurrent packing is compatible with this rule; packing independent tracks into one context is not. Prediction is argmax over four states at every non-padding slot, including missing-input slots; missing-GT slots are unscored. Measure padding/exposure before reconsidering batching or a later model.

## Loss and training

For each track, average cross-entropy over its accepted GT frames; average those track losses over the actual minibatch. Thus each track has equal total loss weight regardless of duration/GT count. Apply no class weights, oversampling or label smoothing. Validation CE likewise averages per-track losses over the whole validation set, not equally over unequal minibatches.

| Setting | Value |
|---|---|
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001, betas (0.9, 0.999), epsilon 1e-8 |
| Schedule / precision | Constant learning rate; float32 |
| Gradient clipping | Global gradient norm 1.0 |
| Budget | At most 30 epochs; stop after 5 consecutive epochs without an improvement in the primary source-validation metric |
| Checkpoint selection | Highest source-validation track-weighted macro-F1; ties use lower track-averaged validation CE, then earlier epoch |
| Training seed | **0 only**, shared across the 16 planned training runs |

Patience resets only for a strict primary-metric improvement; tie-breaking can change the selected checkpoint without resetting patience. Seed Python, NumPy, PyTorch and the existing loader generator with 0 before each run. Keep the same shuffled loader across epochs, with zero workers. Enable standard deterministic algorithms (unsupported operations fail), `CUBLAS_WORKSPACE_CONFIG=:4096:8` and cuDNN determinism; disable cuDNN benchmarking and both matmul/cuDNN TF32. Record versions/settings; identical outcomes across platforms are not promised. A run still improving at epoch 30 is budget-limited; do not silently extend it. Retain failed attempts, selected epoch and stopping reason. No seed sweep, repeated runs, bootstrap or confidence intervals are planned; training variability is not estimated.

## Splits and access

Apply the same **70/15/15 group split** separately to both datasets. One group is one ROAD-Waymo clip or LOKI scenario containing eligible tracks; keep all eligible tracks from that group together. On 2026-10-08 the user accepted **each clip/scenario as independent** for E001. This is an explicit study assumption, not verified physical independence; it does not require another audit or approval before training. Existing group assignments remain unchanged.

Sort group IDs, shuffle with a fresh Python `random.Random(0)` independently for each dataset, assign the first block to validation and the next to test: allocate `ceil(0.15 × N_groups)` to validation and the same number to test, and the remainder to training. Record the shuffle implementation/version and resulting manifests. Preserve ROAD and Waymo native split names as provenance rather than using them as interchangeable experiment splits. Freeze identical manifests for all feature/model combinations.

Before training, report eligible-track/class support in all splits under the accepted clip-independence assumption. Report inadequate support rather than searching for a seed using validation/test scores. Normalization, early stopping and selection use only the source train/validation pools. Both corpora have been examined for research definition; the [strict zero-shot rules](../README.md#strict-zero-shot-access) describe training/selection access, not researcher unfamiliarity. Target-informed later changes are separate comparisons.

## Evaluation

### Primary and secondary metrics

Primary: **track-weighted macro-F1**, with each track contributing total confusion-matrix weight 1 across its accepted GT frames. For track `i` with `n_i` accepted frames:

```text
C[a,b] = sum_i (1 / n_i) * sum_t_in_GT_i 1[y_it = a and prediction_it = b]
macro-F1 = mean of the four class F1 scores derived from C
```

The primary metric always averages over the four declared classes; zero F1 denominators contribute 0, with absent support reported. This is F1 from the aggregated weighted confusion matrix, not mean per-track F1. Use the same definition for source-validation selection and final evaluations.

Also report weighted and raw confusion matrices, per-class precision/recall/F1, pooled-frame macro-F1, support-weighted F1 and accuracy, and mean/median per-track accuracy. Include clip/group/track/accepted-frame counts, class support, eligibility/GT exclusions and compute/exposure. **Point estimates only**; no bootstrap/CIs, and one-seed variability remains unmeasured. No segment/event metric is part of this comparison.

### Observation-condition strata

Report each axis separately, not every cross-product. Track strata use the full native union extent and resampled context; frame strata score accepted-GT frames only. Within a stratum, reweight each represented track by `1 / n_i,stratum`, where that count includes only its accepted GT frames in the stratum.

| Level / axis | Bins or definition |
|---|---|
| Track duration | [0,1), [1,5), [5,10), [10,15), [15,∞) seconds; LOKI nominal |
| Track usable 3D-position coverage | Valid pedestrian-position slots / all non-padding context slots: [0,0.25), [0.25,0.75), [0.75,1] |
| Track accepted GT count | 1; 2–4; 5–19; 20+ frames |
| Track annotated 2D availability | Never; partial; complete over all context slots |
| Track longest internal missing-position run | None; >0 to ≤1 s; >1 s. Count missing slots × 0.2 s between first/last valid position only |
| Frame position availability | Valid / missing pedestrian position |
| Frame velocity availability | Valid / missing pedestrian velocity |
| Frame annotated 2D availability | Valid / missing box, with native provenance |
| Frame horizontal range | `norm(ped_xy - ego_xy)`: [0,10), [10,20), [20,40), [40,∞) m; unknown without valid pedestrian/ego positions |
| Frame GT transition proximity | Within 0.4 s of the nominal-time midpoint between adjacent grid slots with accepted, different labels; steady farther away. GT gaps break the analysis; do not bridge gaps when assigning proximity |

Report supporting tracks/frames and represented GT classes in every slice. Empty slices have no score. In slice reports, class F1 without positive GT support is unavailable and macro-F1 averages only supported classes, with that class set shown explicitly; this differs from the fixed four-class primary metric. Transition analysis uses continuously labeled runs; singleton runs without a boundary have unknown proximity.

Annotated-box availability is not causal visibility or occlusion. RGB availability metadata is retained, but no RGB-quality cohort is inferred from 2D boxes. Occlusion and LiDAR sparsity remain unknown until verified; do not proxy them by range or missing boxes.

## Reader/preparation implementation and acceptance

The first increment combines native readers and complete-track 5 Hz preparation, ending with saved collections/audits and human inspection. [Shared schema](../../docs/datasets/README.md#reader-and-prepared-track-schema), [native LOKI transform evidence](../../docs/datasets/loki.md#verified-preparation-transform-2026-10-07), [runnable commands](../../README.md#prepare-and-inspect-complete-tracks) and [nine end-to-end acceptance scenarios](../../tests/test_track_preparation.py) own details. The [code architecture](../../ARCHITECTURE.md) owns downstream module responsibilities and setup/sample/batch flow.

Acceptance exercises stationary pedestrians with moving/turning ego, known motion/uneven timing, full tilted transforms/global invariance, union extents/gaps/extensions, independent missingness/initial anchors, exact selection boundaries, native semantics, identities/duplicates and persistence without source roots. Exact fixtures use declared world motion, not behavior labels as physical-motion oracles. The actual inspector values come from reloaded NumPy archives.

[Eight real native cases per dataset](reader-cases.json) were frozen before preparation: four representative action groups and four hard cases. Keep IDs/reasons and failures; conflict reasons record their then-unaccepted status, with the two overrides accepted subsequently. Audit all candidates, native versus selected counts/extents, original versus usable observations and complete native action combinations before any target projection. Artifacts: ignored `outputs/experiments/E001/reader-preparation/`. This first increment leaves targets to the downstream E001 policy; setup/batching and models are now implemented separately.

## Saved tracks to batches (2026-10-08)

[Executable setup/target policy](../../src/pedestrian_behavior/experiments/e001.py) audits complete native supervision against the saved collection sources before filtering, verifies override identities/actions/bounds/counts, derives temporary targets and records every exclusion. Shared [features](../../src/pedestrian_behavior/data/features.py), [group splits](../../src/pedestrian_behavior/data/splits.py) and [loading/collation](../../src/pedestrian_behavior/data/loading.py) implement the agreed data flow. Source statistics include valid unlabeled training context; no validation/test values enter fitting. Runtime needs only saved archives and `setup.json`, with source statistics supplied explicitly for transfer.

[Seven acceptance scenarios](../../tests/test_e001_data.py) check target collapse and both override targets, source/scope/conflict failures, distinct position/GT slots, eligible singletons/3D-only tracks, all four feature sets with a delayed reference and independent masks, exact group assignments, guarded statistics, held-out-value independence, CPU dtypes/references and length-3/length-1 padding with a real internal gap. Random loaders preserve the final partial batch and use a seeded generator across epochs. The [report command](../../README.md#e001-saved-tracks-to-batches) writes hand-written expected values beside actual outputs. Equal-track loss/F1 and padding-invariant model predictions are recorded in the model and training/evaluation sections below.

Setup writes collection/policy references, native audit evidence, split assignments, per-track/class support and frozen normalization to ignored `outputs/experiments/E001/data-setup/<dataset>/setup.json`. It references existing archives; it creates no second processed trajectory dataset. Incomplete or failed setups retain status/failures and cannot be loaded. The real-data counts below are setup evidence; ROAD-Waymo association acceptance is recorded separately below.

### Real-data setup evidence (2026-10-08)

Both setups completed at implementation revision `9698bd1c3da980296baa7fc6a8da02fc03ab855d`, with the checkout's locked uv environment: local CPU LOKI and CUDA-build `aalto` ROAD-Waymo. Native audits reconciled **391,569** LOKI pedestrian label observations (zero duplicate annotations) and **712,630** unique ROAD observations (**10** identical repeated annotations). ROAD source checksums matched the accepted policy; all **49 + 10** native overrides passed identity/action/frame/timestamp/count verification, with **24 MOVING and 5 STOPPED** observations selected on the saved grid. Native-only extensions still have no GT.

- **LOKI:** 13,365 saved candidates → **12,364 eligible**, including **4,139** with no selected native 2D observations. All **1,001 exclusions** lack both usable position and accepted GT. **616** scenarios contain eligible tracks; the other 28 of 644 do not enter the split.
- **ROAD-Waymo:** 9,573 saved candidates → **6,608 eligible**, **2,965 excluded**. Reasons overlap: **2,944** lack usable position and **35** lack accepted GT; **2,930** lack position only, **21** lack GT only and **14** lack both. **514** clips contain eligible tracks; the other 48 of 562 pedestrian clips do not enter the split. No eligible track lacks selected native 2D observations.
- Both datasets have valid initial anchors for every saved candidate. Every split supports all four classes; no alternative seed was searched. All four feature configurations for both sources have **no guarded columns**; means/scales/counts/stds are saved in their setup artifacts.

Class columns below use **MOVING, STOPPED, WAITING_TO_CROSS, CROSSING** order. Track class support overlaps when a track has multiple states; slots include unlabeled context.

| Dataset | Split | Groups | Eligible tracks | Slots | GT frames | Class frames | Tracks per class |
|---|---|---:|---:|---:|---:|---|---|
| loki | training | 430 | 8,574 | 270,355 | 262,071 | 160,612 / 21,035 / 35,322 / 45,102 | 6,566 / 860 / 1,174 / 1,338 |
| loki | validation | 93 | 1,861 | 61,900 | 60,231 | 37,410 / 4,206 / 8,320 / 10,295 | 1,453 / 181 / 226 / 306 |
| loki | test | 93 | 1,929 | 71,558 | 69,267 | 44,718 / 5,768 / 8,953 / 9,828 | 1,509 / 229 / 234 / 323 |
| road-waymo | training | 358 | 4,380 | 332,940 | 167,280 | 86,095 / 39,059 / 11,336 / 30,790 | 2,576 / 1,209 / 318 / 1,038 |
| road-waymo | validation | 78 | 1,110 | 87,824 | 45,388 | 24,018 / 11,232 / 3,317 / 6,821 | 658 / 306 / 91 / 262 |
| road-waymo | test | 78 | 1,118 | 86,380 | 41,488 | 21,239 / 10,631 / 3,523 / 6,095 | 652 / 278 / 88 / 224 |

Evidence: local `outputs/experiments/E001/data-setup/{loki,road-waymo}/setup.json`; ROAD's original artifact is `/home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001/data-setup/road-waymo/setup.json` on `aalto` and its copy was checked against the local saved collection. Manifests retain collection/policy hashes, native action sets, per-track native annotation counts, every candidate/exclusion/split, environment/code hashes, command and times. Local `loki/source-diff.patch` preserves the pre-existing documentation-only changes recorded by its diff checksum; the remote setup checkout was clean. Setup took about **60 s locally / 32 s remotely**, without training or sensor-image/LiDAR decoding.

Validation: **22 unittests passed on both laptops**, affected local links/anchors and `git diff --check` passed. The seven-scenario expected/actual report is `outputs/experiments/E001/data-setup/acceptance/index.html`. `runtime-checks.json` and its retained `runtime-checks.py` also verify the first seeded 64-track batch for each of **four feature sets × two source statistics × two dataset populations**, plus both real override tracks. These smoke checks confirm finite CPU tensors, dtypes, masks, references and pad values with within-dataset and unchanged cross-source normalization; they are not model predictions or an exhaustive batch-throughput study. Reproduce with `uv run --locked --extra cpu python outputs/experiments/E001/data-setup/runtime-checks.py` after both setup artifacts/collections are available locally.

## Training/evaluation implementation and acceptance (2026-10-08)

[Shared training](../../src/pedestrian_behavior/training.py) and [evaluation](../../src/pedestrian_behavior/evaluation.py) connect saved batches to fixed AdamW training, complete source-validation selection, atomic `best.pt`/`last.pt` state dictionaries and selected-checkpoint evaluation on both tests. [Run assembly](../../src/pedestrian_behavior/experiments/e001_run.py) owns E001 settings and [commands](../../README.md#e001-training-and-evaluation). Epoch CE sums per-track losses across unequal batches and divides by tracks; it describes changing training weights, without an extra training-set evaluation pass. Checkpoints retain epoch, validation/selection evidence, model/config references and source normalization. Test results never affect selection. Fresh attempt directories are mandatory; there is no resume or retry.

Final archives contain unpadded float32 logits for all context slots, targets, GT masks and offsets; identities, archive references, source/nominal grid times, native extents, condition memberships and individual `loki_2d`/`road_2d`/`waymo_2d` flags make metrics/plots reproducible without inference. Track duration uses native union extents; coverage and internal gaps use grid context; range uses physical saved positions. ROAD 2D means ROAD **or** Waymo. Transitions use nominal-time midpoints only within continuously labeled runs: distance ≤0.4 s is near, longer constant runs are steady, singleton runs unknown. Each slice renormalizes its represented tracks, reports support and averages only GT-supported classes; unsupported F1 and empty scores are `null`.

Artifacts live under ignored `outputs/experiments/E001/runs/<attempt>/`: config, provenance, status/failure tracebacks, history, per-update `steps.jsonl`, checkpoints, dataset-specific prediction/metric files and PNG/SVG curves, confusion views, per-class and ten-axis plots. Provenance records hashes, source statistics, versions/backend/hardware, deterministic settings, selected epoch/stopping reason, parameters, exposure/padding, runtime and peak CUDA allocated memory. W&B uses one run per attempt, one training-loss record per optimizer update, one curve record per epoch, then final metrics/tables/plots. Provenance and denominator counts are stored in configuration; local history retains exposure/selection details. Numerical evidence is independently local; logging failures preserve available local files and propagate. No model or full prediction artifact is uploaded. External backup storage is **TBD**; back up valuable runs separately.

[Acceptance](../../tests/test_e001_training.py) extends the existing expected/actual report with equal-track versus pooled metrics (1/6 versus 1/22 macro-F1; 1/2 versus 1/10 accuracy; `(ln 2 + ln 4)/2` CE), masks/empty-GT/nonfinite failures, partial-batch aggregation, selection/ties/patience/budget, checkpoint reload, every condition boundary, GT-gap/singleton transition handling, supported-class/empty slices, metric reconstruction and W&B content/failure preservation. Synthetic archive fixtures exercise complete attempt wiring without requiring real datasets. CUDA checks cover deterministic masked/clipped AdamW and checkpoint equality for A/B.

Verification at implementation revision `37d6d5ce122fcc17c77cb8188391a768ffa7852f`: the 37-test suite passed locally (**35 passes / two CUDA-only skips**) and on `aalto` (**37 passes**). Expected/actual reports passed: CPU **21 scenarios / two skips / 165 checks**, CUDA **21 scenarios / no skips / 178 checks**. CPU/CUDA dependency checks passed; affected documentation links/anchors and `git diff --check` passed. The first acceptance report failed while serializing a NumPy expected value; its local evidence remains in `training-evaluation/acceptance-local/`. Corrected reports: `training-evaluation/{acceptance-cpu,acceptance-cuda}/index.html`, both available locally. The first CUDA suite failed because earlier GPU tests initialized cuBLAS before the workspace environment was set. The CLI/test setup now sets it before GPU work; the corrected suite passed without weakening determinism. No comparison results.

Separate **GPU smoke only**, attempt `outputs/experiments/E001/runs/road-k-b-gpu-smoke/` on both laptops, generated at the verified clean revision above. ROAD-Waymo K/B on RTX 4080 completed **one epoch / two optimizer steps / 128 training tracks**, processing **9,387 context / 5,191 GT slots**, **3,413 padding / 12,800 batch slots (26.66%)**. Only one 64-track source-validation batch was selected/scored: **8 groups / 4,575 context / 2,133 GT frames**, class frames **1,269 / 422 / 183 / 259**. No held-out split was evaluated and no comparison result is claimed. End-to-end attempt time was **17.1 s**, including setup/plots/W&B; peak CUDA allocated memory **110,455,808 bytes (105.3 MiB)**. This is not a throughput estimate for full runs.

[W&B smoke run](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ci7s33jl) finished successfully with exactly one epoch record, final metrics/tables and **17 PNG plots**; matching **17 SVG exports** remain local. Independent `verification.json` checks both checkpoints, exact reloaded validation logits (maximum difference **0**), complete primary/ten-axis metric reconstruction from saved predictions and absence of held-out evaluation directories. Cloud file/artifact inspection found only agreed content: the SDK stores the **34 metric tables** as `run_table` artifacts, with no checkpoint or full prediction upload. Representative condition/confusion/per-class PNGs were visually inspected after copying the complete smoke evidence locally. Smoke scores remain diagnostic setup evidence, not the 16-run comparison.

LOKI transfer: **13,369 files** from the existing `reader-preparation/loki/` and `data-setup/loki/` match local SHA-256 on `aalto`. Manifest: local `outputs/experiments/E001/training-evaluation/loki-transfer-sha256.txt`, remote `outputs/experiments/E001/data-setup/loki-transfer-sha256.txt`. Original setup provenance and collection bytes are unchanged; native data were not transferred/reprocessed. The already-confirmed native path needs no further verification or approval.

## ROAD association acceptance (2026-10-08)

**Accepted for E001 with qualifications**, following the user's instruction to close this gate. The [versioned record](road-association-review.json) freezes the rule, source/collection/review hashes, counts, selected identities/timestamps and Codex's per-case observations. This is an acceptance of the acquired official-link population for this kinematic diagnostic, not a measured population-wide association accuracy or confirmation of every tiny/occluded actor.

Rule declared before visual review: trust scene-scoped official camera-to-LiDAR IDs only after source, uniqueness, exact timestamp/pair and native-type checks pass; retain missing observations; reject demonstrated identity mismatch/ambiguity. Review all non-Pedestrian native-type tracks plus frozen controls and deterministic small-box, crowded and large-velocity cases. Inconclusive visual cases are disclosed and retained on the verified official link; no speed threshold, inferred association or new GT is introduced.

[Executable audit](../../scripts/audit-road-associations.py) ran on `aalto` in the locked CUDA uv environment at verified Git revision `055df5f08e84c4695a78b29e637810e0bc4f1941`. It rechecked all **9,573 candidates / 562 clips**, matched every saved source/component checksum and native inventory/extent/official ID, and verified all **426,491** unique exported pairs against the native linked ID/type and all ROAD timestamps against FRONT images. No ambiguous/shared scene-scoped LiDAR ID, missing exported native pair, conflicting repeat or source mismatch was encountered. Counts reproduce the [native context audit](../../docs/datasets/road-waymo.md#native-identity-and-context-audit-2026-10-06): **6,809** unique official links; **6,634** tracks with native 3D; **495,830** additional 3D and **8,913** additional FRONT timestamps. Those extensions remain unlabeled context.

The purposive review covers **33 tracks / 236 same-frame snapshots**: eight frozen native cases, all **17** native Cyclist-type tracks (15 exported-disagreement tracks plus two with Cyclist geometry only in native extensions), and the top three distinct clips each by smallest median ROAD box, most same-frame ROAD pedestrians and greatest saved velocity. Snapshot selection includes original annotation/position boundaries, internal samples, native extensions and velocity-peak neighbors. Native linked 3D contains **921,036 Pedestrian / 1,285 Cyclist observations**; no other type was found. The **419** exported disagreement observations remain unchanged.

Codex reviewed native RGB/BEV and original-resolution crops. Several cases show people with bicycles; the largest-speed case shows a scooter rider despite native Pedestrian type, demonstrating that this population includes wheeled human motion. Peak neighboring positions are smooth in the three speed cases. No wrong actor link or abrupt identity switch was demonstrated in the selected views. Dark, tiny, edge-truncated and vehicle/railing-occluded cases cannot all be independently confirmed; these limitations appear per case in the record. Missing 3D does not prove occlusion. Neither a visual error rate nor exact physical sensor accuracy is established.

Evidence: ignored `outputs/experiments/E001/association-review/complete/{audit.json,index.html,SHA256SUMS,case-*}`, generated in `/home/user20/projects/pedestrian-behavior-labeling/` on `aalto` and copied locally. All **376** original report/image files match the frozen checksum manifest on both laptops. [Reproduction command](../../README.md#prepare-and-inspect-complete-tracks); [audit fixture](../../tests/test_road_association_audit.py) covers native-only Cyclist context, missing exported pairs, retained timestamps and source rejection. **23 unittests passed locally and remotely**; local documentation links/anchors and `git diff --check` passed. Source annotations, archives, targets/overrides, eligibility, split assignments and frozen normalization are unchanged; the previously reported **6,608 ROAD / 12,364 LOKI** eligible tracks remain the comparison populations.

Unknown upstream acquisition/release revisions remain disclosed in native notes; the frozen acquired CSV/manifest/native component hashes and versioned audit support reproducing this population without claiming a rebuild of the original acquisition. Independent RGB projection and sensor calibration accuracy remain broader checks. Clips/scenarios are independent by the accepted user assumption; no further approval/audit of that assumption is a prerequisite.

## Verification and unresolved readiness

Protocol choices and data/association/model acceptance above are settled for E001. Training/evaluation and separate GPU smoke verification are complete; comparison execution remains next; broader unknowns below are retained as limitations. Other datasets are not prerequisites.

| Gate / setting | Required evidence or remaining value |
|---|---|
| Releases / acquisition / associations | Frozen acquired CSV/manifest/native component hashes and versioned full audit recorded; original upstream acquisition/release revisions remain **unknown**, disclosed limitations for E001 |
| Native identity/context acceptance | [Accepted with qualifications](#road-association-acceptance-2026-10-08); full audit and purposive 33-track review recorded; all 59 known movement/stop conflicts have accepted overrides |
| Geometry and time | Full LOKI release transform/marker check and official Waymo pose interpretation support preparation; independent sensor accuracy/RGB projection remain open; physical LOKI timestamps **unknown** |
| Final population / manifests | Setup implements the accepted projection/eligibility, reports exclusions and split/class support, and freezes clip groups; clip/scenario independence is accepted as an assumption (2026-10-08), not a remaining gate |
| Normalization guard | Numerical near-zero scale threshold **1e−8**, inclusive; source-training statistics and guarded columns recorded before runs |
| Implementation / environment | Models, training/evaluation/checkpoints and commands implemented; CPU/CUDA acceptance and separate W&B GPU smoke verified; locked CPU/CUDA environments verified |
| Low-shot / later ablations | None specified in E001; extensions require explicit labels, access and justification |

At implementation, check: a stationary pedestrian stays stationary under ego translation/turning; canonical features are invariant to global translation/yaw rotation; derivative endpoints/unequal times/gaps behave as declared; duplicates and input/GT masks stay distinct; normalization uses source training only; padding does not change valid predictions; loss and primary F1 give equal total track weight. Preserve unknowns and failures rather than manufacturing geometry or GT. Run the [required validation](../../AGENTS.md#validation) and record actual acceptance evidence before training.

## Variants and runs

Each row below represents four independent runs: A/B × ROAD-Waymo/LOKI source, seed 0, each evaluated on both held-out tests. Add individual rows for actual runs or failed attempts; keep negative transfer.

| Feature set | Models | Training sources, separately | Status | Failure / result |
|---|---|---|---|---|
| K | A; B | ROAD-Waymo; LOKI | Planned | Not run |
| K+T | A; B | ROAD-Waymo; LOKI | Planned | Not run |
| K+T+R | A; B | ROAD-Waymo; LOKI | Planned | Not run |
| RAW | A; B | ROAD-Waymo; LOKI | Planned | Not run |

## Run provenance

- W&B destination: `entity="yammo-unipd"`, `project="pedestrian-behaviour-labeling"` ([project](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling)). On 2026-10-08 the user reported successful login on both laptops. W&B 0.30.0 is installed in both uv environments; Training logging is implemented; actual upload evidence is recorded in the training/evaluation acceptance section. Generated run artifacts remain under ignored `outputs/experiments/E001/` or documented external storage; Checkpoints and full predictions remain local; only config/provenance, metrics, tables and plots are uploaded.
- Fixed executable settings live in `src/pedestrian_behavior/experiments/e001_run.py`; each attempt saves its effective config, command, revision/diff, versions and environment. No comparison attempt has run; see [separate smoke evidence](#trainingevaluation-implementation-and-acceptance-2026-10-08).
- Intended compute: RTX 4080 laptop on SSH host `aalto`; [uv-managed Python 3.11.16 and locked dependencies](../../README.md#setup-and-validation), with PyTorch 2.7.1 CPU locally/CUDA 11.8 remotely. The 2026-10-06 check found no PyTorch remotely; initial Conda setup on 2026-10-07 was superseded by uv that day. Both uv environments passed the required six tests and packed BiLSTM forward/backward/AdamW smoke checks. The remote package resolves to `/home/user20/projects/pedestrian-behavior-labeling`; no comparison training or throughput benchmark has run; see [separate restricted smoke evidence](#trainingevaluation-implementation-and-acceptance-2026-10-08).
- Reader/preparation collections, full audits, independent exact-check fixtures and sixteen-track inspection pack: ignored `outputs/experiments/E001/reader-preparation/`, generated 2026-10-07. Local LOKI: 13,365 candidates; remote ROAD-Waymo: 9,573, copied locally for reload. Collection manifests record effective rules, source/code hashes, revision/diff and Python/NumPy versions. Comparison training has not run; the [separate smoke artifacts](#trainingevaluation-implementation-and-acceptance-2026-10-08) retain logs/checkpoints/predictions/metrics under ignored `outputs/experiments/E001/`. External backup storage remains **TBD**.
- Record source/release/checksum and split IDs, all seeds/RNG settings, selected checkpoint/validation evidence, host/hardware without credentials, start/end dates, exposure/runtime/parameters, denominators, failures and deviations using the [shared recording rules](../README.md#comparison-records-and-artifacts).

## W&B workspace and logging

New run names are `E001-<source>-<configuration>-<MLP|BiLSTM>-seed0-<UTC YYYYMMDD-HHMMSS>`, for example `E001-loki-K-MLP-seed0-20261008-170000`. Restricted smoke names add `-smoke`; naming is independent of the output directory.

Use the [manual E001 view](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling?nw=71j6lc3jv4g): three main native curve panels, followed by collapsed final-evaluation and ten-axis observation-strata sections for each dataset. Automatic panel generation is disabled in that saved view. It is separate from the existing personal automatic view; prior runs and their evidence are preserved. The official workspace client was used through a temporary `wandb-workspaces==0.4.13` uv overlay, without a project dependency. Layout/source/readback evidence: ignored `outputs/experiments/E001/wandb-workspace/`.

| W&B metric | Frequency | X-axis |
|---|---|---|
| `train/loss_step` | Each optimizer update; actual minibatch's equal-track CE, computed before that update | `optimizer_step`, cumulative from 1 across epochs |
| `train/loss_epoch` | Completed epoch; existing aggregation over tracks | `epoch` |
| `val/loss` | Full source validation after each epoch; track-averaged CE | `epoch` |
| `val/macro_f1` | Same source-validation pass; fixed four-class track-weighted macro-F1 | `epoch` |

Custom axes are declared through [W&B's `Run.define_metric`](https://docs.wandb.ai/ref/python/experiments/run/#method-rundefine_metric). W&B's internal row counter increases on logging calls; the command does not set it to the epoch number. Counter axes and `details/*` are hidden from automatic plots. Final scalar metrics use `eval/<dataset>/...`, ten supported-class/count plots use `strata/<dataset>/<axis>`, and complete per-class/confusion/slice tables and supplementary plots use `details/...`. Denominators and final provenance update configuration, avoiding metadata-count charts. No metric, class support, failure or local export is discarded. Original smoke keys (`training_ce`, `validation_ce`, etc.) remain historical evidence; subsequent attempts use the names above.

Step records are appended to `steps.jsonl` before external logging; epoch history, checkpoint selection and validation frequency are unchanged. A logging error still stops the attempt and retains local evidence. CPU suite: **38 tests / 36 passes / two CUDA skips**; CUDA suite: **38 passes**. They check global step offsets, actual-batch CE, unequal-batch epoch aggregation, declared axes/namespaces and step-logging failure preservation. The 22-scenario expected/actual reports are `outputs/experiments/E001/training-evaluation/logging-acceptance-{cpu,cuda}/index.html` (two CPU CUDA-only skips).

Fresh restricted [logging smoke](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ixe6a2nc), attempt `outputs/experiments/E001/runs/road-k-b-logging-smoke/`, ran at verified clean revision `45bba13d528a5782e16f86d9f00ec525ee53318b`. W&B readback confirmed **two optimizer-step loss records / one epoch record**, nine final scalar metrics, all ten stratum plots, and provenance/support in configuration rather than history. Saved predictions reconstruct every primary/slice metric; both checkpoint model states exactly match the original deterministic smoke. All **17 PNG / 17 SVG** exports and **34 metric tables** are retained; cloud inspection found no checkpoint or full prediction upload. The attempt took **16.3 s**, with **110,455,808 bytes** peak CUDA allocation. Independent checks are in its `verification.json`, retained locally; no held-out tests or comparison runs occurred.

The saved workspace was read back through the official API: main curve axes/keys, manual panel generation, section counts and collapsed final sections match the intended layout. Browser rendering was not checked because no browser surface was available.

## Conclusions, limitations and next step

No measured conclusion. Poor within-dataset performance can reflect limited evidence, annotation ambiguity or data/model/protocol failure, without proving sensor insufficiency. A B gain supports learned temporal context under these features; strong within-dataset results with poor transfer suggest mismatch. Geography, hardware, scene structure, semantics, class frequencies and annotation selection all change, so directional scores cannot isolate selection policy or establish causality/novelty. Study difficult conditions and Stopped/Waiting errors even after strong aggregate scores.

Examine whether added trajectory and interaction information helps Stopped/Waiting and Moving/Crossing, and whether gains survive transfer. RAW is a diagnostic control, not an automatic final representation: structured trajectory variants supply explicit track-start displacement that framewise RAW cannot directly reconstruct. Positive results do not establish safe use by a larger Transformer; negative results do not prove a feature inherently useless. Use observed errors to guide later temporal/multimodal comparisons, without permanently selecting features from this baseline alone.

Next: arrange a separate backup, then run the 16 comparisons in a later increment. The frozen LOKI collection/setup are now available on `aalto` with checksum verification. Model padding invariance is verified; ROAD association acceptance and clip/scenario independence need no further approval. Inspect failures before adding modalities, sources or architecture. All broader methods and ontology/head choices remain provisional.
