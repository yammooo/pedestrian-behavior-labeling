# E001 — Kinematic transfer

Status: **Planned**. Created 2026-10-05 from the 2026-10-02 proposal. Protocol agreed 2026-10-06; input comparison revised 2026-10-08. Native-data verification remains pending. No model implementation, training or results.

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
| Classifier, A and B | Linear 64→32, ReLU, dropout 0.1, linear 32→4 |

Each recurrent direction receives the complete 64-feature embedding; it is not split into two input halves. No extra recurrent layers, internal recurrent dropout or hidden-state carry between tracks. Record actual parameter counts at implementation. Future Transformers/token layouts are not fixed by E001.

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

Patience resets only for a strict primary-metric improvement; tie-breaking can change the selected checkpoint without resetting patience. Record exact RNG/determinism settings and versions when implemented. A run still improving at epoch 30 is budget-limited; do not silently extend it. Retain failed attempts, selected epoch and stopping reason. No seed sweep, repeated runs, bootstrap or confidence intervals are planned; training variability is not estimated.

## Splits and access

Apply the same **70/15/15 group split** separately to both datasets. One group is one ROAD-Waymo clip or LOKI scenario containing eligible tracks; keep all eligible tracks from that group together. Recording/identity overlap remains an evidence gate before training: IDs alone do not establish physical independence, and verified overlap would require a documented protocol/manifests revision before any run.

Sort group IDs, shuffle with a fresh Python `random.Random(0)` independently for each dataset, assign the first block to validation and the next to test: allocate `ceil(0.15 × N_groups)` to validation and the same number to test, and the remainder to training. Record the shuffle implementation/version and resulting manifests. Preserve ROAD and Waymo native split names as provenance rather than using them as interchangeable experiment splits. Freeze identical manifests for all feature/model combinations.

Before training, audit recording/identity overlap and eligible-track/class support in all splits. Report inadequate support rather than searching for a seed using validation/test scores. Normalization, early stopping and selection use only the source train/validation pools. Both corpora have been examined for research definition; the [strict zero-shot rules](../README.md#strict-zero-shot-access) describe training/selection access, not researcher unfamiliarity. Target-informed later changes are separate comparisons.

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

[Eight real native cases per dataset](reader-cases.json) were frozen before preparation: four representative action groups and four hard cases. Keep IDs/reasons and failures; conflict reasons record their then-unaccepted status, with the two overrides accepted subsequently. Audit all candidates, native versus selected counts/extents, original versus usable observations and complete native action combinations before any target projection. Artifacts: ignored `outputs/experiments/E001/reader-preparation/`. This first increment leaves targets to the downstream E001 policy; the data increment below now implements setup and batching. Models remain unimplemented.

## Saved tracks to batches (2026-10-08)

[Executable setup/target policy](../../src/pedestrian_behavior/experiments/e001.py) audits complete native supervision against the saved collection sources before filtering, verifies override identities/actions/bounds/counts, derives temporary targets and records every exclusion. Shared [features](../../src/pedestrian_behavior/data/features.py), [group splits](../../src/pedestrian_behavior/data/splits.py) and [loading/collation](../../src/pedestrian_behavior/data/loading.py) implement the agreed data flow. Source statistics include valid unlabeled training context; no validation/test values enter fitting. Runtime needs only saved archives and `setup.json`, with source statistics supplied explicitly for transfer.

[Seven acceptance scenarios](../../tests/test_e001_data.py) check target collapse and both override targets, source/scope/conflict failures, distinct position/GT slots, eligible singletons/3D-only tracks, all four feature sets with a delayed reference and independent masks, exact group assignments, guarded statistics, held-out-value independence, CPU dtypes/references and length-3/length-1 padding with a real internal gap. Random loaders preserve the final partial batch and use a seeded generator across epochs. The [report command](../../README.md#e001-saved-tracks-to-batches) writes hand-written expected values beside actual outputs. Equal-track loss/F1 and padding-invariant model predictions belong to the following increment.

Setup writes collection/policy references, native audit evidence, split assignments, per-track/class support and frozen normalization to ignored `outputs/experiments/E001/data-setup/<dataset>/setup.json`. It references existing archives; it creates no second processed trajectory dataset. Incomplete or failed setups retain status/failures and cannot be loaded. Real-data population evidence: **TBD until the verified-revision setup runs finish**. ROAD-Waymo association acceptance still gates training.

## Verification and unresolved readiness

Protocol choices above are settled for E001; evidence below still gates execution. Other datasets are not prerequisites.

| Gate / setting | Required evidence or remaining value |
|---|---|
| Releases / acquisition / associations | Native notes preserve acquired paths/counts; complete release revisions/checksums and versioned reproducible mapping/extension audit **TBD** |
| Native identity/context acceptance | Reproduce [ROAD native-ID extension counts](../../docs/datasets/road-waymo.md#native-identity-and-context-audit-2026-10-06), inspect continuity/semantic disagreements/varied associations and declare visual association acceptance **TBD**; all 59 known movement/stop conflicts have accepted overrides |
| Geometry and time | Full LOKI release transform/marker check and official Waymo pose interpretation support preparation; independent sensor accuracy/RGB projection remain open; physical LOKI timestamps **unknown** |
| Final population / manifests | Setup implements the accepted projection/eligibility, reports exclusions and split/class support, and freezes clip groups; physical recording/identity independence remains **TBD** |
| Normalization guard | Numerical near-zero scale threshold **1e−8**, inclusive; source-training statistics and guarded columns recorded before runs |
| Implementation / environment | Model code/config/commands **TBD**; [local CPU/remote CUDA uv setup](../../README.md#setup-and-validation) verified 2026-10-07, with no model implementation or training |
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

- Comparison-specific effective configs: future files beside this record; none yet. Commands, code revision/uncommitted diff, dependency versions and environment for actual training runs: **TBD**.
- Intended compute: RTX 4080 laptop on SSH host `aalto`; [uv-managed Python 3.11.16 and locked dependencies](../../README.md#setup-and-validation), with PyTorch 2.7.1 CPU locally/CUDA 11.8 remotely. The 2026-10-06 check found no PyTorch remotely; initial Conda setup on 2026-10-07 was superseded by uv that day. Both uv environments passed the required six tests and packed BiLSTM forward/backward/AdamW smoke checks. The remote package resolves to `/home/user20/projects/pedestrian-behavior-labeling`; no E001 training or throughput benchmark has run.
- Reader/preparation collections, full audits, independent exact-check fixtures and sixteen-track inspection pack: ignored `outputs/experiments/E001/reader-preparation/`, generated 2026-10-07. Local LOKI: 13,365 candidates; remote ROAD-Waymo: 9,573, copied locally for reload. Collection manifests record effective rules, source/code hashes, revision/diff and Python/NumPy versions. No model training artifacts yet; future normalization/logs/checkpoints/predictions/metrics stay under ignored `outputs/experiments/E001/` or documented external storage.
- Record source/release/checksum and split IDs, all seeds/RNG settings, selected checkpoint/validation evidence, host/hardware without credentials, start/end dates, exposure/runtime/parameters, denominators, failures and deviations using the [shared recording rules](../README.md#comparison-records-and-artifacts).

## Conclusions, limitations and next step

No measured conclusion. Poor within-dataset performance can reflect limited evidence, annotation ambiguity or data/model/protocol failure, without proving sensor insufficiency. A B gain supports learned temporal context under these features; strong within-dataset results with poor transfer suggest mismatch. Geography, hardware, scene structure, semantics, class frequencies and annotation selection all change, so directional scores cannot isolate selection policy or establish causality/novelty. Study difficult conditions and Stopped/Waiting errors even after strong aggregate scores.

Examine whether added trajectory and interaction information helps Stopped/Waiting and Moving/Crossing, and whether gains survive transfer. RAW is a diagnostic control, not an automatic final representation: structured trajectory variants supply explicit track-start displacement that framewise RAW cannot directly reconstruct. Positive results do not establish safe use by a larger Transformer; negative results do not prove a feature inherently useless. Use observed errors to guide later temporal/multimodal comparisons, without permanently selecting features from this baseline alone.

Next: review the saved native and data-setup evidence, resolve ROAD-Waymo association/recording acceptance, then implement models/training with the agreed loss/F1 and padding-invariance tests. Inspect failures before adding modalities, sources or architecture. All broader methods and ontology/head choices remain provisional.
