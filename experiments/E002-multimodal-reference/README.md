# E002 — Multimodal offline reference

**Overview, updated 2026-11-03.** E002 groups the multimodal offline-labeling direction. Each concrete comparison owns its architecture, settings, preparation and results. [E002a](../E002a-feature-feasibility/README.md) owns the first behavior pilot and RGB seed repeat; [E002b](../E002b-normalization-capacity/README.md) owns normalization/capacity and the [joint 26-model review](../E002b-normalization-capacity/results-review.md); [E002c](../E002c-rgb-regularization/README.md) owns the completed RGB dropout diagnostic; [E002d](../E002d-rgb-lidar-complementarity/README.md) owns the completed fusion pilot and descriptor-dropout controls; [E002e](../E002e-weight-decay-sensitivity/README.md) owns the completed `g-rgb` weight-decay sensitivity check. Earlier extraction, timing and operational evidence remains in E002a.

## Question and interpretation

Do appearance and local 3D scene evidence improve offline pedestrian behavior labeling beyond geometry, including incomplete RGB observations and cross-dataset transfer? Establish an empirical reference before considering one new retrieval or supervision mechanism. Neither modality benefit nor architectural novelty is assumed; see the [literature positioning](../../docs/literature/offline-labeling-research-direction.md) and [research direction](../../docs/research.md#tentative-experiment-direction).

The [E001-family review](../E001b-2d-bbox-contribution/results-review.md) motivates geometry as a pragmatic reference, not a universal winner. Existing test inspection informs this development and must be disclosed.

## Comparisons and ownership

| Record | Scope | Status |
|---|---|---|
| [E002a — first multimodal behavior pilot](../E002a-feature-feasibility/README.md) | Original three arms plus a separate geometry + D-DITR follow-up; frozen encoders, complete tracks, indexed point supply; Utonia deferred | Eight seed0 source-validation runs complete; geometry + D-DITR best in this seed |
| [E002b — normalization and capacity](../E002b-normalization-capacity/README.md) | Geometry / +RGB / +LiDAR; normalized reference versus smaller temporal or branch representations | 18 best checkpoints evaluated; 15 normal stops, three LOKI LiDAR interrupted at23 epochs |
| [E002c — RGB temporal dropout](../E002c-rgb-regularization/README.md) | Matched0.1/0.3/0.5 temporal dropout across full/small normalized `g-rgb` models on both sources | Eight additional seed0 source-validation runs complete; original E002b paths retained |
| [E002d — RGB–LiDAR complementarity](../E002d-rgb-lidar-complementarity/README.md) | Sequential `g-rgb-dditr / temporal-small` with RGB branch dropout0/0.5; two follow-up `g-dditr` LiDAR-dropout controls | Four fusion runs and two LiDAR-dropout controls complete |
| [E002e — weight-decay sensitivity](../E002e-weight-decay-sensitivity/README.md) | Matched `g-rgb / temporal-small` source runs at AdamW weight decay0.01/0.1 against E002b's0.0001 controls | Four seed0 source-validation runs complete; no selected-F1 gain, despite modest late CE-gap reduction at weight decay0.1 |

E002a's architecture is specific to that pilot. Later comparisons may revise it with explicit hypotheses and controls; they do not inherit its encoders, widths or readout by default. Completed PCA, alignment, semantic and resource probes remain [preparatory evidence inside E002a](../E002a-feature-feasibility/README.md#preparatory-work), with original settings and failures preserved.

The22 October revision supplies only enabled descriptors and their validity (65/194/323→256 fusion). It tests the utility of adding a modality including availability. Earlier shared-three-flag attempts retain their original interpretation; do not pool protocols. Existing feature caches remain valid, but revised model/execution hashes require new resource calibration and the epoch-time gate remains unmet.

## Shared constraints and next decision

Retain the given-track, full-record offline contract, eligible missing-observation/3D-only pedestrians and explicit GT/input masks. Predeclare populations, splits, source-only fitting, model-selection access and budgets under the [shared evaluation safeguards](../README.md#strict-zero-shot-access). Disclose test-informed development, pretrained exposure and any target-informed choices; frozen encoders do not erase prior access.

The [joint 26-model review](../E002b-normalization-capacity/results-review.md) identifies geometry + LiDAR /`temporal-small` as the strongest development candidate on both sources (66.05% ROAD /76.41% LOKI), with lower selected CE than E002a LiDAR. Normalization helps ROAD RGB but is not a universal improvement; branch compression has no consistent benefit. RGB's smaller temporal model has different F1/CE trade-offs on the two sources. Three LOKI LiDAR trainings remain interrupted at23 epochs despite complete best-checkpoint evaluation. Loss overfitting persists; neither its cause nor final generalization is established. Shared-process runtime is not a per-model speed comparison. Original execution/calibration evidence stays in the concrete records.

The [E002c dropout result](../E002c-rgb-regularization/README.md#selected-source-validation-results) finds no consistent RGB F1 improvement from stronger temporal dropout. The only gain over a matched0.1 control is +0.74 pp on ROAD `temporal-small`; all LOKI higher-dropout variants lose F1. This remains single-seed validation development.

The [E002e weight-decay result](../E002e-weight-decay-sensitivity/README.md#results) finds essentially unchanged ROAD selected F1 and a0.58–0.77 pp LOKI decrease at weight decay0.01/0.1. Weight decay0.1 lowers the late validation CE gap modestly on both sources while selected F1 remains flat or slightly lower. E002e train F1 is far above validation F1; historical controls lack train-F1 logging. This is still one seed per setting.

## Exploratory cross-dataset validation transfer

Agreed **2026-11-02**: evaluate the frozen eight-checkpoint list in [transfer-comparison.json](transfer-comparison.json) once on the opposite source's validation split, using each checkpoint's training-source normalization. Compare with its already saved source-validation metrics; do not rerun source inference. The eight models cover geometry, geometry + RGB, geometry + LiDAR, and all three inputs for ROAD and LOKI training sources. The original within-source selected-F1 checkpoint rule is retained; CE is reported alongside F1.

This is **exploratory, target-informed validation transfer**, not held-out or strict zero-shot evidence: these same validation populations have already guided source-specific checkpoint/model selection. No test split is accessed. Existing train/validation RGB and indexed LiDAR caches are read-only; no new cache is prepared. See the compact [transfer results](transfer-results.md).

Full multi-seed held-out modality/transfer comparisons remain deferred. E002b is source-validation development; it does not authorize the previously discussed40-attempt /80-test-cell comparison. Confirmatory seed sets and final test access remain separately declared.

## RGB and LiDAR availability in the frozen E002 populations

The **2026-11-02 metadata audit** quantifies natural model-input availability before training dropout on the frozen E002 populations. The [dataset conventions](../../docs/datasets/README.md#measured-rgb-and-lidar-availability) own the metric definitions; the [ROAD-Waymo](../../docs/datasets/road-waymo.md#e002-rgb-and-lidar-availability-2026-11-02) and [LOKI](../../docs/datasets/loki.md#e002-rgb-and-lidar-availability-2026-11-02) notes own the train/validation coverage tables, track quantiles, missing/full-track counts and GT-only rates.

Validation flags exactly match prepared RGB counts, indexed LiDAR validity/token counts and frozen GT support. Validation context slots `(both / RGB-only / LiDAR-only / neither)` are ROAD **38,382 / 7,571 / 40,348 / 1,523** and LOKI **24,784 / 1,295 / 35,447 / 374**. Within validation GT, ROAD has 37,891 both-modal and 7,497 RGB-only slots; LOKI has 24,784 both-modal and 35,447 LiDAR-only slots. These are source-observation patterns, not evidence that the modalities cause the dataset-level behavior differences.

The full [per-track CSV](../../outputs/experiments/E002/availability-20261102/per_track.csv), [JSON summary](../../outputs/experiments/E002/availability-20261102/summary.json) and standalone [reproduction snippet](../../outputs/experiments/E002/availability-20261102/summarize_availability.py) are in ignored outputs. The analysis reads only existing RGB preparation/audit metadata, indexed point scene manifests, frozen setup summaries and saved E002d validation predictions. It reads no feature payloads or native sensor files and uses no GPU or cache writes.

## Training GT runs and temporal-mask feasibility (2026-11-02)

A separate audit counted only the frozen **training** tracks, measured accepted-label runs and natural input masks, then simulated input-only block masks. It parsed combined split manifests to select training rows; computations used no non-training track contents, per-frame targets or predictions. Per-frame reads were limited to training GT and the three model-input validity masks; no feature vectors, encoder outputs or native sensor files were read. It did not train or evaluate models. This is feasibility evidence, not an E003 protocol decision or a performance estimate.

Accepted-GT runs are contiguous slots with the same mapped class; every GT gap and label change starts a new run. Transitions count adjacent label changes only within uninterrupted labeled segments. Slot duration is 0.2 s. Track-edge runs are context-censored, especially for LOKI whose physical timing is not independently verified.

| Source | Tracks; T median (P10–P90) | Accepted-GT runs; P25 / median / P90 slots | Singleton runs | Tracks with 0 / 1 / >1 transitions |
|---|---:|---:|---:|---:|
| ROAD-Waymo | 4,380; 85 (37–99) | 7,781; 6 / 15 / 50 | 4.50% | 3,712 / 568 / 100 |
| LOKI | 8,574; 23 (5–79) | 10,153; 10 / 20 / 57 | 1.48% | 7,372 / 882 / 320 |

Per-class accepted-GT run lengths (P25 / median / P90 slots):

| Source | Moving | Stopped | Waiting to cross | Crossing |
|---|---:|---:|---:|---:|
| ROAD-Waymo | 6 / 14 / 50 | 6 / 14 / 58 | 9 / 17 / 56 | 7 / 16 / 43 |
| LOKI | 9 / 18 / 52 | 11 / 18 / 51 | 12 / 22 / 68 | 14 / 28 / 69 |

Natural joint availability `(G, RGB, LiDAR)` by number of available branches `(0 / 1 / 2 / 3)` is ROAD context **3,462 / 0 / 187,032 / 142,446** and accepted GT **0 / 0 / 26,830 / 140,450**; LOKI context **1,774 / 1 / 150,132 / 118,448** and accepted GT **0 / 1 / 143,622 / 118,448**. Thus no accepted-GT slot is RGB-only under this three-flag contract. Training LiDAR-on-GT coverage is **83.96% ROAD / 99.998% LOKI**. LOKI RGB is missing for the entire track in 2,888/8,574 tracks; ROAD has no fully RGB-missing training tracks. Missing-run median lengths (slots; internal / leading / trailing) are RGB **2 / 11 / 40 ROAD** and **2 / 3 / 9 LOKI**; LiDAR **1 / 17 / 8** and **1 / 3 / 2**, respectively. The LOKI whole-track RGB-missing runs have median length12 slots; geometry-aggregate missing runs are internal, with median2 ROAD /3 LOKI. `G` is the exact E002 aggregate validity OR across pedestrian velocity, displacement, ego-relative position, ego velocity, 2D bbox geometry and 2D bbox velocity; it does not mean every geometry feature group is present. These are branch counts, not counts of independent sensors: geometry itself includes 2D bbox fields.

The input-only Monte Carlo keeps exactly `k` of the three branches uniformly per block, then intersects that choice with natural availability. With `k=1`, exact expected empty-input rates are **5.35% / 18.27%** of ROAD / LOKI accepted-GT slots, independent of block length; for `k=2` they are **0% / 0.00013%**, and for `k=3` zero. At `k=1` with one full-track block, two-repeat Monte Carlo made the entire accepted-GT track input-empty in **0.64% ROAD / 11.30% LOKI** samples; the entire full context was empty in **0% / 11.27%**. These are distinct: ROAD tracks can retain unlabeled context while every accepted-GT slot is empty. For LOKI, the P90 longest empty accepted-GT gap was 20 slots at L20 and 35 slots at L40, conditional on tracks longer than that block. `T < 20` applies to 3.06% ROAD and 42.90% LOKI; `T < 40` to 11.74% and 72.49%. Shorter tracks are excluded from each finite-L summary rather than clamped into repeated full-track cases; full-track is represented once per track. The Monte Carlo uses seed 0 and two mask draws per track-setting, so its gap percentiles are rough feasibility signals only.

The [summary JSON](../../outputs/experiments/E003/masking-feasibility-20261102/summary.json), [per-track masks](../../outputs/experiments/E003/masking-feasibility-20261102/training-masks.npz), [GT runs](../../outputs/experiments/E003/masking-feasibility-20261102/gt-runs.csv), [natural modality runs](../../outputs/experiments/E003/masking-feasibility-20261102/natural-modality-runs.csv), [mask simulation table](../../outputs/experiments/E003/masking-feasibility-20261102/mask-simulation.csv) and [one-off script](../../outputs/experiments/E003/masking-feasibility-20261102/analyze_masks.py) retain the detailed counts and reproduction inputs. No block length or dropout policy is selected here.
