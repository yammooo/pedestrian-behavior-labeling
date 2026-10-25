# E002 — Multimodal offline reference

**Overview, updated 2026-10-25.** E002 groups the multimodal offline-labeling direction. Each concrete comparison owns its architecture, settings, preparation and results. [E002a](../E002a-feature-feasibility/README.md#completed-source-validation-runs--2026-10-25) completed eight seed0 source-validation runs; [E002b](../E002b-normalization-capacity/README.md) defines the agreed normalization/capacity follow-up. Earlier extraction, timing and operational evidence remains in E002a.

## Question and interpretation

Do appearance and local 3D scene evidence improve offline pedestrian behavior labeling beyond geometry, including incomplete RGB observations and cross-dataset transfer? Establish an empirical reference before considering one new retrieval or supervision mechanism. Neither modality benefit nor architectural novelty is assumed; see the [literature positioning](../../docs/literature/offline-labeling-research-direction.md) and [research direction](../../docs/research.md#tentative-experiment-direction).

The [E001-family review](../E001b-2d-bbox-contribution/results-review.md) motivates geometry as a pragmatic reference, not a universal winner. Existing test inspection informs this development and must be disclosed.

## Comparisons and ownership

| Record | Scope | Status |
|---|---|---|
| [E002a — first multimodal behavior pilot](../E002a-feature-feasibility/README.md) | Original three arms plus a separate geometry + D-DITR follow-up; frozen encoders, complete tracks, indexed point supply; Utonia deferred | Eight seed0 source-validation runs complete; geometry + D-DITR best in this seed |
| [E002b — normalization and capacity](../E002b-normalization-capacity/README.md) | Geometry / +RGB / +LiDAR; normalized reference versus smaller temporal or branch representations | Independent geometry/RGB user-started; shared LiDAR equivalence/timing pass, online smoke pending specific approval |

E002a's architecture is specific to that pilot. Later comparisons may revise it with explicit hypotheses and controls; they do not inherit its encoders, widths or readout by default. Completed PCA, alignment, semantic and resource probes remain [preparatory evidence inside E002a](../E002a-feature-feasibility/README.md#preparatory-work), with original settings and failures preserved.

The22 October revision supplies only enabled descriptors and their validity (65/194/323→256 fusion). It tests the utility of adding a modality including availability. Earlier shared-three-flag attempts retain their original interpretation; do not pool protocols. Existing feature caches remain valid, but revised model/execution hashes require new resource calibration and the epoch-time gate remains unmet.

## Shared constraints and next decision

Retain the given-track, full-record offline contract, eligible missing-observation/3D-only pedestrians and explicit GT/input masks. Predeclare populations, splits, source-only fitting, model-selection access and budgets under the [shared evaluation safeguards](../README.md#strict-zero-shot-access). Disclose test-informed development, pretrained exposure and any target-informed choices; frozen encoders do not erase prior access.

E002a's [completed validation results](../E002a-feature-feasibility/README.md#completed-source-validation-runs--2026-10-25) show useful geometry + LiDAR performance in one seed and loss overfitting, especially in ROAD RGB arms. They do not identify a shortcut mechanism or establish transfer. E002b fixes three variants: final branch LayerNorm at the original widths; LayerNorm with temporal128; LayerNorm with geometry32/RGB64/LiDAR64 and temporal256. Its owner defines unchanged data/loss/fitting and the conditional use of E002a controls. Shared LiDAR preparation now passes bounded execution equivalence/timing checks (1.10–1.16× warm training-update speedup), with actual W&B integration pending specific approval. Historical single-model timings and the [epoch-time gate](../E002a-feature-feasibility/README.md#throughput-gate--agreed-2026-10-22) remain qualified by build/scope; a full shared epoch is unmeasured.

Full multi-seed held-out modality/transfer comparisons remain deferred. E002b is source-validation development; it does not authorize the previously discussed40-attempt /80-test-cell comparison. Confirmatory seed sets and final test access remain separately declared.
