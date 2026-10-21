# E002 — Multimodal offline reference

**Overview, updated 2026-10-21.** E002 groups the multimodal offline-labeling direction. Each concrete comparison owns its architecture, settings, preparation and results; the first is [E002a](../E002a-feature-feasibility/README.md). Its model/trainer are implemented and bounded GPU integration is checked; full pilot comparisons remain unlaunched.

## Question and interpretation

Do appearance and local 3D scene evidence improve offline pedestrian behavior labeling beyond geometry, including incomplete RGB observations and cross-dataset transfer? Establish an empirical reference before considering one new retrieval or supervision mechanism. Neither modality benefit nor architectural novelty is assumed; see the [literature positioning](../../docs/literature/offline-labeling-research-direction.md) and [research direction](../../docs/research.md#tentative-experiment-direction).

The [E001-family review](../E001b-2d-bbox-contribution/results-review.md) motivates geometry as a pragmatic reference, not a universal winner. Existing test inspection informs this development and must be disclosed.

## Comparisons and ownership

| Record | Scope | Status |
|---|---|---|
| [E002a — first multimodal behavior pilot](../E002a-feature-feasibility/README.md) | Three-arm geometry/RGB/frozen D-DITR pilot; full tracks/all crop voxels with memory-budgeted batching and conditional recomputation; Utonia deferred | Model/trainer/caches and manual launcher implemented; GPU integration checked; full RGB preparation/readback complete; no full pilot results |

E002a's architecture is specific to that pilot. Later comparisons may revise it with explicit hypotheses and controls; they do not inherit its encoders, widths or readout by default. Completed PCA, alignment, semantic and resource probes remain [preparatory evidence inside E002a](../E002a-feature-feasibility/README.md#preparatory-work), with original settings and failures preserved.

## Shared constraints and next decision

Retain the given-track, full-record offline contract, eligible missing-observation/3D-only pedestrians and explicit GT/input masks. Predeclare populations, splits, source-only fitting, model-selection access and budgets under the [shared evaluation safeguards](../README.md#strict-zero-shot-access). Disclose test-informed development, pretrained exposure and any target-informed choices; frozen encoders do not erase prior access.

RGB preparation/readback and [operational acceptance](../E002a-feature-feasibility/README.md#model-and-training-acceptance--2026-10-21) are complete. Next, manually launch the source-training/validation pilot. Semantic quality and feature visualizations support feasibility; they do not establish behavior usefulness. Feature caches and training are implemented; bounded smoke attempts are operational evidence only.

Later modality ablations remain deferred: geometry+availability, +RGB, +LiDAR without RGB, +both. Five seeds × two sources × four variants would require 40 attempts / 80 test cells, separately authorized after pilot settings freeze. Their detailed architecture and protocol remain unset.
