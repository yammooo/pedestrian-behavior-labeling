# Research dashboard

Updated **2026-10-27**. We are developing a reliable **offline, given-track pedestrian behavior labeler**. Final architecture and publication contribution remain provisional. Deadline **17 December 2026**; **7–17 December** is protected for buffer and handover. The [literature review](literature/offline-labeling-research-direction.md) supports an empirical study first, with one conditional retrieval or supervision comparison.

## Current evidence

- **[E002a/E002b comparison](../experiments/E002b-normalization-capacity/results-review.md):** 26 selected seed0 source-validation models, 13 per dataset. Geometry + LiDAR with the smaller temporal representation reaches **66.05% ROAD /76.41% LOKI** track-weighted macro-F1. Normalization and branch compression are not universal improvements; RGB has different F1/CE trade-offs by source. Overfitting persists. Three LOKI LiDAR trainings were interrupted after 23 epochs; their best checkpoints now have complete evaluation-only exports, without restarting training. No E002 held-out or cross-dataset results.
- **[E001/E001b/E001c review](../experiments/E001b-2d-bbox-contribution/results-review.md):** 130 completed five-seed attempts /260 held-out cells. BiLSTM improves over MLP; bbox geometry helps transfer in one direction; replacement native-box inputs and size velocities provide no consistent gain. These test results informed later development.
- **[Data and pretrained features](../experiments/E002a-feature-feasibility/README.md):** frozen populations/splits,3× DINO crops and indexed D-DITR features remain unchanged. Earlier native/PCA/semantic probes, storage revisions, throughput measurements and failures stay in that owner. [ROAD split verification](datasets/road-waymo.md#official-download-verification-2026-10-16) retains the documented paper/release mismatch; experiment splits remain frozen.
- **[Prediction viewer](../README.md#inspect-model-predictions):** compares E001-family test or E002 validation predictions, including the recovered LOKI models. [Online training F1](../experiments/E002b-normalization-capacity/README.md#online-training-f1--2026-10-26) is implemented for future runs; historical models have no training-F1 curve. GPU/runtime verification of that new logging remains pending.

## Next work

1. Discuss the [comparison and shortlist](../experiments/E002b-normalization-capacity/results-review.md#what-this-supports-next) before declaring confirmation seeds, budgets and any new experiment. Keep original geometry controls; do not assume every modality or smaller model helps. Validation-guided development is not final generalization evidence.
2. Arrange a separate backup for valuable checkpoints/predictions; external storage is **TBD**. Original interrupted attempts and all earlier evidence remain preserved.
3. Choose at most one substantial follow-up using measured error/cost evidence and the [schedule](research.md#schedule). Broader ontology, heterogeneous supervision and reliability experiments remain hypotheses.

## Navigation and ownership

| Owner | Contents |
|---|---|
| [Research](research.md) | Task, RQs, provisional methods, priorities and schedule |
| [Architecture](../ARCHITECTURE.md) | Code responsibilities, data flow and implementation conventions |
| [Datasets](datasets/README.md) | Shared conventions/comparison; native notes own labels, schema, counts, sources and limitations |
| [Experiments](../experiments/README.md) | Shared evaluation/recording rules; each comparison owns its design, settings and observed results |
| [Literature](literature/README.md) | Prior work, references and qualified contribution comparison |
| [Archive](archive/README.md) | Dated investigation and former decision reasoning; historical context |
| [Log](log/README.md) | Brief chronology and milestone pointers |
| [Root README](../README.md) | Setup and runnable inspection/validation commands |

Update only affected owners; link instead of repeating explanations. Update this dashboard when status/priorities change and log meaningful milestones. Preserve sources and uncertainty; handoffs are project context, not independent evidence.
