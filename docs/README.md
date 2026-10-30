# Research dashboard

Updated **2026-10-30**. We are developing a reliable **offline, given-track pedestrian behavior labeler**. Final architecture and publication contribution remain provisional. Deadline **17 December 2026**; **7–17 December** is protected for buffer and handover. The [literature review](literature/offline-labeling-research-direction.md) supports an empirical study first, with one conditional retrieval or supervision comparison.

## Current evidence

- **[E002a/E002b comparison](../experiments/E002b-normalization-capacity/results-review.md):** 26 selected seed0 source-validation models, plus a five-seed E002a RGB summary (**58.12% ±0.90 pp ROAD /66.74% ±0.50 pp LOKI**). Smaller-temporal geometry + LiDAR reaches **66.05% ROAD /76.41% LOKI** at seed0. [E002c](../experiments/E002c-rgb-regularization/README.md) finds no consistent gain from stronger temporal dropout, while RGB branch dropout raises seed0 validation F1 by up to2.08 pp ROAD /0.57 pp LOKI. Three LOKI LiDAR trainings remain interrupted with evaluated best checkpoints. No E002 held-out or cross-dataset results.
- **[E001/E001b/E001c review](../experiments/E001b-2d-bbox-contribution/results-review.md):** 130 completed five-seed attempts /260 held-out cells. BiLSTM improves over MLP; bbox geometry helps transfer in one direction; replacement native-box inputs and size velocities provide no consistent gain. These test results informed later development.
- **[Data and pretrained features](../experiments/E002a-feature-feasibility/README.md):** frozen populations/splits,3× DINO crops and indexed D-DITR features remain unchanged. The [packed spatial-readout benchmark](../experiments/E002a-feature-feasibility/README.md#packed-spatial-readout-benchmark--2026-10-29) rejects one segmented PyTorch implementation: model updates are 2.15–2.25× slower on matched ROAD/LOKI inputs. Earlier native/PCA/semantic probes, storage revisions, throughput measurements and failures stay in that owner. [ROAD split verification](datasets/road-waymo.md#official-download-verification-2026-10-16) retains the documented paper/release mismatch; experiment splits remain frozen.
- **[Prediction viewer](../README.md#inspect-model-predictions):** compares E001-family test or E002 validation predictions, including the recovered LOKI models. The [E002a RGB seed curves](../experiments/E002a-feature-feasibility/README.md#geometry--rgb-repeat--seeds-14-prepared-2026-10-28) show online training F1 and source validation for eight completed repeats; historical seed0 has no training-F1 curve.

## Next work

1. Run the bounded [E002d RGB–LiDAR complementarity pilot](../experiments/E002d-rgb-lidar-complementarity/README.md): four sequential seed0 source-validation attempts, with temporal dropout0.1 and RGB branch dropout0/0.5, against the existing LiDAR-only reference. Then decide whether matched repeat seeds and held-out transfer are warranted. Validation-guided development is not final generalization evidence.
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
