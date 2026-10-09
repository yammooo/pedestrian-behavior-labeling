# Research dashboard

Updated 2026-10-09. Research-definition stage; comparison analysis is pending. The [current framing](research.md#research-questions) asks about offline recoverability/modality contribution (RQ1) and heterogeneous-supervision transfer (RQ2). The user is running [E001](../experiments/E001-kinematic-transfer/README.md): four kinematic feature sets × MLP/BiLSTM × two sources, **16 runs / 32 evaluation cells**. Native readers, 5 Hz preparation, setup/batching, training/evaluation and [models](../experiments/E001-kinematic-transfer/README.md#model-implementation-and-acceptance-2026-10-08) are implemented (12,364 LOKI / 6,608 ROAD-Waymo eligible tracks); the [ROAD association gate is accepted with qualifications](../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08), and clips/scenarios are independent by user assumption. [Local CPU/remote CUDA uv environments and the Git checkout on `aalto`](../README.md#setup-and-validation) are verified. Broader methods and the final ontology/architecture remain provisional. Traineeship ends **17 December 2026**, with **7–17 December** protected for buffer and handover.

[Tentative roadmap](research.md#tentative-experiment-direction): E001 baseline → E002 Transformer reference → E003 input evidence → E004 complementary supervision; one optional follow-up. Both RQs remain core objectives.

## Next work

1. The user is running the 16 E001 comparisons; wait for completion before analysis. [Training/evaluation acceptance](../experiments/E001-kinematic-transfer/README.md#trainingevaluation-implementation-and-acceptance-2026-10-08) passed CPU/CUDA checks and a separate ROAD BiLSTM smoke with actual W&B logging; [step/epoch curves and the tidy E001 workspace](../experiments/E001-kinematic-transfer/README.md#wb-workspace-and-logging) are verified. Comparison analysis is pending.
2. Arrange a separate backup for valuable checkpoints/predictions; external storage is **TBD**. Frozen LOKI collection/setup are checksum-verified on `aalto`; reuse the [populations/splits/source statistics](../experiments/E001-kinematic-transfer/README.md#real-data-setup-evidence-2026-10-08).
3. [E001b bbox preparation and numerical verification](../experiments/E001b-2d-bbox-contribution/README.md#real-preparation-and-numerical-verification-2026-10-08) are complete on unchanged E001 populations/data. The [geometry-source gate](../experiments/E001b-2d-bbox-contribution/README.md#geometry-source-review-and-open-gate-2026-10-09) remains open: several reviewed ROAD boxes are shifted from their actors. Agree the source rule before E001b training; the current caches remain intact. Broader RGB/3D directions remain tentative; follow the [schedule](research.md#schedule).

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
