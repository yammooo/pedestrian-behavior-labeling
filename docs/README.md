# Research dashboard

Updated 2026-10-09. Research-definition stage; E001 has a brief single-seed comparison review. The [current framing](research.md#research-questions) asks about offline recoverability/modality contribution (RQ1) and heterogeneous-supervision transfer (RQ2). [E001](../experiments/E001-kinematic-transfer/README.md) completed two rounds of four kinematic feature sets × MLP/BiLSTM × two sources: **32 attempts / 64 evaluation cells**. Native readers, 5 Hz preparation, setup/batching, training/evaluation and [models](../experiments/E001-kinematic-transfer/README.md#model-implementation-and-acceptance-2026-10-08) are implemented (12,364 LOKI / 6,608 ROAD-Waymo eligible tracks); the [ROAD association gate is accepted with qualifications](../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08), and clips/scenarios are independent by user assumption. [Local CPU/remote CUDA uv environments and the Git checkout on `aalto`](../README.md#setup-and-validation) are verified. Broader methods and the final ontology/architecture remain provisional. Traineeship ends **17 December 2026**, with **7–17 December** protected for buffer and handover.

[Tentative roadmap](research.md#tentative-experiment-direction): E001 baseline → E002 Transformer reference → E003 input evidence → E004 complementary supervision; one optional follow-up. Both RQs remain core objectives.

## Next work

1. Run the accepted repetition launcher after the [brief E001 budget/metric review](../experiments/E001-kinematic-transfer/README.md#brief-comparison-and-budget-review-2026-10-09): 30/5 and user-modified 50/8 rounds are complete, all seed 0. Longer patience helped some BiLSTMs; the [five-seed 75/8 launcher](../README.md#e001-multiple-seeds) is implemented, defaulting to two concurrent attempts. User reports the five-seed repetition is running; leave its checkout/environment unchanged. [Training/evaluation acceptance](../experiments/E001-kinematic-transfer/README.md#trainingevaluation-implementation-and-acceptance-2026-10-08) passed CPU/CUDA checks and a separate ROAD BiLSTM smoke with actual W&B logging; [step/epoch curves and the tidy E001 workspace](../experiments/E001-kinematic-transfer/README.md#wb-workspace-and-logging) are verified. A stable feature ranking remains unknown.
2. Arrange a separate backup for valuable checkpoints/predictions; external storage is **TBD**. Frozen LOKI collection/setup are checksum-verified on `aalto`; reuse the [populations/splits/source statistics](../experiments/E001-kinematic-transfer/README.md#real-data-setup-evidence-2026-10-08).
3. [E001b bbox preparation and numerical verification](../experiments/E001b-2d-bbox-contribution/README.md#real-preparation-and-numerical-verification-2026-10-08) are complete on unchanged E001 populations/data. The user accepted [native Waymo FRONT boxes without ROAD fallback](../experiments/E001b-2d-bbox-contribution/README.md#native-only-cache-acceptance-2026-10-09); both native-only caches are complete and verified on both machines. The [documented root override](../README.md#e001b-bbox-extension) resolves the current `/run/media/...` mount without changing frozen E001 references. Retain the original ROAD-preferred evidence. E001b is BiLSTM only; its [shared runner and 20-attempt launcher](../experiments/E001b-2d-bbox-contribution/README.md#shared-runner-and-launcher-2026-10-09) are implemented, with CPU acceptance complete and CUDA/smoke checks next. Full comparison execution follows E001. Broader RGB/3D directions remain tentative; follow the [schedule](research.md#schedule).

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
