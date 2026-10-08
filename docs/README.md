# Research dashboard

Updated 2026-10-08. Research-definition stage; no comparison training or measured model results. The [current framing](research.md#research-questions) asks about offline recoverability/modality contribution (RQ1) and heterogeneous-supervision transfer (RQ2). [E001](../experiments/E001-kinematic-transfer/README.md) is **Planned** for training: four kinematic feature sets × MLP/BiLSTM × two sources, **16 runs / 32 evaluation cells**. Native readers, 5 Hz preparation, setup/batching and [models](../experiments/E001-kinematic-transfer/README.md#model-implementation-and-acceptance-2026-10-08) are implemented (12,364 LOKI / 6,608 ROAD-Waymo eligible tracks); the [ROAD association gate is accepted with qualifications](../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08), and clips/scenarios are independent by user assumption. [Local CPU/remote CUDA uv environments and the Git checkout on `aalto`](../README.md#setup-and-validation) are verified. Broader methods and the final ontology/architecture remain provisional. Traineeship ends **17 December 2026**, with **7–17 December** protected for buffer and handover.

[Tentative roadmap](research.md#tentative-experiment-direction): E001 baseline → E002 Transformer reference → E003 input evidence → E004 complementary supervision; one optional follow-up. Both RQs remain core objectives.

## Next work

1. Implement E001 training/evaluation/checkpoints, observation-condition strata and W&B logging with the agreed equal-track loss/F1 checks. Models and padding-invariance acceptance are implemented; native association acceptance is closed for E001 with recorded visual uncertainty. [W&B destination and login status](../experiments/E001-kinematic-transfer/README.md#run-provenance) are recorded; upload access remains untested.
2. Transfer the existing LOKI saved collection/setup to `aalto` before comparison runs; reuse the [frozen populations/splits/source statistics](../experiments/E001-kinematic-transfer/README.md#real-data-setup-evidence-2026-10-08).
3. Run the comparison and inspect failures before broader methods.
4. Alongside E001, define the first RGB/3D representations and verify auxiliary-source feasibility; follow the [tentative schedule](research.md#schedule).

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
