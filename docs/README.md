# Research dashboard

Updated 2026-10-08. Research-definition stage; no labeling model or trained-model results. The [current framing](research.md#research-questions) asks about offline recoverability/modality contribution (RQ1) and heterogeneous-supervision transfer (RQ2). [E001](../experiments/E001-kinematic-transfer/README.md) is **Planned** for modeling: four kinematic feature sets × MLP/BiLSTM × two sources, **16 runs / 32 evaluation cells**. Its native readers, 5 Hz preparation and saved-track setup/batching with acceptance checks are implemented (12,364 LOKI / 6,608 ROAD-Waymo eligible tracks); [track inspection/audits](../README.md#prepare-and-inspect-complete-tracks) are ready for review, with remaining training gates. [Local CPU/remote CUDA uv environments and the Git checkout on `aalto`](../README.md#setup-and-validation) are verified. Broader methods and the final ontology/architecture remain provisional. Traineeship ends **17 December 2026**, with **7–17 December** protected for buffer and handover.

[Tentative roadmap](research.md#tentative-experiment-direction): E001 baseline → E002 Transformer reference → E003 input evidence → E004 complementary supervision; one optional follow-up. Both RQs remain core objectives.

## Next work

1. Review the sixteen-track native/prepared inspection pack and full inventory audits; resolve ROAD-Waymo association acceptance. The 59 known behavior conflicts have [accepted overrides](../experiments/E001-kinematic-transfer/behavior-overrides.json).
2. Review [data-setup acceptance/population evidence](../experiments/E001-kinematic-transfer/README.md#saved-tracks-to-batches-2026-10-08), frozen clip splits and source-only normalization; clip/scenario independence is accepted as an E001 assumption.
3. Implement/run E001's MLP/BiLSTM comparison after its remaining gates; inspect failures before broader methods.
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
