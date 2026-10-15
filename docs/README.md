# Research dashboard

Updated **2026-10-15**. Research-definition stage; the [joint E001/E001b/E001c review](../experiments/E001b-2d-bbox-contribution/results-review.md) records **130 completed five-seed attempts / 260 test cells**, retaining earlier rounds. The [current framing](research.md#research-questions) asks about offline recoverability/modality contribution (RQ1) and heterogeneous-supervision transfer (RQ2). [E001](../experiments/E001-kinematic-transfer/README.md), E001b and E001c completed 80, 20 and 30 five-seed attempts respectively. Native readers, 5 Hz preparation, setup/batching, training/evaluation and [models](../experiments/E001-kinematic-transfer/README.md#model-implementation-and-acceptance-2026-10-08) are implemented (12,364 LOKI / 6,608 ROAD-Waymo eligible tracks); the [ROAD association gate is accepted with qualifications](../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08), and clips/scenarios are independent by user assumption. [Local CPU/remote CUDA uv environments and the Git checkout on `aalto`](../README.md#setup-and-validation) are verified. Broader methods and the final ontology/architecture remain provisional. Traineeship ends **17 December 2026**, with **7–17 December** protected for buffer and handover.

[Tentative roadmap](research.md#tentative-experiment-direction): E001 baseline → E002 Transformer reference → E003 input evidence → E004 complementary supervision; one optional follow-up. Both RQs remain core objectives.

## Next work

The [prediction review viewer](../README.md#inspect-model-predictions) now compares up to four saved runs, with scoped per-track scores, synchronized native context and separate case notes. All 77 CUDA tests and real browser checks on both datasets passed; [acceptance evidence](../outputs/inspection/prediction-viewer/acceptance/native-b297d5a/verification.json) records timings and 366 unchanged source artifacts. Use the isolated `aalto` viewer checkout; existing training checkouts remain unchanged.

E001c [completed all 30 five-seed attempts](../experiments/E001c-bbox-inputs/README.md#five-seed-results-2026-10-15), with source-selected epochs 10–48 and patience stops at 18–56. The [joint 130-attempt / 260-cell review](../experiments/E001b-2d-bbox-contribution/results-review.md#e001c-native-box-inputs) adds class/stratum diagnostics and PNG/SVG comparisons: basic boxes average below E001b geometry in every direction; 2D size velocities show no systematic gain. Added 3D size velocities remain zero/noise, not evidence of new dynamics. Original results and caches remain unchanged; no cache rebuild or larger epoch cap is justified by these checks.

1. Inspect the joint review's Stopped/Waiting and Crossing/Moving trade-offs in the prediction viewer before choosing the next bounded comparison. Preserve all seeds and disclose test-informed development; no larger epoch cap is justified by the stopping evidence.
2. Arrange a separate backup for valuable checkpoints/predictions; external storage is **TBD**. Frozen LOKI collection/setup are checksum-verified on `aalto`; reuse the [populations/splits/source statistics](../experiments/E001-kinematic-transfer/README.md#real-data-setup-evidence-2026-10-08).
3. Keep the [local numerical evidence and W&B navigation](../experiments/E001b-2d-bbox-contribution/results-review.md#artifacts-and-wb-navigation) usable. Confusion matrices exist in collapsed final-evaluation sections; the saved E001 workspace contains ten original axes, while E001b's three additional bbox plots are logged separately. Transition diagnostics score frame states near boundaries; transition-event/type F1 is not implemented. Broader RGB/3D directions remain tentative; follow the [schedule](research.md#schedule).

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
