# Experiments

Shared evaluation safeguards and comparison index. [E001’s brief single-seed comparison review](E001-kinematic-transfer/README.md#brief-comparison-and-budget-review-2026-10-09) is recorded alongside implementation acceptance. Inspection galleries are evidence about data, not model experiments.

| ID | Question | Status | Result |
|---|---|---|---|
| [E001](E001-kinematic-transfer/README.md) | Four kinematic feature sets × MLP/BiLSTM; within and bidirectional ROAD-Waymo/LOKI transfer | 32 attempts complete; five-seed launcher ready | Longer patience helped some BiLSTMs; feature rankings mixed |
| [E001b](E001b-2d-bbox-contribution/README.md) | Exact K+T+R versus added bbox availability versus added geometry | Preparation verified; geometry-source gate open | Not trained |
| [E002](../docs/research.md#tentative-experiment-direction) | Kinematic Transformer reference and temporal-context comparison | Tentative | Not run |
| [E003](../docs/research.md#tentative-experiment-direction) | RGB/local 3D evidence on the Transformer | Tentative | Not run |
| [E004](../docs/research.md#tentative-experiment-direction) | One complementary supervision source with an additional-data control | Tentative | Not run |

[Research](../docs/research.md#tentative-experiment-direction) owns the tentative direction, branches and schedule. E002–E004 details/records are TBD; write their protocols before execution. Their IDs are reserved, with no model implementation or results.

## Before execution

Predeclare each question, baseline and key controls. Fix/version applicable releases and association provenance, native/output semantics, splits/cohorts, metrics/aggregation, selection rule, budget accounting, sampling and seed policy before its comparison. Definitions depend on the semantic/population gates; do not invent missing settings. Record failures, negative results and deviations. A score improvement alone does not establish causality or novelty.

## Strict zero-shot access

For ROAD-Waymo → LOKI, no LOKI observations are used for representation training, including unlabeled adaptation. For LOKI → ROAD-Waymo, apply the same exclusion to ROAD-Waymo. No target behaviour labels or evaluation scores select checkpoints, hyperparameters, architectures, features or source additions. Source validation determines the selected source model; feature normalization also uses source training data only.

Native annotation definitions may be compared to establish a predeclared evaluation projection. Existing qualitative LOKI inspection is preserved in the [inspection archive](../docs/archive/2026-09-28-loki-inspection.md#selected-clip-observations-2026-09-28); do not claim the target dataset was entirely unknown.

Within-dataset diagnostics and later low-shot experiments are separate access regimes. Freeze the baseline design and source-only selection rules before inspecting cross-dataset scores. Because both datasets are studied, do not call either corpus entirely unseen by the researcher. If target diagnostics or errors motivate a design change, disclose target-informed development; it cannot select a strict zero-shot model. Retain the original frozen comparison and separate any subsequent target-informed result.

Zero-shot must be tested when compatible semantics support it. Excellent zero-shot performance is not required for useful annotation-efficiency results.

## Low-shot label efficiency and human effort

Candidate adaptation budgets are 1%, 5%, 10%, 25%, and 100%. They are not accepted settings: explicit label counts may replace the smaller percentages. The population gate must determine the sampling unit and denominator. A 100% reference means the permitted training/development pool, never the held-out test set.

Compare scratch and source-pretrained methods using exactly the same target labels, input representation, architecture where applicable, and model-selection access. Report absolute unique tracks, annotated frames/duration, and scene coverage alongside any percentage. A track can contain several states; do not count it repeatedly as independent examples.

Declare how labels used for validation, early stopping, budget selection, or sampling are accounted for. Record any label knowledge used to stratify selection. Predeclare repeated draws/seeds where small budgets could be unstable.

No human-label quota or performance threshold has been accepted. If selective export is evaluated, report adaptation, audit, and review effort separately as well as their total. Do not equate reusing existing benchmark GT with measured human annotation time.

## Population and source ablations

Report evaluated pedestrian populations and modality availability at declared frame/track levels. [E001](E001-kinematic-transfer/README.md#samples-and-eligibility) includes eligible 3D-only LOKI tracks and defines annotated-box availability strata. Later RGB-visible cohorts require verified definitions. A missing 2D box is an availability signal, not automatically proof of being outside the camera FOV.

Examine class-wise errors, especially Stopped versus Waiting to cross, and failure patterns by distance, point sparsity, occlusion, and track length where these quantities are verified. Score only frames with applicable GT; do not fill missing behaviour labels as truth.

Test source additions individually before combinations. Keep splits, target labels, and evaluation rules fixed; record training exposure, compute, parameter counts, and tuning access so extra data/compute is not mistaken for a methodological effect. Negative transfer is a valid result. Modality, scene, and sensor-perturbation ablations require a named hypothesis.

## Recoverability and complementary supervision

Separate physical timing/horizon changes from dataset shift before interpreting transfer. Audit all pretraining data exposure for the declared access regime, including auxiliary tasks. Any context-removal invariance test needs a justified intervention and label policy: individually irrelevant elements can have a joint effect, and informative context should not be forced out of a representation. [Evidence](../docs/literature/transferable-motion-representations.md).

For RQ1, compare kinematics, added scene/LiDAR evidence and added RGB/context using matched evaluated frames/cohorts, GT semantics, splits and selection access. Report modality gains by class and observation condition (RGB availability, verified range, duration, occlusion and LiDAR sparsity); predeclare thresholds, metadata quality, support and exclusions. Natural missingness and controlled modality removal answer different questions. A weak model does not establish that sensing is insufficient; inspect data, labels, optimization and alternative explanations before attributing failure to ambiguity.

If a human-evidence comparison becomes useful, predeclare sample selection, annotator access to exactly the model's evidence and temporal extent, label definitions, blinding to benchmark/model labels, uncertainty/disagreement recording and effort. Human judgments are an independent diagnostic, not automatic replacement GT or proof of intrinsic ambiguity. Keep a target-informed audit separate from strict zero-shot model selection.

For RQ2, compare single-dataset end-task training with individually justified complementary sources and progressive combinations. Preserve native semantics/heads and independent supervision masks. To attribute gains to complementary supervision rather than extra data or compute, include appropriate matched data/exposure/tuning controls and, where feasible, supervision/task and shared-head/factorization ablations. Settings depend on the observed gap; no new comparison or dataset order is frozen. Report native label coverage, source/task conflicts, shortcuts, negative transfer and each target's access regime. Shared representation/native heads are candidates, not assumed solutions.

## Metrics for later comparisons

[E001](E001-kinematic-transfer/README.md#evaluation) owns its accepted primary/secondary metrics and strata. The candidates below concern later comparisons; they do not replace that protocol.

| Evaluation aspect | Candidates / definition needed |
|---|---|
| Frame states | Macro-F1, per-class precision/recall/F1, confusion matrices, balanced accuracy; aggregation and missing-GT policy. |
| Temporal quality | Segment overlap, segment F1 or edit measures, onset/offset error; segment definitions, thresholds, and gap handling. |
| Annotation efficiency | Target performance versus explicit label budget; any desired performance level must be declared first. |
| Missing RGB and source value | Matched cohort/modality/source comparisons with denominators and variability. |
| Conditional confidence experiment | Calibration and reliability versus automatic coverage; confidence definition, acceptance threshold, and independent audit. |

No study-wide primary metric, segment/event threshold, calibration method or numeric coverage target is canonical. Declare these per comparison.

## Split safeguards and later applications

Keep correlated frames and overlapping chunks of a track together. Verify scene/sequence/location grouping and identity overlap before accepting splits. Select adaptation examples only from permitted training scenes; freeze test manifests and never use held-out results for selection.

Confidence/abstention is an end goal and conditional experiment after baseline error analysis. ECP2.0 is a possible later application, not a required internship evaluation. Any export to an unlabeled dataset needs independent quality evidence; model-generated labels alone cannot validate that model.

## Comparison records and artifacts

Use a stable `E###-short-description/README.md` per meaningful comparison, with dates inside the record. Keep future comparison-specific configs beside it; shared configuration stays in `configs/`. Generated artifacts belong under ignored `outputs/experiments/E###/` or documented external storage. Create extra analysis files only when useful.

Each main record contains the question/baseline, design and controls; data/protocol references and unresolved settings; a compact variant/run table covering ablations, seeds, failures and results; config/command/revision/environment/output references; conclusions, limitations and next justified step.

For every actual run record code revision and uncommitted changes, exact command/effective config revision, dependency versions, hardware/host without credentials, start/end times, seed/draw and split/subset IDs, release/checksum provenance, available/derived modalities and masks, selected checkpoint and selection evidence, output locations including failures, metric artifacts, denominators/variability, compute/exposure/parameters and deviations. Use `unknown` or explain non-applicability.

Observed results live once in the comparison record. Dataset notes own measured release facts; [research](../docs/research.md) owns changed understanding; [log](../docs/log/README.md) records brief milestone pointers. Update only affected owners. Do not silently drop failed seeds or relabel target-informed development as strict zero-shot. Generated labels require independent validation.
