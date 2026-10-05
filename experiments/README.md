# Experiments

Shared evaluation safeguards and comparison index. No trained-model results. Inspection galleries are evidence about data, not model experiments.

| ID | Question | Status | Result |
|---|---|---|---|
| [E001](E001-kinematic-transfer/README.md) | Framewise MLP versus whole-track BiLSTM; within and bidirectional ROAD-Waymo/LOKI transfer | Planned | Not run |

## Before execution

Predeclare each question, baseline and key controls. Fix/version applicable releases and association provenance, native/output semantics, splits/cohorts, metrics/aggregation, selection rule, budget accounting, sampling and seed policy before its comparison. Definitions depend on the semantic/population gates; do not invent missing settings. Record failures, negative results and deviations. A score improvement alone does not establish causality or novelty.

## Strict zero-shot access

For ROAD-Waymo → LOKI, no LOKI observations are used for representation training, including unlabeled adaptation. For LOKI → ROAD-Waymo, apply the same exclusion to ROAD-Waymo. No target behaviour labels or evaluation scores select checkpoints, hyperparameters, architectures, features or source additions. Source validation determines the selected source model; feature normalization also uses source training data only.

Native annotation definitions may be compared to establish a predeclared evaluation projection. Existing qualitative LOKI inspection is disclosed in the [inspection archive](../docs/archive/2026-09-28-loki-inspection.md#selected-clip-observations-2026-09-28); do not claim the target dataset was entirely unknown. Decide treatment of inspected scenarios at the split gate.

Within-dataset diagnostics and later low-shot experiments are separate access regimes. Freeze the baseline design and source-only selection rules before inspecting cross-dataset scores. Because both datasets are studied, do not call either corpus entirely unseen by the researcher. If target diagnostics or errors motivate a design change, disclose target-informed development; it cannot select a strict zero-shot model. Retain the original frozen comparison and separate any subsequent target-informed result.

Zero-shot must be tested when compatible semantics support it. Excellent zero-shot performance is not required for useful annotation-efficiency results.

## Low-shot label efficiency and human effort

Candidate adaptation budgets are 1%, 5%, 10%, 25%, and 100%. They are not accepted settings: explicit label counts may replace the smaller percentages. The population gate must determine the sampling unit and denominator. A 100% reference means the permitted training/development pool, never the held-out test set.

Compare scratch and source-pretrained methods using exactly the same target labels, input representation, architecture where applicable, and model-selection access. Report absolute unique tracks, annotated frames/duration, and scene coverage alongside any percentage. A track can contain several states; do not count it repeatedly as independent examples.

Declare how labels used for validation, early stopping, budget selection, or sampling are accounted for. Record any label knowledge used to stratify selection. Predeclare repeated draws/seeds where small budgets could be unstable.

No human-label quota or performance threshold has been accepted. If selective export is evaluated, report adaptation, audit, and review effort separately as well as their total. Do not equate reusing existing benchmark GT with measured human annotation time.

## Population and source ablations

Report LOKI all evaluable pedestrians, RGB-visible observations/tracks, and 3D-only observations/tracks. Exact cohort definitions and whether assignment is frame- or track-based remain open. A missing 2D box is an availability signal, not automatically proof of being outside the camera FOV.

Examine class-wise errors, especially Stopped versus Waiting to cross, and failure patterns by distance, point sparsity, occlusion, and track length where these quantities are verified. Score only frames with applicable GT; do not fill missing behaviour labels as truth.

Test source additions individually before combinations. Keep splits, target labels, and evaluation rules fixed; record training exposure, compute, parameter counts, and tuning access so extra data/compute is not mistaken for a methodological effect. Negative transfer is a valid result. Modality, scene, and sensor-perturbation ablations require a named hypothesis.

## Metrics still to define

| Evaluation aspect | Candidates / definition needed |
|---|---|
| Frame states | Macro-F1, per-class precision/recall/F1, confusion matrices, balanced accuracy; aggregation and missing-GT policy. |
| Temporal quality | Segment overlap, segment F1 or edit measures, onset/offset error; segment definitions, thresholds, and gap handling. |
| Annotation efficiency | Target performance versus explicit label budget; any desired performance level must be declared first. |
| Missing RGB and source value | Matched cohort/modality/source comparisons with denominators and variability. |
| Conditional confidence experiment | Calibration and reliability versus automatic coverage; confidence definition, acceptance threshold, and independent audit. |

No primary metric, temporal threshold, calibration method, or numeric coverage target is canonical yet.

## Split safeguards and later applications

Keep correlated frames and overlapping chunks of a track together. Verify scene/sequence/location grouping and identity overlap before accepting splits. Select adaptation examples only from permitted training scenes; freeze test manifests and never use held-out results for selection.

Confidence/abstention is an end goal and conditional experiment after baseline error analysis. ECP2.0 is a possible later application, not a required internship evaluation. Any export to an unlabeled dataset needs independent quality evidence; model-generated labels alone cannot validate that model.

## Comparison records and artifacts

Use a stable `E###-short-description/README.md` per meaningful comparison, with dates inside the record. Keep future comparison-specific configs beside it; shared configuration stays in `configs/`. Generated artifacts belong under ignored `outputs/experiments/E###/` or documented external storage. Create extra analysis files only when useful.

Each main record contains the question/baseline, design and controls; data/protocol references and unresolved settings; a compact variant/run table covering ablations, seeds, failures and results; config/command/revision/environment/output references; conclusions, limitations and next justified step.

For every actual run record code revision and uncommitted changes, exact command/effective config revision, dependency versions, hardware/host without credentials, start/end times, seed/draw and split/subset IDs, release/checksum provenance, available/derived modalities and masks, selected checkpoint and selection evidence, output locations including failures, metric artifacts, denominators/variability, compute/exposure/parameters and deviations. Use `unknown` or explain non-applicability.

Observed results live once in the comparison record. Dataset notes own measured release facts; [research](../docs/research.md) owns changed understanding; [log](../docs/log/README.md) records brief milestone pointers. Update only affected owners. Do not silently drop failed seeds or relabel target-informed development as strict zero-shot. Generated labels require independent validation.
