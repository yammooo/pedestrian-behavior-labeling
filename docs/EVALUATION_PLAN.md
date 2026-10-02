# Evaluation plan

Status: Working protocol; metrics, splits, and budget units remain open

Last updated: 2026-10-01

## Purpose and ownership

- Contains: experimental comparisons, data-access regimes, split safeguards, label budgets, candidate metrics, and audit protocol.
- Links out: semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), population checks to [DATASET_INSPECTION_PLAN.md](DATASET_INSPECTION_PLAN.md), and run records to [EXPERIMENTS/README.md](EXPERIMENTS/README.md).

## Protocol gate before training

Fix and version the applicable dataset releases, association provenance, native labels, output semantics, splits, evaluated populations, metrics, model-selection rule, budget accounting, sampling method, and run/seed policy before the corresponding comparison. Definitions depend on the ontology and population gates; none is silently settled by this document.

Predeclare the hypothesis and baseline for each experiment. Record deviations and failures. Keep configs in `configs/`, results once in experiment records, and large artifacts under ignored `outputs/`. A performance change alone does not establish its cause or a novelty claim.

## Baselines and transfer regimes

ROAD-Waymo is the baseline source candidate, conditional on validated 3D linkage. LOKI is the main target.

| Comparison | Purpose / condition |
|---|---|
| LOKI trajectory-only scratch baseline | Diagnose what verified kinematics can explain with target supervision. |
| LOKI trajectory + scene scratch baseline, if feasible | Measure scene value beyond kinematics. |
| ROAD-Waymo source baseline with held-out source evaluation | Establish source learnability and source-based model selection. |
| ROAD-Waymo → LOKI strict zero-shot | Test transfer with no target training or model selection, only after semantic projection is defensible. |
| Source-pretrained → LOKI versus LOKI scratch | Measure adaptation gains at exactly the same target-label budget and comparable architecture/inputs. |
| ROAD-Waymo plus one justified source versus ROAD-Waymo alone | Test heterogeneous supervision; nuScenes/ROAD order is undecided. IDD-PeD is later optional. |

Good source performance with poor transfer may indicate sensor, geography, scene, selection, or semantic shift. Poor target scratch performance may indicate insufficient evidence or ambiguous labels. These are diagnostics, not proofs.

## Strict zero-shot access

No LOKI observations are used for representation training, including unlabeled adaptation. No LOKI behaviour labels or evaluation scores select checkpoints, hyperparameters, architectures, or source additions. Source validation determines the selected source model.

Native annotation definitions may be compared to establish a predeclared evaluation projection. Existing qualitative LOKI inspection is disclosed in the [dataset note](DATASETS/LOKI.md#selected-clip-observations-2026-09-28); do not claim the target dataset was entirely unknown. Decide treatment of inspected scenarios at the split gate.

LOKI-supervised diagnostics and later low-shot experiments are separate access regimes. If target diagnostics or errors motivate a design change, disclose that target-informed development; it cannot be used to select the strict zero-shot model. Retain the original frozen comparison and separate any subsequent target-informed result.

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
