# Evaluation plan

Status: Working protocol; metrics, splits, and budget units remain open

Last updated: 2026-10-02

## Purpose and ownership

- Contains: experimental comparisons, data-access regimes, split safeguards, label budgets, candidate metrics, and audit protocol.
- Links out: semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), population checks to [DATASET_INSPECTION_PLAN.md](DATASET_INSPECTION_PLAN.md), and run records to [EXPERIMENTS/README.md](EXPERIMENTS/README.md).

## Protocol gate before training

Fix and version the applicable dataset releases, association provenance, native labels, output semantics, splits, evaluated populations, metrics, model-selection rule, budget accounting, sampling method, and run/seed policy before the corresponding comparison. Definitions depend on the ontology and population gates; none is silently settled by this document.

Predeclare the hypothesis and baseline for each experiment. Record deviations and failures. Keep configs in `configs/`, results once in experiment records, and large artifacts under ignored `outputs/`. A performance change alone does not establish its cause or a novelty claim.

## First two-dataset diagnostic

Run both the framewise MLP and whole-track BiLSTM proposed in [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md#first-diagnostic-baseline) for the following matrix, after the protocol gate. This is a proposed first experiment, not a completed run or a frozen final method.

| Train dataset | Evaluate dataset | Purpose |
|---|---|---|
| ROAD-Waymo | Held-out ROAD-Waymo | Within-dataset learnability and source validation. |
| LOKI | Held-out LOKI | Within-dataset learnability and source validation for the reverse direction. |
| ROAD-Waymo | LOKI | Transfer from camera-selected supervision toward a 3D-first population. |
| LOKI | ROAD-Waymo | Reverse transfer and possible asymmetry. |

Keep pedestrian/ego features, encoders and classifier design identical across the model comparison; only the BiLSTM adds learned full-track temporal context. Use a consistent coordinate/temporal convention and the tentative four-state mapping from [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md#tentative-first-baseline-projection). Each source-trained model can support its within-dataset and cross-dataset evaluation; these are eight model/evaluation cells, not necessarily eight independent training runs.

Begin with ROAD behaviour-labeled tracks that have official 3D pairs. Required paired coverage and the initial LOKI cohort remain open: a visibility-comparable subset is reasonable for a clean diagnostic but cannot establish performance on the full 3D-only population. Predeclare cohort criteria and excluded denominators. Preserve whole tracks, missing observations and independent label masks rather than retaining only consecutive labeled/paired frames.

No RGB, raw LiDAR, scene encoders, factorized heads, extra datasets, distillation, modality dropout or domain adaptation enters this first comparison. No aggressive short-track filter is intended; settle only the minimal feature-validity criterion.

Poor within-dataset performance suggests insufficient kinematic evidence, label ambiguity or a data/model problem. A BiLSTM gain supports the value of learned temporal context under the chosen features. Strong within-dataset performance with poor transfer suggests dataset/ontology/domain mismatch. Better LOKI → ROAD-Waymo transfer is consistent with a selection-asymmetry hypothesis, but different class frequencies, ontology, geography, sensors and scene structure also change; directional scores alone cannot establish causality. Strong transfer motivates examination of difficult subsets and Stopped/Waiting failures.

## Later comparisons, conditional on baseline failures

- Kinematics plus scene versus kinematics alone, if road relation is an identified gap.
- Optional RGB/3D supervision and missing-visual-evidence comparisons for provisional RQ1.
- Factorized versus shared temporal representation for provisional RQ2.
- A single source versus one justified complementary dataset; inclusion/order remain open.
- Source-pretrained versus target scratch at equal target-label budgets.

These additions are not required to run the first diagnostic. The eventual primary transfer direction, and zero-shot versus low-shot emphasis, remain open.

## Strict zero-shot access

For ROAD-Waymo → LOKI, no LOKI observations are used for representation training, including unlabeled adaptation. For LOKI → ROAD-Waymo, apply the same exclusion to ROAD-Waymo. No target behaviour labels or evaluation scores select checkpoints, hyperparameters, architectures, features or source additions. Source validation determines the selected source model; feature normalization also uses source training data only.

Native annotation definitions may be compared to establish a predeclared evaluation projection. Existing qualitative LOKI inspection is disclosed in the [dataset note](DATASETS/LOKI.md#selected-clip-observations-2026-09-28); do not claim the target dataset was entirely unknown. Decide treatment of inspected scenarios at the split gate.

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
