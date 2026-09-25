# Research direction

Status: Draft  
Last updated: 2026-09-24

## Purpose and ownership

- Contains: motivation, central research question, hypotheses, and criteria for continuing or pivoting.
- Links out: operational task to [PROBLEM_DEFINITION.md](PROBLEM_DEFINITION.md), label semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), and comparisons to [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## Current summary

The prior ZOD-specific crossing-label refinement direction was set aside because validating its upstream detections, 3D lifting, tracking, and identity continuity would require substantial manual work. This project instead investigates whether pedestrian behavior can be annotated from existing, reasonably validated tracks in autonomous-driving datasets.

The primary task is **label-efficient offline automatic pedestrian-behavior annotation**. The current research question is whether fine-grained synthetic behavior supervision reduces the real human annotation needed to learn LOKI-style `MOVING`, `STOPPED`, `WAITING_TO_CROSS`, and `CROSSING` labels. PedSynth++ is a candidate supervision source; LOKI is the main real benchmark. Their actual released labels and shared inputs must be inspected before the synthetic route becomes a project foundation.

## Decided

- This is a new repository and not an extension of ZOD-IAC.
- The immediate phase is research formalization and data inspection, not model implementation.
- The tracked pedestrian, rather than a ZOD sequence, is the intended future core abstraction.
- Offline annotation is the primary task ([decision record](DECISIONS/0001-offline-annotation-primary-task.md)).
- ZOD-IAC is a methodological reference, not this system's architecture foundation.

## Working hypotheses

- PedSynth++ pretraining may improve LOKI performance at fixed small budgets of labeled pedestrian tracks. Zero-shot transfer remains a diagnostic, not a condition for success.
- Rich synthetic supervision may reduce the real track budget more than binary crossing supervision; this is an experimental hypothesis.
- EMT is a possible external test; ECP2.0 is a later enrichment target if transfer and independent audit justify it.

## Continuation gate

Inspect released PedSynth++ and LOKI annotations, especially `STOPPED` coverage and the shared input fields. The [inspection plan](DATASET_INSPECTION_PLAN.md) specifies the evidence. [Open questions](OPEN_QUESTIONS.md) tracks unresolved decisions.

If PedSynth++ is unsuitable, the fallback remains LOKI-supervised behavior labeling, cross-dataset transfer, and uncertainty-aware enrichment with a measured human budget. This is a contingency, not a selected alternative direction.

The [literature gap analysis](LITERATURE/GAP_ANALYSIS.md) tracks the separate novelty question.
