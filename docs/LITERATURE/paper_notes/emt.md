# EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving in the Arab Gulf Region

Citation: Abdel Madjid et al., 2025.

Link: <https://arxiv.org/abs/2502.19260>

Project relevance updated: 2026-10-01. Background reference; no current EMT experiment is selected.

## Problem

Autonomous-driving benchmark for tracking, trajectory forecasting, and intention prediction in the Arab Gulf region.

## Inputs

Dash-camera video is reported. The full release must be inspected for other modalities and track representations.

## Outputs / labels

Reported pedestrian categories: Stopping, Walking, Waiting to cross, Crossing. Exact semantics and timing need verification.

## Online or offline?

The reported intention benchmark is prediction-oriented; suitability of labels for offline behavior annotation remains open.

## Evaluation / relevance

Historically considered for domain-shift comparison. It remains a reminder that similarly named labels require a semantic audit; the active source/target experiment does not depend on EMT.

## Dataset(s)

EMT, reported as a Gulf-region dash-camera benchmark.

## Method

Dataset and benchmark paper; it reports tracking, trajectory forecasting, and intention-prediction evaluations rather than one project-reusable labeling method.

## Most relevant results

The paper reports more than 30,000 frames and 570,000 annotated boxes. Treat all task-specific label details as unverified until release inspection.

## What we can reuse

Potentially the dataset and its processing/evaluation code, after access and schema review.

## What it already solves / does not solve

It supplies a multi-task benchmark; it does not establish LOKI-to-EMT ontology equivalence or portable pseudo-labeling.

## Open questions raised

Exact fields, license, label definitions, and whether `Stopping` is a state or transition.
