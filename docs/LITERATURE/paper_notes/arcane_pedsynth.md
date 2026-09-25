# ARCANE-PedSynth: Synthetic Multi-Pedestrian Datasets with Behavioural Crossing Annotations

Citation: Riaz, Wielgosz, López Peña, 2026.  
Link: <https://arxiv.org/abs/2605.24950>

## Problem

Generate reproducible synthetic multi-pedestrian autonomous-driving data with dense behavior annotations beyond binary crossing/non-crossing labels.

## Inputs

CARLA simulation with configurable pedestrian/traffic scenarios. The paper reports synchronized RGB, LiDAR, DVS, and estimated 2D pose keypoints for the resulting data.

## Outputs / labels

PedSynth++ provides per-frame crossing labels and a paper-reported 12-state behavioral FSM. [Table 4](https://arxiv.org/pdf/2605.24950) lists `RETREAT`; the post-checkpoint inspection of the public generator enum found `NORMAL_CROSSING` instead. Released CSV values, frequencies, and generator-version agreement remain unverified.

## Online or offline?

The framework supports data generation for pedestrian crossing prediction, but its dense state labels are a candidate supervision source for this project's offline annotation task.

## Dataset(s)

ARCANE-PedSynth framework; PedSynth++ example dataset (533 multi-pedestrian clips across 12 weather conditions, as reported).

## Method

CARLA-based generation with a hybrid AI/manual pedestrian controller and a 12-state behavior FSM; pose is estimated from rendered RGB rather than injected as perfect simulator skeleton ground truth.

## Evaluation

The paper demonstrates the generated dataset/framework. It does not establish PedSynth++→LOKI label compatibility or real-world multi-state zero-shot transfer.

## Most relevant results

It supplies a candidate synthetic source with rich behavior states, synchronized modalities, and an estimated-pose pathway that could be shared with real RGB.

## What we can reuse

The dataset as source supervision, shared pose-extraction principle, and explicit distinction between available sensors and simulator-private state.

## What this paper already solves

Synthetic behavior-rich pedestrian data generation and dense source labels.

## What it does not solve

Canonical ontology mapping, target observability, zero/few-shot transfer to LOKI, or automatic annotation of real unlabeled datasets.

## Relationship to our possible contribution

PedSynth++ is a candidate source dataset, not the contribution itself. The project would study whether and why its behavior supervision transfers to real-world tracks with minimal target labels.

## Open questions raised

Which raw states appear in the full release, and can they support LOKI adaptation without forced labels? The [paper's data availability statement](https://arxiv.org/pdf/2605.24950) says full PedSynth++ data is available from the corresponding author upon reasonable request; the Zenodo package is a demo subset.
