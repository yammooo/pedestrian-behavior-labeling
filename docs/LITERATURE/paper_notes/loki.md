# LOKI: Long Term and Key Intentions for Trajectory Prediction

Citation: Girase et al., ICCV 2021, pp. 9803–9812.

Link: <https://arxiv.org/abs/2108.08236>

Project relevance updated: 2026-10-01; [local observations](../../DATASETS/LOKI.md) are recorded separately from paper findings.

## Problem

Joint trajectory prediction and frame-wise intention estimation for heterogeneous traffic agents.

## Inputs

Reported RGB, LiDAR, 2D/3D agent labels, trajectories, and scene/lane context.

## Outputs / labels

For pedestrians: Moving, Waiting to cross, Crossing the road, and Stopped. The paper derives an intention as an action four frames (0.8 s) in the future for its prediction experiment.

## Online or offline?

The published task is prediction-oriented. The source frame-wise actions can potentially support offline behavior-annotation evaluation, but that use requires annotation-definition inspection.

## Evaluation / relevance

LOKI is the 3D-first dataset in the initial bidirectional diagnostic with ROAD-Waymo; each direction has separate source/target access rules. Its future-action transformation must not be conflated with current-frame action; Waiting to cross can itself contain inferred intention.

## Dataset(s)

LOKI.

## Method

Joint recurrent trajectory prediction and intention estimation with scene-graph reasoning and long-term goal proposals.

## Most relevant results

The paper reports up to 27% improvement over cited trajectory-prediction baselines; reproduce or compare only after inspecting the released protocol.

## What we can reuse

The source action ontology and its explicit future-action transformation as a warning about label semantics.

## What it already solves / does not solve

It benchmarks multimodal intention-aware trajectory prediction; it does not establish a portable offline pseudo-labeling pipeline across datasets.

## Open questions raised

Whether native actions support a defensible ROAD-Waymo evaluation projection, how RGB-visible and 3D-only populations differ, and which states are identifiable from sparse evidence. Release observations and remaining coordinate/split questions belong in the dataset note.
