# EuroCity Persons 2.0: A Large and Diverse Dataset of Persons in Traffic

Citation: Krebs, Braun, Gavrila, IEEE TPAMI 2024, 46(12), 10929–10943.  
Link: <https://doi.org/10.1109/TPAMI.2024.3471170>

## Problem

Large-scale person detection, tracking, and prediction data with dense pseudo-ground-truth trajectories from sparse keyframes.

## Inputs

Images, detections, sparse manual keyframes, auxiliary LiDAR, and vehicle inertial sensing during pseudo-GT construction.

## Outputs / labels

Dense 2D boxes and world-fixed 3D person locations, with trajectories; not a rich behavior-label dataset.

## Online or offline?

Its pseudo-GT process is offline track construction; it is not pedestrian behavior annotation.

## Evaluation / relevance

Candidate large-scale target for enrichment after behavior-labeling reliability is established on labeled datasets.

## Dataset(s)

ECP2.0.

## Method

Semi-supervised dense pseudo-ground-truth creation from sparse manual labels, detections, LiDAR-based uplift, ego-motion compensation, tracklet generation/merging, and smoothing.

## Most relevant results

The paper reports a roughly 34× speed-up over dense frame-wise manual annotation and ablations for its trajectory-generation process.

## What we can reuse

The separation between validated track generation and downstream behavior enrichment, plus quality-aware provenance thinking.

## What it already solves / does not solve

It solves much of the tracking pseudo-GT layer; it does not provide the target behavior labels currently sought.

## Open questions raised

How to validate added labels, consume weak/unsure metadata, and keep a method compatible with its public fields.
