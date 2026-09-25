# Pedestrian Stop and Go Forecasting with Hybrid Feature Fusion

Citation: Guo, Mordan, Alahi, 2022.  
Link: <https://arxiv.org/abs/2203.02489>

## Problem

Forecast abrupt pedestrian stop/go transitions, which trajectory-only forecasting can handle poorly.

## Inputs

Video sequences and high-level pedestrian/scene attributes in a hybrid fusion approach.

## Outputs / labels

Future stop/go behavior on the TRANS benchmark created from existing datasets.

## Online or offline?

Online forecasting, not offline annotation.

## Evaluation / relevance

It motivates studying transitions and temporal structure rather than treating behavior as independent frames.

## Dataset(s)

TRANS, assembled from existing pedestrian-motion data.

## Method

Hybrid fusion of pedestrian-specific, video, and scene features for stop/go forecasting.

## Most relevant results

The paper reports a benchmark improvement on TRANS; exact metrics are not needed for the present task until a temporal baseline is selected.

## What we can reuse

The problem framing: inspect stop/go transitions as temporal events, not merely independent labels.

## What it already solves / does not solve

It proposes transition forecasting and multimodal fusion; it does not settle a portable observable-behavior ontology or cross-dataset annotation protocol.

## Open questions raised

Whether transition constraints help offline labels without needing its model design.
