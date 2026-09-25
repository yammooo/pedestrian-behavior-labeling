# Offline annotation is the primary task

Date: 2026-09-22  
Status: Accepted

## Context

The project aims to generate behavior labels for recorded autonomous-driving datasets. Online intention prediction would forbid future observations and answer a different deployment problem.

## Decision

Treat the primary task as offline per-frame pedestrian-behavior annotation. The labeler may use complete tracks, future frames, bidirectional temporal processing, and non-causal post-processing.

## Alternatives considered

- Online future-action/intention prediction.
- A single system constrained to support both tasks from the start.

## Evidence / rationale

The intended output is a dataset label, not an in-vehicle action at time `t`. Offline context directly addresses ambiguous behavior such as stationary pedestrians near a curb whose later motion may disambiguate a state.

## Consequences

Evaluation and architecture may use bidirectional context. Any later online comparison must be explicitly separated and causal.

## Revisit if

The project goal changes from dataset enrichment to onboard prediction.
