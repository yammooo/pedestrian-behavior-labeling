# PIE and JAAD

Status: Draft
Last updated: 2026-09-22

## Why they matter

PIE and JAAD are established pedestrian crossing/intention datasets. They are useful for understanding label semantics and comparing prior work, but are not currently selected as primary datasets.

## Known project-relevant properties

They provide video-based pedestrian annotations used broadly for crossing/intention work. Exact annotation fields, ontologies, sampling, track structure, available context, and current licenses need verification before any experiment is proposed.

## Possible role

Literature comparison and potentially an additional cross-dataset test if a defensible ontology mapping and compatible inputs can be established.

## Available modalities, tracks, temporal structure, and scene annotations

All require release-level verification. This combined note is intentionally not a claim that PIE and JAAD expose identical fields.

## Behavior / action labels and quality metadata

Crossing-related labels are reported in the handoff; exact ontologies, temporal semantics, uncertainty fields, and ground-truth role remain open.

## Portability concerns

Classic crossing labels may encode future knowledge or task-specific intention definitions. Do not equate them automatically with frame-wise observable behavior labels.

## Sources

- Rasouli et al., [PIE dataset project](https://data.nvision2.eecs.yorku.ca/PIE_dataset/).
- Rasouli et al., [JAAD dataset project](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

## Task context

PIE concerns egocentric pedestrian behavior/intention estimation; JAAD concerns driver–pedestrian joint attention. Exact citations, prediction protocols, released fields and terms remain unverified. Their background use is vocabulary/semantics and possible qualitative stress tests after access/schema review; no benchmark method is adopted. Offline suitability and which labels use future knowledge remain open.
