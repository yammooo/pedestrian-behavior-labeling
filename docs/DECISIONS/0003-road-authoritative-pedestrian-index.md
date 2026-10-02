# ROAD defines the merged pedestrian population

Date: 2026-10-02

Status: Accepted user policy

## Context

ROAD behaviour annotations join to official Waymo camera-to-LiDAR associations, but some linked camera Pedestrian observations have native LiDAR Cyclist boxes. Missing associations and missing same-frame boxes also occur. The original strict-class coverage audit remains useful history.

## Decision

Use ROAD's `Ped` class as authoritative for the merged pedestrian task, regardless of original Waymo 3D class. Retain all ROAD pedestrian observations. Preserve original labels, geometry and disagreement flags; do not resize associated boxes. Use `has_3d_box` for same-frame 3D supervision. Do not create missing associations or interpolate boxes as ground truth.

## Alternatives considered

- Require native Waymo LiDAR type Pedestrian: excludes 419 officially associated observations contrary to the accepted policy.
- Relabel or resize original source boxes: destroys provenance and invents geometry.
- Treat any published LiDAR ID as a valid per-frame box: falsely supervises frames without geometry.

## Evidence / rationale

The user's 2026-10-02 handoff explicitly accepts ROAD authority. [Dataset findings](../DATASETS/ROAD_WAYMO.md#acquired-index-and-access-2026-10-02) identify the inspected remote export, audits, measured coverage and remaining validation limits.

## Consequences

Original Waymo classes are provenance fields, not merged pedestrian filters. Cyclist geometry may include more than the person. Keep these cases inspectable and support subgroup analysis. This policy does not establish ROAD–LOKI semantic equivalence, a model design, or a visual association error rate.

## Revisit if

Visual review finds incorrect published associations or incompatible geometry; any revised eligibility rule must preserve the original export and report excluded populations.
