# ZOD / ZOD-IAC (historical reference)

Status: Historical / superseded as primary direction  
Last updated: 2026-09-22

## Why it matters

ZOD-IAC motivated this project’s research principles but is not its target architecture. Its pipeline generated tracks from detections and LiDAR-assisted 3D positions, then applied geometry-based crossing logic in keyframe image space.

## Lessons retained

- Upstream track validity can dominate annotation validity.
- Geometry and human labels can disagree, especially on crossing onset timing.
- Observable behavior, future prediction, and human semantic interpretation must be separated.
- Simpler evidence should be measured before adding learned fusion.

## Why it is not the starting point

Reliable validation of detection, association, lifting, continuity, and stitching would require too much manual work for the current traineeship scope. The new work begins after tracking, with datasets that already provide suitable tracks.

## Potential future reuse

Only generic concepts or dependency-free utilities may be ported deliberately after the new contract is settled. Do not port ZOD-specific detector, frustum lifting, GOLD/SILVER, cut/stitch, or keyframe-road assumptions.

## Inputs, labels, temporal structure, and quality metadata

Historical ZOD-IAC details are recorded in the project handoff. Reinspect the original repository only if a generic concept is deliberately considered for reuse.

## Source

- Historical repository: <https://github.com/munirfarzeen/zod-ped>.
