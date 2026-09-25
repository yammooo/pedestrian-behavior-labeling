# EuroCity Persons 2.0 (ECP2.0)

Status: Draft  
Last updated: 2026-09-25

## Why it matters to this project

ECP2.0 is a strong candidate deployment/enrichment target because it reports more than 250K unique person trajectories over more than two million images, across 29 cities in 11 European countries.

## Pedestrian tracking and 2D / 3D information

The paper reports dense pseudo-ground-truth 2D boxes, person IDs, world-fixed 3D person locations, and person height. Its generation uses sparse manual keyframes, detections, LiDAR uplift, ego-motion compensation, tracklet generation/merging, and smoothing. The current public representation should not be described as dense full 3D cuboids.

## Quality / uncertainty metadata

The handoff reports `unsure` frames and `weak` trajectories. Confirm exact values and intended semantics in the release.

## Behavior labels

No equivalent rich pedestrian behavior ontology is currently documented for this project. It is therefore not the main supervised behavior-ground-truth dataset.

## What could be input / ground truth

Dense tracks and locations are candidate pipeline inputs; quality metadata may support confidence. A manual audit or other predeclared protocol is needed to evaluate newly generated behavior labels.

## Possible role and portability concerns

Candidate enrichment and large-scale transfer target. Its large size, access process, and data volume (approximately 11 TB reported) require practical sampling and storage planning.

The current EuroCity website separates the **ECP Tracking** extension from the detection, 2.5D, and dense-pose packages. Tracking access requires a separate account/request; do not infer that assets in the other packages are part of the tracking download.

## Temporal structure / sampling frequency

Needs verification from the release.

## Scene / road annotations

Needs verification; do not infer their availability from the use of driving imagery.

## Open questions

- Exact public fields, frame rate, coordinate conventions, and access conditions.
- Whether road/scene information can be obtained from released assets.
- How to construct a defensible evaluation sample for generated labels.

## Sources

- Krebs, Braun, Gavrila, TPAMI 2024: [DOI](https://doi.org/10.1109/TPAMI.2024.3471170); [ECP Tracking access](https://eurocity-tracking-dataset.tudelft.nl/); [other ECP packages](https://eurocity-dataset.tudelft.nl/).
