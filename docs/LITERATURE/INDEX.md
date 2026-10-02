# Literature index

Status: Living index; contribution claims unverified

Last updated: 2026-10-01

## Purpose and ownership

- Contains: navigation and brief relevance of papers and primary dataset documentation.
- Links out: reviewed findings to paper notes, native evidence to dataset notes, and contribution comparisons to [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

## Current dataset evidence

These are primary documentation entry points, not completed method reviews or verified local releases.

| Reference | Current relevance | Evidence owner |
|---|---|---|
| [ROAD-Waymo repository](https://github.com/salmank255/Road-waymo-dataset) and [Waymo labeling specifications](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md) | Baseline source candidate; native behaviour/3D association must be established | [ROAD-Waymo note](../DATASETS/ROAD_WAYMO.md) |
| [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md) and [map tutorial](https://www.nuscenes.org/tutorials/map_expansion_tutorial.html) | Candidate motion/scene grounding and sensor diversity | [nuScenes note](../DATASETS/NUSCENES.md) |
| [ROAD paper](https://doi.org/10.1109/TPAMI.2022.3150906) and [repository](https://github.com/gurkirt/road-dataset) | Candidate visual action/location supervision | [ROAD note](../DATASETS/ROAD.md) |
| [IDD-PeD project](https://cvit.iiit.ac.in/research/projects/cvit-projects/iddped) and [repository](https://github.com/Ruthvik9/IDD-PeD) | Later optional visual/context supervision | [IDD-PeD note](../DATASETS/IDD_PED.md) |

## Existing paper notes

| Paper | Relevance to current study | Scope already addressed |
|---|---|---|
| [LOKI](paper_notes/loki.md) | Main target; separate native actions, inferred intention, and future-action targets | Multimodal trajectory/intention benchmark |
| [Minimizing Human Labeling](paper_notes/minimizing_human_labeling.md) | Prior automatic labels, target access and independent validation | Synthetic-to-real binary prediction/self-labeling |
| [ARCANE-PedSynth](paper_notes/arcane_pedsynth.md) | Historical synthetic hypothesis; distinguish claims from inspected code/data | Reported synthetic generation and rich labels |
| [Stop and Go Forecasting](paper_notes/stop_go.md) | Transition quality and temporal context | Future stop/go benchmark and fusion |
| [Rasouli & Kotseruba taxonomy](paper_notes/rasouli_kotseruba.md) | Action/intention/prediction vocabulary | Task distinctions |
| [PIE](paper_notes/pie.md), [JAAD](paper_notes/jaad.md) | Background label semantics | Image-centric crossing/intention research |
| [EMT](paper_notes/emt.md) | Background semantic comparison; no selected experiment | Gulf-region driving benchmark |
| [ECP2.0](paper_notes/ecp2.md) | Possible later application | Offline track pseudo-GT construction |

Add focused method reviews when they change a hypothesis, baseline, protocol, or interpretation. Heterogeneous partial supervision, semantic mismatch, offline segmentation, and RGB-to-3D behaviour transfer still need a closest-work review; missing notes are not evidence of novelty.
