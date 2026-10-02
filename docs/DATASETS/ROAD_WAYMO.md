# ROAD-Waymo

Status: Official documentation checked; release not acquired; 3D linkage unproven

Last updated: 2026-10-01

## Candidate role

Baseline real behaviour source. It becomes a multimodal source only if ROAD annotations can be linked robustly to original Waymo 3D tracks. Neither this role nor similar class names establishes LOKI compatibility. [Gate 1](../DATASET_INSPECTION_PLAN.md#gate-1--road-waymo--waymo-linkage) owns acceptance.

## Documented annotation form

The [official repository](https://github.com/salmank255/Road-waymo-dataset) describes frontal videos with agent, action, location, frame-box and tube annotations. Box records include `tube_uid` and native class-ID lists; label vocabularies are dataset-level fields. This is a documented format, not a locally inspected release.

The October checkpoint identifies move, stop, wait-to-cross, crossing and direction variants as potentially relevant concepts. Exact strings/IDs, definitions, pedestrian applicability, overlap, missingness, and annotation cadence in the acquired version remain **unknown**. No mapping or loss is accepted.

## Waymo correspondence and selection

Original Waymo is the intended source of sensor/3D observations. Its [labeling specifications](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md) describe camera and 3D labeling. Compatible release, timestamps, associations, ego data, and map coverage must be checked rather than inferred from the ROAD tube ID.

Waymo has separate Perception and Motion datasets; [the official repository](https://github.com/waymo-research/waymo-open-dataset) describes maps with the Motion dataset. Do not assume those map assets correspond to every ROAD-Waymo segment.

The frontal annotation population differs from LOKI's 3D-first population. Additional Waymo 3D pedestrians are not automatically ROAD behaviour-labeled. Matching must measure retained/excluded populations and ambiguous associations.

## Local status and next evidence

The user confirmed on 2026-10-01 that ROAD-Waymo has not been acquired and linkage is not implemented. Establish release provenance, clip/segment and frame mappings, 2D object/3D track association, and varied manual validation before claiming 3D behaviour supervision.

Record the acquired vocabulary and schema here; record semantic comparison in [LABEL_ONTOLOGY.md](../LABEL_ONTOLOGY.md). Do not download large assets as part of the documentation migration.

## Pedestrian population and sequence scale (2026-10-01)

The [official release README](https://github.com/salmank255/Road-waymo-dataset) reports **1,000 videos**, approximately **20 s** each, and 198k annotated frames. Its 54k agent tracks include all classes.

[Paper v1, Tables 11–12](https://arxiv.org/html/2411.01683v1) reports **11,759 pedestrian agent tubes**, including **2,186 test tubes**, and 867,407 pedestrian boxes. Subtraction gives **9,573 non-test tubes** (train + validation, not training alone). The table totals 52,362 agent tubes, whereas the headline says 54k; verify the acquired release rather than treating all published totals as identical.

| Pedestrian action | All action tubes | Test action tubes |
|---|---:|---:|
| Stop | 3,368 | 641 |
| Move away | 2,331 | 461 |
| Move towards | 2,162 | 403 |
| Move | 1,604 | 334 |
| Wait to cross | 516 | 106 |
| Crossing | 675 | 60 |
| Cross from right | 571 | 81 |
| Cross from left | 531 | 88 |

Action tubes are label-specific sequences; do not sum them as unique pedestrians or equate them to LOKI's distinct tracks per action. The **516 waiting tubes** are especially relevant to supervision scale. Usable 3D-matched counts, lengths, gaps and class coverage remain unknown.
