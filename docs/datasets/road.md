# ROAD — Road Event Awareness Dataset

Status: Official documentation checked; local release unverified

Last updated: 2026-10-01

## Candidate role

Possible visual behaviour source after the ROAD-Waymo baseline. Its inclusion and order relative to nuScenes remain open. The intended value is additional visual action/location evidence from the UK domain; this is a hypothesis to test.

## Documented annotation form

The [official repository](https://github.com/gurkirt/road-dataset) describes a dataset built on Oxford RobotCar, with video boxes/tubes and agent, action, and location class lists. The provided labels should be inspected in their native form.

Similar annotation structure does not prove that every pedestrian label or boundary matches ROAD-Waymo, or that either maps directly to LOKI. Extract the acquired pedestrian vocabulary, multi-label rules, cadence, gaps, and examples before choosing supervision.

## Modalities and limits

Treat ROAD as a visual source unless an independently validated correspondence supplies additional modalities. The [paper](https://doi.org/10.1109/TPAMI.2022.3150906) describes possible synchronization with original RobotCar sensors; that possibility does not establish convenient tracked 3D pedestrian supervision in the behaviour release.

Visual-only supervision must not be forced onto a 3D branch. Any paired/distillation objective requires actual paired observations with verified identities and timing.

## Remaining checks

Local acquisition/version, annotation coverage, camera preprocessing, track continuity, exact native definitions, access terms, and feasible source sampling remain unchecked. Determine whether its additional visual domain actually improves transfer before combinations or further sources. Record release findings here and comparisons in the [ontology audit](README.md#native-semantics-and-evidence).

## Pedestrian population and sequence scale (2026-10-01)

[Paper supplementary Table 15](https://arxiv.org/pdf/2102.11585) reports **4,212 pedestrian tubes** and **228,757 pedestrian boxes** overall, including **645 test tubes**. Its three alternative folds have **3,169 / 3,138 / 3,169 training pedestrian tubes** and **398 / 429 / 398 validation tubes**. These folds reuse data; their counts must not be added.

The [official release README](https://github.com/gurkirt/road-dataset) reports **22 videos**, approximately **8 minutes** each, and 122k annotated frames. Its 7k tubes cover all agent classes. Thus a small number of long videos yields thousands of pedestrian sequences, but only limited independent scene diversity. The mean pedestrian tube has about **54 boxes** (228,757 / 4,212); this is not a median, guaranteed continuity, or full eight-minute pedestrian visibility.

Recount the acquired release by native pedestrian tube ID, split and action; inspect length distributions and fragmentation before choosing temporal windows.
