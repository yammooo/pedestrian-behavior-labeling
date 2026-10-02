# Dataset matrix

Status: Current candidate roles; release and association checks incomplete

Last updated: 2026-10-02

## Purpose and ownership

- Contains: compact source/target roles, modality and annotation coverage, and verification status.
- Links out: detailed facts, primary sources, provenance, and caveats to native dataset notes.

**Documented** means official paper/format evidence; **observed** means local inspection; **checkpoint-reported** means project context not independently revalidated. An available sensor does not imply behaviour GT for its full tracked population.

| Dataset | Current role | Candidate observations | Native supervision / population | Main unresolved gate |
|---|---|---|---|---|
| [ROAD-Waymo](ROAD_WAYMO.md) | Baseline source candidate | Acquired FRONT RGB/2D labels with partial official same-frame Waymo 3D associations; LiDAR/ego components referenced; map coverage unknown | Native action/location labels; frontal annotation population | Visual association acceptance, native semantics, paired lengths/gaps and versioned reproduction code |
| [nuScenes](NUSCENES.md) | Possible next source; order open | RGB, LiDAR, calibrated 3D annotations/ego and scene expansions documented | Limited pedestrian motion attributes; scene labels at native cadence, not dense behaviour GT | Local packages and usable scene supervision unverified |
| [ROAD](ROAD.md) | Possible next visual source; order open | Video/2D tubes documented; no convenient 3D supervision assumed | Native agent/action/location annotations | Pedestrian semantics, temporal coverage and source benefit |
| [IDD-PeD](IDD_PED.md) | Later optional visual/context source | Tracked video/2D/context annotations documented; 3D correspondence unverified | Multiple frame-level behavioural/context attributes; selection needs audit | Native applicability, release coverage, and justification for inclusion |
| [LOKI](LOKI.md) | Main 3D-first target; strict zero-shot and low-shot/scratch regimes kept separate | Local RGB, 2D/3D rows, point clouds, odometry and map files; transforms remain partly unverified | Four actions in 3D label rows; GT can exist without a 2D observation | Complete population/cohort characterization, semantics, coordinates, splits |
| [PedSynth++](PED_SYNTH_PLUS_PLUS.md) | Historical route; optional future synthetic augmentation | Local RGB/2D release; metadata disables LiDAR/DVS; generator export work separate | Required rich FSM not available in inspected active generator | Not the current foundation; retain provenance and prior findings |
| [ECP2.0](ECP2.md) | Possible later application; no internship dependency | Dense person trajectories and metric locations paper-reported; released fields to verify | No matching rich behaviour GT established | Access, suitable 3D/context fields, independent audit |
| [EMT](EMT.md) | Background; no current experiment selected | Dash-camera tracking paper-reported; other modalities unknown | Similar action terminology, compatibility unverified | Reinspect only if deliberately selected later |
| [PIE / JAAD](PIE_JAAD.md) | Literature/background; not selected sources | Video/pedestrian annotations reported | Crossing/intention conventions require native interpretation | No current experiment depends on them |
| [ZOD-IAC](ZOD_IAC.md) | Historical motivation and prior work | Prior detection/lifting/tracking pipeline | Geometry/review-derived crossing records | Track construction is outside the current task |

## Pedestrian sequence scale

Counts below refer to native tracked identities/tubes, not training windows or frame boxes. Dataset notes own sources, splits and caveats.

| Dataset | Recording sequences | Pedestrian tracks/tubes | Evidence / limitation |
|---|---|---:|---|
| ROAD-Waymo | 1,000 × ~20 s videos; 798 with acquired GT | 11,759 paper; 9,573 acquired train/val | 6,540 acquired tracks with any paired 3D; 4,606 fully paired; test GT unavailable |
| nuScenes | 1,000 × 20 s scenes; 850 with public GT | 8,143 reported extraction | Independent study; release/split/category scope needs recount; limited motion attributes |
| ROAD | 22 × ~8 min videos | 4,212 | Paper; 645 test; visual tubes |
| IDD-PeD | Video count unknown; 45 s–10 min | 4,916 in release split description | 3,284 train / 1,632 test; paper says >5,000 |
| LOKI | 644 scenarios; 616 with 3D pedestrians | 12,364 with 3D/action GT | Local count; 4,139 never have a 2D box |

The headline 54k ROAD-Waymo / 7k ROAD / >28k LOKI agent counts include other classes. Usable supervision also depends on class balance, sequence length, behaviour coverage, 3D association and independent scenes. Overlapping windows cannot increase independent pedestrian population size.

## Comparison boundaries

ROAD-Waymo alone comes first if linkage passes. Whether nuScenes or ROAD is added next is undecided; IDD-PeD is optional later. More sources may hurt and must be tested.

Do not map all sources into LOKI's four classes. [LABEL_ONTOLOGY.md](../LABEL_ONTOLOGY.md) owns the native semantic audit; [DATASET_CONTRACT.md](../DATASET_CONTRACT.md) owns heterogeneous availability and leakage rules.

Detailed counts and sensor specifications remain in their dataset notes. Partial ROAD-Waymo correspondence is measured; training acceptance and LOKI compatibility remain open. This matrix does not declare nuScenes behaviour-equivalent or ECP data available before the traineeship ends.
