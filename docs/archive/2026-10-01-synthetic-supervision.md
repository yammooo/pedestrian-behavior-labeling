# Synthetic supervision investigation

Historical context, not current instructions. Preserves former decision record `0002`. Recorded 2026-10-01; direction clarified 2026-10-02. Release facts remain in the [dataset note](../datasets/pedsynth-plusplus.md).

## Generator findings and completed side work

The local sibling `carla-pedestrians/PED_SYNTH_PAPER_CODE_GAPS.md` records inspection of the baseline scenarios commit `2e47ab3`. The October checkpoint reports this active-path distinction:

| Reachability | States |
|---|---|
| Assigned in normal generation | WALKING_SIDEWALK, CROSSING_ROAD, FINISHED_CROSSING |
| Helper methods outside active run path | SUDDEN_CROSSING, JAYWALKING, DISTRACTED_BEHAVIOR |
| Declared but not entered | LOOKING_AROUND, CHECKING_TRAFFIC, HESITATING, RUNNING_ACROSS, PAUSING_MID_CROSS, NORMAL_CROSSING |

Checking/hesitation fields were placeholders. Exporting an enum or the three active states would not supply the advertised rich behaviour implementation. This is evidence about the inspected path/version, not every possible private implementation. The [public generator](https://github.com/wielgosz-info/carla-pedestrians) is distinct from downloaded data.

The checkpoint reports successful CARLA 0.9.13 reproduction on the Aalto RTX 4080 machine and an upstream PR opened for structured pedestrian 3D/ego exports. Local code corroborates the export implementation: scenarios commit `4ac8888` adds `pedestrians_3d.csv`, `ego_pose.csv`, and LiDAR mount metadata; parent checkout commit `ac9bfb2` pins it. PR URL and current upstream status are unknown. Do not imply merging or fully validated sensor alignment.

The local gap note also records a baseline smoke clip with raw sensors, missing rich FSM export, synchronization concerns, and duplicated visible-label rows. Geometry export remains useful infrastructure history; it does not fix behavioural semantics or make the downloaded release multimodal.

## Research history

Former Q1 (release access), Q3 (synthetic mapping), Q4 (three-way input intersection) and Q6 (rich synthetic label efficiency) are superseded/deferred by the evidence below.

The original hypothesis was that rich synthetic framewise behaviour supervision could reduce real labels needed for LOKI adaptation. It was provisional and never an accepted training mapping.

| Former raw-state mapping candidate | Former interpretation / unresolved risk |
|---|---|
| WALKING_SIDEWALK → MOVING | Needed released motion verification |
| CHECKING_TRAFFIC, HESITATING → WAITING_TO_CROSS | Needed comparable onset/episode semantics; absent from inspected active path |
| LOOKING_AROUND → conditional/exclude | Could accompany moving, stopping, or waiting |
| CROSSING_ROAD, JAYWALKING, RUNNING_ACROSS → CROSSING | Only while road relation supports crossing |
| SUDDEN_CROSSING → conditional MOVING/CROSSING | FSM entry might precede road entry |
| PAUSING_MID_CROSS → CROSSING | Active crossing can have zero speed |
| DISTRACTED_BEHAVIOR → conditional/exclude | Distraction is orthogonal to movement |
| FINISHED_CROSSING → conditional MOVING/STOPPED | Depended on later motion |
| RETREAT (paper), NORMAL_CROSSING (enum) → unresolved | Released presence and semantics unverified |
| No obvious state → generic STOPPED gap | Non-crossing stationarity coverage was unclear |

The former evaluation idea compared native rich, binary crossing, and collapsed-state pretraining against scratch at fixed LOKI track budgets, with zero-shot as a diagnostic. It depended on source labels and common inputs that were not established. Git retains the full former plans.

The route was demoted because the inspected release lacked expected 3D/sensor exports and the active generator lacked the needed rich states. The checkpoint also reports reviewed generated clips with close/ahead spawning, early clustered crossing interactions and few useful later interactions; these qualitative examples were not a population frequency study. Developer discussion reportedly involved crossing-label flicker and temporal smoothing; correspondence dates and independent validation are unavailable.

Naveed's later inability to access the promised richer code is checkpoint-reported correspondence. The operational assumption is that implementation is unavailable; the current project must not depend on its arrival. This does not establish that synthetic pretraining is permanently impossible.

The [former priority reasoning](#former-priority-record) owns the strategic priority. Controlled synthetic road/scene supervision, rare configurations, or sensor stress testing may return only if real-source experiments justify them; no new CARLA FSM is planned.

## Former priority record

Date: 2026-10-01

Status: Accepted investigation priority; methods remain provisional

2026-10-02 scope update: the real-data/baseline-first priority remains accepted. The latest [research direction](../research.md) now considers both ROAD-Waymo → LOKI and LOKI → ROAD-Waymo; the original main-target wording below records the 2026-10-01 context. Primary direction and the small proposed baseline remain provisional, not new settled decisions.

## Context

The former plan depended on rich synthetic behavioural supervision and usable shared observations. The inspected PedSynth++ release and active generator did not establish those requirements. Promised richer code is unavailable under the October checkpoint's working assumption. Completed CARLA infrastructure work remains valuable history.

## Decision

Demote synthetic pretraining from the foundation to optional future augmentation. Prioritize real-source feasibility: first establish ROAD-Waymo-to-Waymo 3D linkage and native semantics, then a ROAD-Waymo baseline and controlled transfer comparisons.

LOKI remains the main 3D-first target. Strict zero-shot excludes target training and model selection. Low-shot/scratch experiments have separate declared access rules.

## Alternatives considered

- Continue depending on promised rich PedSynth++ implementation: incompatible with current evidence and availability.
- Build the missing CARLA behaviour FSM now: substantial additional scope without a demonstrated necessity.
- Commit immediately to multiple real sources and a fixed multimodal architecture: premature before linkage and baseline evidence.

## Evidence / rationale

The user-provided 2026-10-01 checkpoint and planning clarifications establish this priority. [evidence/history](#research-history) distinguishes local release findings, inspected code, and reported correspondence. [LOKI observations](2026-09-28-loki-inspection.md#selected-clip-observations-2026-09-28) motivate scene and missing-RGB questions without proving a solution.

## Consequences

ROAD-Waymo is a candidate, not guaranteed 3D behaviour supervision. Failed linkage requires reconsidering source supervision. nuScenes/ROAD inclusion and order, later IDD-PeD, dataset-specific heads, factorization, architecture, metrics, and budget units remain open.

Preserve research history and CARLA work. Do not claim that synthetic transfer is permanently impossible or require ECP2.0 availability before the traineeship ends. The [roadmap](../research.md) retains buffer through 17 December.

## Revisit if

ROAD-Waymo linkage fails, controlled baselines reveal another supervision need, or independently validated synthetic data becomes available and addresses a measured gap.
