# Prioritize real-source feasibility and baseline evidence

Date: 2026-10-01

Status: Accepted investigation priority; methods remain provisional

2026-10-02 scope update: the real-data/baseline-first priority remains accepted. The latest [research direction](../RESEARCH_DIRECTION.md) now considers both ROAD-Waymo → LOKI and LOKI → ROAD-Waymo; the original main-target wording below records the 2026-10-01 context. Primary direction and the small proposed baseline remain provisional, not new settled decisions.

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

The user-provided 2026-10-01 checkpoint and planning clarifications establish this priority. [PedSynth++ evidence/history](../DATASETS/PED_SYNTH_PLUS_PLUS.md#research-history) distinguishes local release findings, inspected code, and reported correspondence. [LOKI observations](../DATASETS/LOKI.md#selected-clip-observations-2026-09-28) motivate scene and missing-RGB questions without proving a solution.

## Consequences

ROAD-Waymo is a candidate, not guaranteed 3D behaviour supervision. Failed linkage requires reconsidering source supervision. nuScenes/ROAD inclusion and order, later IDD-PeD, dataset-specific heads, factorization, architecture, metrics, and budget units remain open.

Preserve research history and CARLA work. Do not claim that synthetic transfer is permanently impossible or require ECP2.0 availability before the traineeship ends. The [roadmap](../ROADMAP.md) retains buffer through 17 December.

## Revisit if

ROAD-Waymo linkage fails, controlled baselines reveal another supervision need, or independently validated synthetic data becomes available and addresses a measured gap.
