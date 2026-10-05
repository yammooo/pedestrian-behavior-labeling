# ROAD-Waymo linkage investigation

Historical context, not current instructions. Preserves former decision record `0003`. Recorded 2026-10-02. Current identity policy, counts, geometry and access paths belong in the [ROAD-Waymo note](../datasets/road-waymo.md).

## Structural recheck

The supplied independent export audit reports a full pass. On 2026-10-02 this repository inspection additionally rechecked all CSV rows, output hashes, annotation-key uniqueness, masks, geometry validity, track totals, duplicates, split/action counts and existence of all manifest paths. One paired observation was checked directly against original association and LiDAR-box Parquet rows. The remote merge synthetic self-check also passed. These checks support structural consistency, not a measured visual matching error rate; no new image/point-cloud visual audit was performed here.

Ignored inspection evidence: `outputs/inspection/road_waymo_handoff/` (remote document/script snapshots, `recheck.py`, `recheck.json`). Source index hashes in the remote report identify the inspected artifact; the CSV SHA256 is `1b1fa986ce901c803946e3b7f55ba8431fcd83fdc9de31aaacee9c6ab12b541f`.

## Visual samples and artifacts

Real inspection runs on `aalto`:

- `train_00015`, track `05204f5c-2029-48f7-820a-8f5714de4f2d`, selected with `Wait2X`: frames inspected with missing 3D, missing both boxes and paired crossing observations. Early held focus is about 96 m ahead of ego; a blank view there is outside available LiDAR returns, not a missing cloud file.
- `train_00019`, track `140a5e3c-7603-4fdd-ac52-f6fb556d2a44`, selected with `Stop`: inspected frame 61 preserves the original Cyclist geometry and marks the disagreement; frames outside ROAD labels have no inferred boxes.

Both videos contain 198 frames at **1564×604, 5 FPS** (39.6 s). Decoded video frames were inspected for overlays, markers and qualitative point/footprint alignment. This is a small visual sample, not an exhaustive association-quality audit. Six unit tests pass locally in `pedestrian-behavior` and remotely in the existing `zod-iac` environment; synthetic checks cover decoding/motion correction, mask semantics, repeated-observation handling, timestamp-based focus and ordered component streams.

Ignored local artifacts and HTML indexes: `outputs/inspection/road_waymo/waiting-001/` and `outputs/inspection/road_waymo/class-conflict-000/`. Remote counterparts are under `/home/user20/road_waymo_mapping/inspection_code_20261002/outputs/inspection/road_waymo/`; a source snapshot is staged in that directory with `src/` on `PYTHONPATH`. The first waiting render (`waiting-000`) preceded the empty-view marker and array-reader improvement; `waiting-001` uses the updated implementation. Original datasets and the merged index were not modified.

## Former population-policy record

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

The user's 2026-10-02 handoff explicitly accepts ROAD authority. [Dataset findings](../datasets/road-waymo.md#acquired-index-and-access-2026-10-02) identify the inspected remote export, audits, measured coverage and remaining validation limits.

## Consequences

Original Waymo classes are provenance fields, not merged pedestrian filters. Cyclist geometry may include more than the person. Keep these cases inspectable and support subgroup analysis. This policy does not establish ROAD–LOKI semantic equivalence, a model design, or a visual association error rate.

## Revisit if

Visual review finds incorrect published associations or incompatible geometry; any revised eligibility rule must preserve the original export and report excluded populations.
