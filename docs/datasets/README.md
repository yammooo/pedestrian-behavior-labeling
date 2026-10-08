# Datasets and shared conventions

The current study contract is **RGB + 2D pedestrian tracks + LiDAR/3D point clouds + 3D pedestrian tracks + ego motion + timestamps/calibration** for a general offline labeler. This specifies the evidence setting, not guaranteed visibility at every timestep: 3D-only, occluded or missing observations remain part of the problem. [E001](../../experiments/E001-kinematic-transfer/README.md) defines its kinematic fields, grid, partial validity and eligibility; [LOKI preparation transforms](loki.md#verified-preparation-transform-2026-10-07) now have full-release evidence; physical LOKI time and independent sensor calibration accuracy remain unverified. Fields for later models remain open. A partial-supervision source need not expose every contract modality; image-only sources may supervise a compatible branch without inventing 3D correspondence. PedSynth++/LOKI/ECP2.0 no longer define a three-way input intersection.

**Documented** means official paper/format evidence; **observed** means local inspection; **checkpoint-reported** means project context not independently revalidated. An available sensor does not imply behaviour GT for its full tracked population.

| Dataset | Current role | Candidate observations | Native supervision / population | Main unresolved gate |
|---|---|---|---|---|
| [ROAD-Waymo](road-waymo.md) | Initial diagnostic dataset; source and target in separate directions | Acquired FRONT RGB/2D labels, partial official 3D pairs and native-ID context extensions; native LiDAR/ego pose preparation; map coverage unknown | Native action/location labels; frontal annotation population | Visual association acceptance, final eligibility and versioned reproduction code |
| [nuScenes](nuscenes.md) | Possible next source; order open | RGB, LiDAR, calibrated 3D annotations/ego and scene expansions documented | Limited pedestrian motion attributes; scene labels at native cadence, not dense behaviour GT | Local packages and usable scene supervision unverified |
| [ROAD](road.md) | Possible next visual source; order open | Video/2D tubes documented; no convenient 3D supervision assumed | Native agent/action/location annotations | Pedestrian semantics, temporal coverage and source benefit |
| [IDD-PeD](idd-ped.md) | Later optional visual/context source | Tracked video/2D/context annotations documented; 3D correspondence unverified | Multiple frame-level behavioural/context attributes; selection needs audit | Native applicability, release coverage, and justification for inclusion |
| [LOKI](loki.md) | Initial 3D-first diagnostic dataset; source and target in separate directions | Local RGB, 2D/3D rows, point clouds, odometry and map files; annotation/world transform supported; RGB projection unverified | Four actions in 3D label rows; GT can exist without a 2D observation | Physical timing, independent sensor accuracy, semantic limitations and final population/split audit |
| [PedSynth++](pedsynth-plusplus.md) | Historical route; optional future synthetic augmentation | Local RGB/2D release; metadata disables LiDAR/DVS; generator export work separate | Required rich FSM not available in inspected active generator | Not the current foundation; retain provenance and prior findings |
| [ECP2.0](ecp2.md) | Possible later application; no internship dependency | Dense person trajectories and metric locations paper-reported; released fields to verify | No matching rich behaviour GT established | Access, suitable 3D/context fields, independent audit |
| [EMT](emt.md) | Background; no current experiment selected | Dash-camera tracking paper-reported; other modalities unknown | Similar action terminology, compatibility unverified | Reinspect only if deliberately selected later |
| [PIE / JAAD](pie-jaad.md) | Literature/background; not selected sources | Video/pedestrian annotations reported | Crossing/intention conventions require native interpretation | No current experiment depends on them |
| [ZOD-IAC](zod-iac.md) | Historical motivation and prior work | Prior detection/lifting/tracking pipeline | Geometry/review-derived crossing records | Track construction is outside the current task |

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

## Shared input conventions

| Concept | Required clarification before an adapter is accepted |
|---|---|
| Dataset, sequence, pedestrian identity | Stable native IDs, identity continuity, and validated cross-release associations. Keep association provenance; a ROAD tube ID is not automatically a Waymo object ID. |
| Time and annotation cadence | Sensor timestamps, annotation times, frame indexing, gaps, and synchronization tolerance. Do not equate annotation cadence with capture rate. |
| 3D boxes and trajectory | Coordinate frame, origin, axes, units, dimensions, rotation convention, and ego/world transforms. Derive metric velocity only after verifying these. |
| RGB and 2D observations | Camera identity, box convention, calibration, crop/context policy, and per-observation availability. Missing boxes do not alone establish a particular visibility condition. |
| Point clouds | Sensor configuration, point fields, timing, accumulation, ego compensation, and pedestrian/local-scene correspondence. |
| Ego/map/scene context | Coordinate transforms, semantic definitions, coverage, and whether context is provided or derived. |
| Availability and quality | Distinguish missing modality, occlusion, sparse returns, missing annotation, and uncertain association. |
| Supervision | Preserve native labels, definitions, annotation coverage, and masks independently from model inputs. |

This table specifies inspection obligations, not a software schema or adapter framework.

## Reader and prepared-track schema

Steps 1–2 use a common native-reader contract and [shared preparation](../../src/pedestrian_behavior/preparation.py). Readers own native IDs, associations, units, clocks and full native-to-world transforms. The creator owns scene-frame selection, fixed canonical coordinates and adjacent-slot velocities; [E001](../../experiments/E001-kinematic-transfer/README.md#first-baseline-temporal-representation) owns their scientific rules.

| Reader operation | Result |
|---|---|
| `index_scene(scene_id)` | Ordered scene frame keys/times/RGB availability; all candidate identities, native union extents/counts and scene provenance |
| `read_frames(scene_id, frame_ids)` | Exactly requested frames: full ego poses, world pedestrian positions, native semantic dictionaries and independent availability/source flags |
| `resolve_track(locator)` | Decode and validate the dataset-specific native track address against the source inventory |

Track locators are compact, sorted-key JSON strings. Their native fields belong to [LOKI](loki.md#track-locators-and-native-payload) and [ROAD-Waymo](road-waymo.md#track-locators-and-native-payload). Frame keys plus the track locator recover same-frame native observations/geometry; decoding a locator does not establish immutable content or a correct association. No inferred identity links or component-reference registry.

### Collection manifest

One **JSON manifest per dataset** stores schema version 1, dataset/source identity, preparation ID, release/checksum evidence or explicit unknowns, vocabularies, clock qualification, effective preparation rules, code/environment references and scene provenance. Preparation is marked incomplete until every indexed candidate is saved and reloaded successfully. Failures remain in its audit; a nonempty destination is never overwritten.

One **NumPy archive per track**, read with `allow_pickle=False`, stores numeric arrays and scalar JSON metadata. Global information is not repeated in every track. Generated collections/audits/inspection packs live under ignored `outputs/experiments/E001/reader-preparation/`.

### Prepared track

| Field | Meaning |
|---|---|
| `track_locator`, `native_extent_us` | Reversible native address and first-to-last 2D/3D union bounds; no outside padding or cross-scene stitching |
| `source_frames[T]` | Nullable selected native frame keys; LOKI retains leading-zero suffix strings |
| `source_time_us[T]` | Actual selected integer-us times, omitted when the frame key itself is the timestamp (Waymo) |
| `ped_position`, `ped_velocity`, `ego_position`, `ego_velocity` | Named unnormalized float64 `[T,2]` quantities in the fixed canonical frame |
| `ego_yaw[T]` | Float64 relative ego heading in radians |
| `ped_position_valid`, `ped_velocity_valid`, `ego_pose_valid`, `ego_velocity_valid` | Independent boolean `[T]` usability masks |
| `native_annotations[T]` | Native semantic dictionaries, preserving original label IDs/names, co-labels, missing/empty/null distinctions and annotation IDs/record indices; no common target projection |
| `availability[T]` | Source-specific 2D presence, native 3D presence and tri-state RGB availability; these are independent of numeric usability and supervision |
| `native_metadata[T]`, `issues[T]` | Original pairing/type/association flags where present; explicit unusability reasons |
| `anchor_valid`, `native_counts` | Initial-anchor qualification and pre-selection inventory counts for audit |

Requested grid times derive from the native start plus `j × 200000 µs`; no redundant grid array or saved batch padding. Invalid physical values are NaN; valid zero remains zero. Missing initial anchor never selects a later replacement: the candidate/native annotations remain, with unusable canonical arrays. Missing/nonfinite measurements and invalid/unavailable poses are masked and reported independently. Malformed schemas, conflicting duplicates/associations and unverified interpretation fail explicitly. Unused pedestrian yaw/dimensions cannot invalidate a usable center. No speed/jump/outlier filtering.

Keep **all indexed candidates**, including 2D-only, 3D-only, single-observation and unlabeled tracks. E001 later derives its eligible view, four-state target/mask, source-only normalization and [four float32 feature configurations](../../experiments/E001-kinematic-transfer/README.md#four-input-configurations) from the same saved arrays; a future head can interpret the same native supervision differently. Model targets/eligibility/splits/normalization belong to the experiment, not the common collection. RGB, point clouds and original boxes stay in the native sources for later retrieval; the numeric cache independently supports E001's declared condition slices.

The [acceptance checks](../../tests/test_track_preparation.py) exercise native-format readers through save/reload, independent known-motion answers, missingness, selection boundaries, semantics, identities and persistence. The portable inspector reads its actual numeric values from saved/reloaded archives; source overlays never interpolate or substitute another frame. [Runnable commands](../../README.md#prepare-and-inspect-complete-tracks).

## Missing modalities and labels

Represent actual availability explicitly. Apply a task loss only where its native annotation exists. Do not fabricate behaviour targets, treat an unlabeled frame as negative, or fill annotation gaps as ground truth. Interpolation used for a gallery camera center is not a recovered pedestrian observation.

The [ROAD-Waymo note](road-waymo.md) owns its index masks, booleans, duplicate observations and split/geometry provenance.

A projected 3D box and an annotated 2D observation have different provenance. Any resampling or derived geometry must record its method and uncertainty. E001 owns experiment-specific missing-input behavior.

## Native semantics and evidence

Preserve native names/IDs, applicability, definitions, cadence, annotation population, missingness and overlapping labels in each note. Inspect representative timelines and transitions before mapping or loss design. Do not equate nuScenes standing, ROAD stop and LOKI Stopped by name; waiting may include inferred intention. Dataset-specific heads are a hypothesis, not an accepted universal ontology or `OTHER` class. A transfer projection needs definitions, examples, exclusions and a fixed evaluated population before target scores are seen. Derived segments, onset/end and waiting duration also need definitions.

Record source URL, acquisition date, release/version or commit, checksum where available, files, schema, units, IDs, missing and duplicate rows. Distinguish paper-reported, official-format-documented, locally observed and checkpoint-reported claims. Handoffs are project context, not independent verification.

LOKI and ROAD-Waymo native meanings belong in their notes; the accepted experiment-specific [four-state projection](../../experiments/E001-kinematic-transfer/README.md#accepted-four-state-projection) and baseline features belong in E001. Access/split safeguards belong in [experiments](../../experiments/README.md). Complete recorded future observations are allowed for offline labeling; behavior GT, future-derived prediction targets and simulator-private route/intention fields are supervision, never inference inputs.

## Outstanding dataset checks

These checks carry former Q12 (linkage), Q13 (native semantics), Q17 (sequence supervision), and Q2 (fair LOKI low-label population).

- ROAD-Waymo: varied visual associations (crowded, small, occluded and failed cases), semantic disagreements, acquisition/code provenance and a predeclared acceptance rule. Verify clip/segment release compatibility, FRONT timestamps, camera-to-LiDAR IDs and track continuity; filenames alone are insufficient. Measure matched/unmatched/ambiguous/excluded coverage. Failed robust linkage requires reconsidering supervision.
- LOKI: independent physical identities, action episodes/transitions, physical timing, transforms, map/context usability and scene-safe grouping. [LOKI](loki.md#track-extents-and-gaps-2026-10-05) and [ROAD-Waymo](road-waymo.md#track-extents-and-gaps-2026-10-05) own native annotation-track extents/gaps and capacity statistics; E001 now defines resampling/eligibility and requires final counts under that protocol. Verify distance and point sparsity before using them. Missing 2D boxes do not prove outside-FOV status; later visibility cohorts require evidence.
- Later candidates: audit native semantics and usable modalities before inclusion. Reconcile IDD-PeD's 4,916 release count with the paper's >5,000; recount nuScenes's reported 8,143 with explicit split/category scope. Population size also depends on scene diversity and identity fragmentation.

The [research definition](../research.md) owns priorities; [E001](../../experiments/E001-kinematic-transfer/README.md) owns protocol readiness. Native readers and galleries are inspection tools, not frozen training adapters. Do not rebuild the existing validated remote join or introduce a schema/adapter hierarchy before the contract is settled.
