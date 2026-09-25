# Label ontology

Status: Draft  
Last updated: 2026-09-24

## Purpose and ownership

- Contains: target-state meanings, source labels, candidate mappings, ambiguous cases, and evidence needed to accept a mapping.
- Links out: raw release fields and counts to `DATASETS/`, and training-objective comparisons to [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## LOKI target actions

The [LOKI paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Girase_LOKI_Long_Term_and_Key_Intentions_for_Trajectory_Prediction_ICCV_2021_paper.pdf) reports four frame-wise pedestrian actions. Its prediction experiments turn a later action into an intention target; this project needs the original current-frame actions. Exact release values, annotation boundaries, and class counts still need inspection.

| Candidate state | Intended meaning | Known source terminology |
|---|---|---|
| `MOVING` | Pedestrian is moving, without asserting road crossing. | LOKI: Moving; EMT: Walking (compatibility unverified) |
| `STOPPED` | Pedestrian is not moving. | LOKI: Stopped; EMT: Stopping (compatibility unverified) |
| `WAITING_TO_CROSS` | Pedestrian waits in a crossing-relevant situation. | LOKI/EMT reported label; observability and semantics open |
| `CROSSING` | Pedestrian crosses the road. | LOKI/EMT reported label |

The proposed canonical names mirror these reported actions, but operational definitions remain open. In particular, `WAITING_TO_CROSS` may depend on context, future outcome, or annotation convention rather than motion alone.

## PedSynth++ source ontology (paper-reported; mapping unaccepted)

The [ARCANE-PedSynth paper, Table 4](https://arxiv.org/pdf/2605.24950) lists `WALKING_SIDEWALK`, `LOOKING_AROUND`, `CHECKING_TRAFFIC`, `HESITATING`, `CROSSING_ROAD`, `SUDDEN_CROSSING`, `JAYWALKING`, `RUNNING_ACROSS`, `PAUSING_MID_CROSS`, `DISTRACTED_BEHAVIOR`, `FINISHED_CROSSING`, and `RETREAT`. The post-checkpoint review of the [public generator repository](https://github.com/wielgosz-info/carla-pedestrians) instead found `NORMAL_CROSSING` in its 12-value enum and no `RETREAT` enum member, while retreat logic exists elsewhere. This code observation must be rechecked against the precise tagged version and released CSV. **Neither list is yet verified as the set of released annotation values.**

| PedSynth++ raw state | Candidate LOKI-style state | Status / risk |
|---|---|---|
| `WALKING_SIDEWALK` | `MOVING` | Strong candidate; verify movement in released examples. |
| `CHECKING_TRAFFIC`, `HESITATING` | `WAITING_TO_CROSS` | Strong candidates, but compare onset and episode semantics. |
| `LOOKING_AROUND` | Conditional or exclude | Looking may accompany moving, stopping, or waiting; later crossing may clarify an offline episode. |
| `CROSSING_ROAD`, `JAYWALKING`, `RUNNING_ACROSS` | `CROSSING` | Strong candidates while on the road. |
| `SUDDEN_CROSSING` | Conditional `MOVING`/`CROSSING` | Road entry may occur after the FSM state begins. |
| `PAUSING_MID_CROSS` | `CROSSING` | An active crossing can have near-zero speed. |
| `DISTRACTED_BEHAVIOR` | Conditional or exclude | Distraction is orthogonal to movement/crossing. |
| `FINISHED_CROSSING` | Conditional `MOVING`/`STOPPED` | Depends on post-crossing motion. |
| `RETREAT` (paper), `NORMAL_CROSSING` (code) | Unresolved | Confirm whether either appears in the released labels before mapping. |
| No obvious source state | `STOPPED` | Generic non-crossing stationary pedestrians may be missing. |

This table is an investigation plan, not a training-label mapping. A pure 12→4 dictionary is not defensible for all states. Some assignments need motion, road occupancy, and temporal context. Ambiguous frames may need exclusion or `UNKNOWN`/`OTHER`; none of these policies is decided. The `STOPPED` coverage gap is a specific early suitability test for PedSynth++.

## Implication for training targets

Native PedSynth++ states can supervise pretraining without forcing every source frame into four LOKI classes. A direct zero-shot four-state evaluation requires a defensible source-to-target projection; ambiguous source frames may need exclusion. The [evaluation plan](EVALUATION_PLAN.md) owns the proposed comparison with binary and collapsed-state pretraining.

## Candidate derived labels

- Binary crossing/non-crossing.
- Crossing onset and end.
- Waiting duration.
- Transition sequence.
- Future-crossing labels, only when deliberately defined as prediction targets.

The source-label discrepancy and mapping decision are tracked in [open questions Q1 and Q3](OPEN_QUESTIONS.md). The [inspection plan](DATASET_INSPECTION_PLAN.md) defines the release checks needed before accepting a mapping. EMT's `Stopping` remains a separate possible external-test question.
