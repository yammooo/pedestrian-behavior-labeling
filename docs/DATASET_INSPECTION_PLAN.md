# Dataset inspection plan

Status: Draft; execute against released annotations when available.  
Last updated: 2026-09-24

## Purpose and ownership

- Contains: checks and required outputs for the current PedSynth++/LOKI suitability gate.
- Links out: observed fields and counts to `DATASETS/`, mapping conclusions to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), and shared fields to [DATASET_CONTRACT.md](DATASET_CONTRACT.md).

## Gate

Decide whether PedSynth++ can serve as useful pretraining for label-efficient LOKI annotation. The paper and generator code do not establish the contents of the released labels. Full PedSynth++ data access must be confirmed; its [paper](https://arxiv.org/pdf/2605.24950) says the dataset is available from the corresponding author upon reasonable request. The [Zenodo demo subset](https://zenodo.org/records/20444839) may validate a parser but cannot establish full-dataset frequencies.

## Record provenance before counting

For each inspected source, record release/version or code commit, access date, annotation files, schema, units, time base, scene/track identifiers, and any missing or duplicated rows. Keep data and generated plots outside Git; commit only summarized findings and inspection scripts when the schema is known.

## PedSynth++ report

| Check | Output | Why |
|---|---|---|
| Behavior fields | Raw column names, unique values, missing-value counts, per-state frame counts | Resolve paper `RETREAT` versus code `NORMAL_CROSSING` against the released CSV. |
| Tracks | Number of unique `(clip, pedestrian ID)` tracks; class-bearing tracks per state | Estimate independent source examples; avoid counting frames as tracks. |
| Time | Timestamp/frame-step distribution, track durations, per-state contiguous episode durations | Verify dense labels and available temporal context. |
| Transitions | Raw-state transition count matrix within each track, excluding discontinuities | Identify actual FSM paths and rare/absent states. |
| Crossing | Crossing/non-crossing frame, episode, and track counts; relation to behavior state | Check whether crossing flags and FSM states agree. |
| Stationary cases | Generic non-crossing stops versus hesitating and mid-cross pauses, using motion and clips | Test the suspected `STOPPED` supervision gap. |
| Modalities | Per-track presence and fields for RGB, boxes/positions, pose, ego motion, calibration | Test the shared input contract without using simulator-private inputs. |

Inspect several short labeled timelines and trajectory/road overlays for ambiguous states (`LOOKING_AROUND`, `DISTRACTED_BEHAVIOR`, `FINISHED_CROSSING`, `NORMAL_CROSSING`/`RETREAT`) and waiting-like sequences. Choose examples from observed transitions, including rare ones; do not present them as representative frequencies.

## LOKI report

| Check | Output | Why |
|---|---|---|
| Action values | Exact raw names/codes, missing labels, per-class frame counts | Verify the four target actions in the released data. |
| Tracks and episodes | Unique tracks containing each class, unique tracks overall, contiguous episode counts/durations | Determine feasible track budgets and class coverage. |
| Time and transitions | Frame/timestamp steps, track lengths, four-state transition matrix | Define temporal context and leakage-safe splits. |
| Inputs | Per-track availability of 2D boxes, 3D positions, RGB, orientation, ego motion, pose, calibration | Find the actual intersection with PedSynth++. |
| Ambiguity | Clips of `STOPPED` versus `WAITING_TO_CROSS`, and action changes near road entry | Check annotation semantics and plausible source mapping. |

Track totals by class may overlap because one track can contain several states. The budget report must also count distinct selected tracks and scene/sequence coverage.

## EMT scope

Inspect official definitions and examples for `Stopping`, `Walking`, `Waiting to cross`, and `Crossing`; establish whether they are frame-wise actions or future targets, and whether an evaluation subset can share the chosen inputs. EMT is an optional external test and does not determine the initial input contract.

## Deliverables and decision

1. Add observed counts and precise provenance to the dataset notes and matrix; preserve `unknown` for inaccessible fields.
2. Update [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md) with a source→target mapping marked accepted, conditional, or excluded, and examples supporting each choice.
3. Update [DATASET_CONTRACT.md](DATASET_CONTRACT.md) from the actual field intersection and [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) with resolved/new questions.
4. Decide whether PedSynth++ has enough relevant states and common observations for a fair pretraining test. If not, document the reason and pursue the LOKI-first fallback in [RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md).

Write the smallest inspection scripts only after seeing sample files and their schema. They should print deterministic summary tables and, if useful, save a few behavior timelines/trajectory plots. No training code is needed for this gate.
