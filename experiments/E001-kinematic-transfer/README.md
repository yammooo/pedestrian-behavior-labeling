# E001 — Kinematic transfer

Status: **Planned**. Created 2026-10-05 from the 2026-10-02 proposal. Updated 2026-10-05 for the RQ framing and agreed initial batching policy. No training, model implementation or results.

## Question and controls

Does full-track learned temporal context improve kinematic frame-state labeling within ROAD-Waymo and LOKI and in both transfer directions? This is the first RQ1 recoverability/temporal-context diagnostic and establishes the single-dataset end-task reference for RQ2 transfer comparisons. It uses a kinematic subset of the full study contract; it cannot determine the value of omitted modalities or settle the final architecture/contribution. Follow [shared evaluation rules](../README.md), [research questions](../../docs/research.md) and [dataset conventions](../../docs/datasets/README.md).

## First two-dataset diagnostic

After the protocol gate, evaluate both [baseline variants](#first-diagnostic-baseline) in this matrix.

| Train dataset | Evaluate dataset | Purpose |
|---|---|---|
| ROAD-Waymo | Held-out ROAD-Waymo | Within-dataset learnability and source validation. |
| LOKI | Held-out LOKI | Within-dataset learnability and source validation for the reverse direction. |
| ROAD-Waymo | LOKI | Transfer from camera-selected supervision toward a 3D-first population. |
| LOKI | ROAD-Waymo | Reverse transfer and possible asymmetry. |

Use the same coordinate/temporal convention and [projection](#tentative-first-baseline-projection) in every cell. Training controls are specified below.

Begin with ROAD behaviour-labeled tracks that have official 3D pairs. Required paired coverage and the initial LOKI cohort remain open: a visibility-comparable subset is reasonable for a clean diagnostic but cannot establish performance on the full 3D-only population. Predeclare cohort criteria and excluded denominators. Preserve whole tracks, missing observations and independent label masks rather than retaining only consecutive labeled/paired frames.

No RGB, raw LiDAR, scene encoders, factorized heads, extra datasets, distillation, modality dropout or domain adaptation enters this first comparison.

Poor within-dataset performance is compatible with limited kinematic evidence, annotation ambiguity or a data/model/protocol problem; it does not prove sensor insufficiency or intrinsic ambiguity. A BiLSTM gain supports the value of learned temporal context under the chosen features. Strong within-dataset performance with poor transfer suggests dataset/ontology/domain mismatch. Better LOKI → ROAD-Waymo transfer is consistent with a selection-asymmetry hypothesis, but different class frequencies, ontology, geography, sensors and scene structure also change; directional scores alone cannot establish causality. Analyze class-wise errors and RGB availability, verified range, track duration, occlusion and LiDAR sparsity even though RGB/clouds are not model inputs. Declare cohort definitions, thresholds, exclusions and denominators before scoring; unavailable condition metadata remains unknown. Strong transfer still warrants difficult-subset and Stopped/Waiting analysis.

## First diagnostic baseline

### Minimal inputs and separate encoders

Use common pedestrian kinematics `k_t` and a separate ego vector `e_t`. Candidate pedestrian features are position `[x,y,z]` and velocity `[vx,vy,vz]`; candidate ego features describe ego velocity/motion. The exact features and coordinate convention remain open. Verify a common convention and ego compensation before computing motion; dataset-native position changes alone are not comparable pedestrian velocities.

Both encoders are small MLPs: `h_ped = E_ped(k_t)` and `h_ego = E_ego(e_t)`. Concatenate their outputs as `u_t = [h_ped; h_ego]`. Yaw, box dimensions, accelerations and relative velocity are later controlled ablations, not required first inputs.

### Missing observations and whole tracks

When the relevant pedestrian 3D observation is entirely missing, replace its encoded output with a learned vector `m_ped` of the same width. This vector is inserted **after** the pedestrian encoder; it is not an invented raw position or velocity. Valid ego observations continue through the ego encoder. A separate binary availability feature is probably unnecessary for this branch but remains an open choice. Partial feature availability, especially unavailable velocity despite a valid box, and missing ego handling still need definition.

Use whole variable-length tracks on a regular temporal grid, with **5 Hz a candidate**, not a frozen rate. Keep internal same-identity gaps as missing positions; do not compress valid detections together or split solely because of an occlusion. Padding, missing input and missing GT are separate conditions; [batch construction](#batch-construction) owns padding and sampling.

Predict a state at every grid position, with supervised loss only where applicable behaviour GT exists: `sum(label_mask * loss) / sum(label_mask)` over non-padding positions. A batch with no usable GT needs an explicit skip policy. Do not aggressively filter short tracks; determine minimal validity for the chosen features and later analyze performance by duration. [Temporal data details](#first-baseline-temporal-representation) remain open.

### Model A versus Model B

```text
Model A: Ped MLP --\
                   +--> concatenate --> classifier --> framewise state
         Ego MLP --/

Model B: Ped MLP --\
                   +--> concatenate --> whole-track BiLSTM --> classifier
         Ego MLP --/                                      --> framewise states
```

Use the same pedestrian encoder, ego encoder, input features and simple classifier design in both models. Only Model B adds bidirectional temporal processing. Choose compatible latent widths so this comparison does not require a more expressive head for one model; widths, capacity and training settings remain to be specified. The two models have independently trained parameters.

The comparison tests per-timestep kinematic classification versus full-track learned temporal context. Velocity derivation can itself use multiple observations; use the same derivation in both models and disclose its temporal support rather than calling Model A strictly single-observation inference.

## First-baseline temporal representation

Track start/end, grid alignment, resampling tolerance and GT assignment at grid times remain open. Retain elapsed time through gaps; resampling must not invent behavior GT or conceal missing evidence.

Keep `has_3d`, `has_2d`, `has_rgb` and `has_behavior`, provenance and quality independently of the learned missing embedding. An image does not imply a usable crop or 2D observation. ROAD-Waymo's `has_3d_box` masks native paired geometry; four-state supervision separately requires an accepted projection. Retain internal unpaired positions and population metadata for later 2D-visible/3D-only evaluation.

Verify feature units, axes, timestamps and ego/world transforms in both datasets. Ego-relative displacement is not pedestrian world velocity. Declare derivative support near gaps/endpoints and normalize only from permitted source training data. Partial-feature validity, missing ego and minimum usable observations need definition before implementation.

## Batch construction

Agreed initial policy (2026-10-05): **random batches with dynamic padding** for both baseline variants. Shuffle eligible whole tracks each training epoch and form minibatches without length bucketing. Each track remains a separate sample; pad only to the longest temporal extent in that minibatch. Retain internal gaps as sequence positions. Do not crop tracks, concatenate different tracks into one temporal sequence, or pad every batch to a fixed 128-position length.

Padding must be excluded from recurrent processing as well as supervised loss; masking the loss alone is insufficient for the backward BiLSTM. Recurrent state must not carry between independent tracks. Internal missing-input positions remain inside each track's temporal context, with supervision controlled by the separate label mask defined above.

Sampler seed, tracks per batch and incomplete-final-batch handling remain TBD. Record these settings and actual track/labeled-frame exposure for each run. Length bucketing or sequence packing can be reconsidered after measuring padding overhead; any change must preserve the declared sampling and loss-weighting policy. Future transformer token layout and batching are not settled by this choice. Native [LOKI](../../docs/datasets/loki.md#track-extents-and-gaps-2026-10-05) and [ROAD-Waymo](../../docs/datasets/road-waymo.md#track-extents-and-gaps-2026-10-05) extent statistics provide sizing evidence.

## Tentative first-baseline projection

The 2026-10-02 baseline proposal uses LOKI's four states for this kinematic diagnostic. This is an experiment-specific mapping to audit manually, not an accepted equivalence of native ontologies or a replacement for later dataset-specific heads.

| Concept in handoff | Observed ROAD-Waymo action strings to inspect | Candidate four-state output |
|---|---|---|
| Move | `Mov`, `MovAway`, `MovTow` | Moving (`MOVING`) |
| Stop | `Stop` | Stopped (`STOPPED`) |
| Wait-to-cross | `Wait2X` | Waiting to cross (`WAITING_TO_CROSS`) |
| Crossing variants | `Xing`, `XingFmLft`, `XingFmRht` | Crossing the road (`CROSSING`) |

Direction variants are candidate collapses, not yet validated. `PushObj` and any other action without a defensible mapping must not be forcibly assigned. ROAD actions can overlap: define treatment of unmapped co-labels and conflicts between mapped states before applying a single-state loss. No arbitrary class priority is accepted. Keep source-native action lists and mapping provenance; report excluded frames/tracks and class support.

Inspect definitions, representative timelines and transition boundaries, especially Stop versus Wait2X and movement versus crossing. A missing or ambiguous projected label has no four-state supervision, even if a native action or a 3D box exists. Use the same accepted projection in within-dataset and cross-dataset comparisons; score LOKI's native current-frame actions without shifting them to future targets.

## Unresolved settings and readiness

These settings carry former Q14 (comparable inputs), Q8 (temporal/sampling protocol), Q19 (initial population), Q20 (model controls) and the kinematics/observability test in Q5.

All applicable gates must pass before implementation/training: varied ROAD-Waymo visual association acceptance and reproducible mapping provenance; native semantic audit; LOKI population/cohort and scene grouping; verified common timing/coordinates/ego motion; then protocol freeze. Other datasets are not prerequisites for E001.

| Setting | Current value / needed evidence |
|---|---|
| Releases, association revision, acquisition/checksums | Native notes record acquired artifacts; complete provenance TBD |
| Semantic projection, co-label/conflict exclusions | Tentative above; manual definitions/timelines/boundaries audit TBD |
| ROAD paired coverage; initial LOKI cohort | TBD; a visible-only cohort cannot test the full 3D-only population |
| Grid, track extent, alignment and GT tolerance | TBD; 5 Hz is a candidate |
| Feature/velocity semantics; partial validity; missing ego | TBD; verify units/axes/transforms and derivative support |
| Splits, inspected-scene treatment, manifests | TBD; scene/identity overlap audit required |
| Training batch construction | Agreed: random whole-track batches with dynamic padding; [details](#batch-construction). Sampler seed and final-batch handling TBD |
| Widths/capacity, optimizer, training schedule, batch size | TBD; same encoders/head design, added recurrent capacity reported |
| Primary/secondary metrics, aggregation/temporal thresholds | TBD; shared candidate metrics are not accepted settings |
| Observation-condition strata and metadata validity | TBD; RGB availability, range, duration, occlusion and LiDAR sparsity |
| Source-only model selection, seeds/repeated runs | TBD; freeze before cross-dataset scores |
| Empty-GT skip policy, minimum usable observation | TBD; distinguish padding, missing input and missing label |
| Label budgets | No low-shot E001 comparison specified; any extension must declare units/counts/access |

## Variants and runs

Each trained variant may evaluate within its source and on the other dataset: eight evaluation cells, not necessarily eight training runs. Add one row per actual seed/run or failed attempt; retain negative results. No ablation or seed has been selected.

| Variant | Training source | Evaluations | Ablation / seed | Status | Failure / result |
|---|---|---|---|---|---|
| A: framewise MLP | ROAD-Waymo | ROAD-Waymo; LOKI | Base / TBD | Planned | Not run |
| B: whole-track BiLSTM | ROAD-Waymo | ROAD-Waymo; LOKI | Temporal context / TBD | Planned | Not run |
| A: framewise MLP | LOKI | LOKI; ROAD-Waymo | Base / TBD | Planned | Not run |
| B: whole-track BiLSTM | LOKI | LOKI; ROAD-Waymo | Temporal context / TBD | Planned | Not run |

## Run provenance

- Comparison-specific effective configs: future files beside this record; none yet.
- Commands, code revision/uncommitted diff, dependencies and environment versions: TBD. Project environment: `pedestrian-behavior`, Python 3.11; hardware/host and start/end dates recorded when run.
- Split/subset/release IDs or checksums, selected checkpoint and source-validation evidence: TBD.
- Generated manifests, logs, checkpoints, predictions and plots: intended `outputs/experiments/E001/` (ignored), or documented external storage; no run artifacts yet.
- Record exposure, runtime/compute/parameters, cohort/class denominators, variability, association/data failures and protocol deviations for every run.

## Conclusions, limitations and next step

No measured conclusion. Domain, ontology, sensors, geography, scene structure and class frequencies confound transfer asymmetry. The first comparison does not isolate visibility selection or establish 3D-only coverage. Next: complete the readiness gates and predeclare the applicable settings. Later ablations require an observed gap.
