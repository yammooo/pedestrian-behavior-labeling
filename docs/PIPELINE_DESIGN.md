# Conceptual pipeline

Status: First diagnostic baseline proposed; final architecture open; no implementation authorized

Last updated: 2026-10-02

## Purpose and ownership

- Contains: conceptual processing flow and candidate modeling choices.
- Links out: fields to [DATASET_CONTRACT.md](DATASET_CONTRACT.md), semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), comparisons to [EVALUATION_PLAN.md](EVALUATION_PLAN.md), and optional techniques to [IDEAS_BACKLOG.md](IDEAS_BACKLOG.md).

## Baseline progression

Verify ROAD-Waymo linkage, ROAD-Waymo/LOKI semantics, coordinates and timing first. The latest handoff proposes a kinematics-only diagnostic in both datasets and both transfer directions. [EVALUATION_PLAN.md](EVALUATION_PLAN.md#first-two-dataset-diagnostic) owns those comparisons; no model is to be implemented yet.

Scene information, RGB, raw LiDAR, factorized heads, extra datasets, distillation, modality dropout and domain adaptation are deferred until this baseline identifies a gap. Source additions must address a named gap and improve a controlled comparison, or be reported as negative results.

## First diagnostic baseline

### Minimal inputs and separate encoders

Use common pedestrian kinematics `k_t` and a separate ego vector `e_t`. Candidate pedestrian features are position `[x,y,z]` and velocity `[vx,vy,vz]`; candidate ego features describe ego velocity/motion. The exact features and coordinate convention remain open. Verify a common convention and ego compensation before computing motion; dataset-native position changes alone are not comparable pedestrian velocities.

Both encoders are small MLPs: `h_ped = E_ped(k_t)` and `h_ego = E_ego(e_t)`. Concatenate their outputs as `u_t = [h_ped; h_ego]`. Yaw, box dimensions, accelerations and relative velocity are later controlled ablations, not required first inputs.

### Missing observations and whole tracks

When the relevant pedestrian 3D observation is entirely missing, replace its encoded output with a learned vector `m_ped` of the same width. This vector is inserted **after** the pedestrian encoder; it is not an invented raw position or velocity. Valid ego observations continue through the ego encoder. A separate binary availability feature is probably unnecessary for this branch but remains an open choice. Partial feature availability, especially unavailable velocity despite a valid box, and missing ego handling still need definition.

Use whole variable-length tracks on a regular temporal grid, with **5 Hz a candidate**, not a frozen rate. Keep internal same-identity gaps as missing positions; do not compress valid detections together or split solely because of an occlusion. Padding to the longest sequence in a minibatch must be excluded from temporal processing and loss. Padding, missing input and missing GT are separate conditions.

Predict a state at every grid position, with supervised loss only where applicable behaviour GT exists: `sum(label_mask * loss) / sum(label_mask)` over non-padding positions. A batch with no usable GT needs an explicit skip policy. Do not aggressively filter short tracks; determine minimal validity for the chosen features and later analyze performance by duration. [DATASET_CONTRACT.md](DATASET_CONTRACT.md#first-baseline-temporal-representation) owns the unresolved data details.

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

## Later candidate learned flow

```text
existing pedestrian track + available observations
  -> native readers and verified associations
  -> candidate visual | 3D/scene | kinematic representations
  -> fusion, if justified
  -> offline temporal representation
  -> native dataset supervision heads / target output head
  -> dense framewise behaviour labels
  -> later: uncertainty, selective acceptance, review/export
```

Candidate inputs include pedestrian crops with local context, pedestrian-centered point-cloud/BEV context, trajectory/bbox/orientation information, and ego/map context. Their necessity and availability are experimental questions. A temporal encoder may use past and future observations; bidirectional recurrent models, temporal convolutions, Transformers, or ASFormer-like models remain candidates.

A shared temporal representation with dataset-specific heads is a methodological hypothesis. It could allow source annotations to supervise their own concepts without asserting a universal taxonomy. The tentative `z_motion / z_scene / z_crossing` factorization is only a candidate interpretation, not a required latent structure.

## Later candidate multi-source procedure

If source additions are justified, ordinary joint training is the preferred starting hypothesis: dataset-balanced sampling, losses masked by native annotation availability, and explicit missing-modality support. Loss weights and balancing policies remain open. Avoid assuming that an arbitrary sequential source curriculum will retain earlier knowledge.

Paired RGB/3D observations may support modality dropout, consistency, or distillation into a 3D pathway. Raw embeddings need not be identical because modalities contain complementary evidence. Compare 3D-only performance before adding such objectives.

Measure negative transfer through controlled source/task ablations. Gradient diagnostics and conflict-handling methods belong in the backlog until ordinary joint training exhibits a problem. No fusion or optimization method is fixed.

## Implementation boundaries

The existing LOKI and ROAD-Waymo readers and RGB/BEV galleries remain native inspection tools, not training adapters. Keep native annotation meanings and coordinates intact. ROAD-Waymo association/export utilities already exist in the external `aalto` mapping workspace; the [dataset note](DATASETS/ROAD_WAYMO.md#acquired-index-and-access-2026-10-02) records their paths. Preserve their reproducible provenance before a training adapter depends on them; do not rebuild the validated join.

Later adapters may produce comparable observations with provenance and masks. Do not build a model, rigid shared schema, adapter hierarchy, or viewer framework during this documentation migration. [Experiment records](EXPERIMENTS/README.md) will tie any later implementation to its hypothesis, configuration, and observed result.
