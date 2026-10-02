# Conceptual pipeline

Status: Working hypotheses; no model architecture selected

Last updated: 2026-10-02

## Purpose and ownership

- Contains: conceptual processing flow and candidate modeling choices.
- Links out: fields to [DATASET_CONTRACT.md](DATASET_CONTRACT.md), semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), comparisons to [EVALUATION_PLAN.md](EVALUATION_PLAN.md), and optional techniques to [IDEAS_BACKLOG.md](IDEAS_BACKLOG.md).

## Baseline progression

Verify ROAD-Waymo linkage and native semantics first. Establish LOKI trajectory-only diagnostics, a trajectory-plus-scene diagnostic if feasible, a ROAD-Waymo source baseline, and basic source-to-target transfer before extending the method.

The baseline source is ROAD-Waymo alone. The next source may be nuScenes or ROAD; inclusion and order remain undecided. IDD-PeD is a later option. Source additions must address a named gap and improve a controlled comparison, or be reported as negative results.

## Candidate learned flow

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

## Candidate multi-source procedure

If source additions are justified, ordinary joint training is the preferred starting hypothesis: dataset-balanced sampling, losses masked by native annotation availability, and explicit missing-modality support. Loss weights and balancing policies remain open. Avoid assuming that an arbitrary sequential source curriculum will retain earlier knowledge.

Paired RGB/3D observations may support modality dropout, consistency, or distillation into a 3D pathway. Raw embeddings need not be identical because modalities contain complementary evidence. Compare 3D-only performance before adding such objectives.

Measure negative transfer through controlled source/task ablations. Gradient diagnostics and conflict-handling methods belong in the backlog until ordinary joint training exhibits a problem. No fusion or optimization method is fixed.

## Implementation boundaries

The existing LOKI reader and RGB/BEV gallery remain native inspection tools. Keep native annotation meanings and coordinates intact. ROAD-Waymo association/export utilities already exist in the external `aalto` mapping workspace; the [dataset note](DATASETS/ROAD_WAYMO.md#acquired-index-and-access-2026-10-02) records their paths. Preserve reproducible code before adding a small native reader; do not rebuild the validated join.

Later adapters may produce comparable observations with provenance and masks. Do not build a model, rigid shared schema, adapter hierarchy, or viewer framework during this documentation migration. [Experiment records](EXPERIMENTS/README.md) will tie any later implementation to its hypothesis, configuration, and observed result.
