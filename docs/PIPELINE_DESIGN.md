# Conceptual pipeline

Status: Working design  
Last updated: 2026-09-24

## Purpose and ownership

- Contains: current conceptual processing flow and modeling choices worth testing.
- Links out: fields to the [contract](DATASET_CONTRACT.md), states to the [ontology](LABEL_ONTOLOGY.md), comparisons to the [evaluation plan](EVALUATION_PLAN.md), and unselected techniques to the [ideas backlog](IDEAS_BACKLOG.md).

```text
PedSynth++ / LOKI inspection and adapters (if compatible)
  -> common temporal track sample
  -> optional representations: motion | pose | visual crop/context | road/scene
  -> one modular learned temporal labeler with a replaceable output head
  -> per-frame canonical state sequence
  -> derived segments/events and, later, confidence/export
```

The intended model set is deliberately small:

| Model | Purpose |
|---|---|
| Physical sanity baseline | Establish what velocity, trajectory, and optional road geometry solve without learning. |
| Shared structured learned model | Test motion alone, motion+pose, and motion+pose+visual through modality masks/ablations. |

The learned model's final state decision remains learned. Motion is the first candidate representation; pose and visual/context branches are conditional on measured failures. A small bidirectional temporal model is plausible because the annotation task permits future context, but no architecture is selected. Native PedSynth++ pretraining with a replaceable LOKI head is a working option; its value is tested in the [evaluation plan](EVALUATION_PLAN.md).

The model processes a sequence into a state sequence rather than independently classifying only a window center. Because annotation is offline, its temporal encoder may use past and future context.
