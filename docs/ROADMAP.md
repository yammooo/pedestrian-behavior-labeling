# Roadmap

Status: Draft  
Last updated: 2026-09-25

## Purpose and ownership

- Contains: order of work and the next decision gate; update it when evidence changes priorities.
- Links out: inspection details to [DATASET_INSPECTION_PLAN.md](DATASET_INSPECTION_PLAN.md) and comparison protocols to [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

1. **Inspect LOKI first, then gate synthetic suitability:** verify LOKI's raw tracks, labels, timing, and linked sensor data; inspect actual PedSynth++ release labels, counts, transitions, track structure, and full-data access. Use [DATASET_INSPECTION_PLAN.md](DATASET_INSPECTION_PLAN.md).
2. **Resolve semantics:** check conditional PedSynth++→LOKI mappings, especially generic `STOPPED` and `WAITING_TO_CROSS`; keep ambiguous states out of direct projections unless justified.
3. **Define shared contract:** document equivalent, leakage-safe motion/pose/visual inputs and a split-safe target protocol.
4. **Establish real reference and source learnability:** train the same minimal representation on LOKI with ample labels (`R→R`) and on held-out PedSynth++ (`S→S`).
5. **Measure label efficiency:** compare scratch and synthetic-pretrained models at the same fixed LOKI track budgets; report zero-shot separately where a valid four-state projection exists.
6. **Diagnose representation:** add pose, then visual embeddings only if class-wise transfer errors support them.
7. **Extend only when justified:** road context, domain adaptation, confidence/audit, EMT external validation, or ECP2.0 enrichment follow demonstrated needs.

No dates are assigned because none have been agreed.
