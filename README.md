# Pedestrian Behavior Labeling

Research workspace for **label-efficient offline pedestrian-behavior annotation** in autonomous-driving datasets. The current question is whether fine-grained synthetic supervision reduces the number of real pedestrian tracks that must be labeled to learn LOKI's behavior states. PedSynth++ is a candidate source, not a prerequisite for the project.

The project begins with dataset and literature comparison. There is intentionally no labeling model yet.

Start with the [research documentation guide](docs/README.md).

## Layout

- `docs/` — canonical research documents, dataset/literature notes, decisions, and log.
- `src/` — future reusable package; empty by design until data inspection is complete.
- `data/` — local datasets only; never commit datasets.
- `notebooks/`, `scripts/`, `configs/`, `tests/` — reserved for later, evidence-driven work.

## Status

Initialized 2026-09-22. The prior ZOD-IAC work is historical context, not this repository's implementation base.
