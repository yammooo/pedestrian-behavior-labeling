# Pedestrian Behavior Labeling

Research workspace for **offline pedestrian-behaviour labeling** in autonomous-driving datasets. We start from existing pedestrian tracks and investigate behavior supervision that remains useful without visual observations.

The first proposed diagnostic compares kinematics-only framewise MLP and whole-track BiLSTM baselines within ROAD-Waymo and LOKI and in both transfer directions. The four-state mapping, comparable features and initial subsets still need validation. Factorization, multimodal methods and extra datasets are later hypotheses.

The intended artifact is a working offline research labeler supported by reproducible experiments and traceable results. There is no labeling model yet. Start with the [research documentation guide](docs/README.md).

## Layout

- `docs/` — current research understanding, dataset/literature evidence, decisions, experiment records, and log.
- `src/` — native dataset readers and inspection tools.
- `data/` — local datasets only; never commit datasets.
- `notebooks/`, `scripts/`, `configs/`, `tests/` — exploration, inspection scripts, versioned experiment configurations, and checks.

## Status

Checkpoint: 2026-10-02. The latest handoff authorizes documentation updates only; no model implementation yet. The two research questions and final method remain provisional. Prior ZOD-IAC and synthetic work are preserved as history. The traineeship ends 2026-12-17.

## Inspect LOKI

Create the project Conda environment from the repository root (with `ffmpeg` on `PATH`):

```bash
conda create -n pedestrian-behavior python=3.11 pillow pip
conda activate pedestrian-behavior
python -m pip install --no-deps -e .
```

See the [inspection commands](src/pedestrian_behavior/inspection/README.md) for LOKI summaries and visual galleries.
