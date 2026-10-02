# Pedestrian Behavior Labeling

Research workspace for **transferable offline pedestrian-behaviour annotation** in autonomous-driving datasets. We start from existing pedestrian tracks and investigate how to produce reliable dense labels in a new 3D-first domain with minimal target-specific supervision.

ROAD-Waymo is the baseline source candidate, subject to verifying its connection to original Waymo 3D tracks. Heterogeneous real supervision is an approach to test against that baseline; adding nuScenes or ROAD, and later possibly IDD-PeD, remains conditional.

The intended artifact is a working offline research labeler supported by reproducible experiments and traceable results. There is no labeling model yet. Start with the [research documentation guide](docs/README.md).

## Layout

- `docs/` — current research understanding, dataset/literature evidence, decisions, experiment records, and log.
- `src/` — native dataset readers and inspection tools.
- `data/` — local datasets only; never commit datasets.
- `notebooks/`, `scripts/`, `configs/`, `tests/` — exploration, inspection scripts, versioned experiment configurations, and checks.

## Status

Checkpoint: 2026-10-01. The prior ZOD-IAC and synthetic-pretraining work is preserved as research history. Architecture, source additions, metrics, and label-budget units remain open. The traineeship ends 2026-12-17.

## Inspect LOKI

Create the project Conda environment from the repository root (with `ffmpeg` on `PATH`):

```bash
conda create -n pedestrian-behavior python=3.11 pillow pip
conda activate pedestrian-behavior
python -m pip install --no-deps -e .
```

See the [inspection commands](src/pedestrian_behavior/inspection/README.md) for LOKI summaries and visual galleries.
