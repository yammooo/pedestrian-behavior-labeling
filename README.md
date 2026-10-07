# Pedestrian Behavior Labeling

Aalto research workspace for a general offline pedestrian-behavior labeler from existing tracks under the [multimodal study contract](docs/datasets/README.md). No labeling model yet. Start with the [research dashboard](docs/README.md); the first kinematic transfer comparison is [E001, Planned](experiments/E001-kinematic-transfer/README.md).

## Layout

- `docs/`: research definition, dataset/literature evidence, historical archive and log.
- `experiments/`: stable comparison records and future comparison-specific configs.
- `src/`: reusable native readers/inspection tools; `scripts/` and `notebooks/`: one-off inspection/exploration.
- `configs/`: shared configuration; `tests/`: checks.
- `data/`: local datasets; `outputs/`: ignored generated artifacts. Never commit either datasets or large outputs.

## Setup and validation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once on each machine, then run from the repository root. Setup targets Linux x86-64. uv uses managed Python **3.11.16** from [.python-version](.python-version), dependencies from [pyproject.toml](pyproject.toml) and their resolved versions/hashes in [uv.lock](uv.lock). Inspection requires `ffmpeg` on `PATH`.

Choose **one** backend. Each checkout gets an ignored `.venv`; uv installs this package in editable mode and downloads the pinned Python if necessary.

```bash
# ThinkPad: CPU
uv sync --locked --extra cpu
```

```bash
# RTX 4080 laptop: CUDA
uv sync --locked --extra cu118
```

PyTorch **2.7.1** provides both builds in the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v271). CUDA 11.8 is selected for the inspected 535.309.01 NVIDIA driver; newer drivers support older CUDA runtimes through [backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). The wheel supplies its runtime dependencies. The [explicit PyTorch indexes and conflicting extras](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies) prevent installing both backends together.

Validate locally; replace `cpu` with `cu118` on `aalto`. Include the selected extra on project commands. `--locked` rejects an outdated lockfile instead of changing it.

```bash
uv lock --check
uv pip check --python .venv/bin/python
uv run --locked --extra cpu python -m unittest discover -s tests
uv run --locked --extra cpu python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

## Git checkout on the RTX 4080 laptop

The remote checkout is `/home/user20/projects/pedestrian-behavior-labeling`, reached with SSH host `aalto`:

```bash
ssh aalto
cd ~/projects/pedestrian-behavior-labeling
git status --short
git pull --ff-only
uv sync --locked --extra cu118
git rev-parse HEAD
```

Develop and validate locally, commit reviewed changes including `uv.lock` and push them, then pull and sync the intended revision on `aalto`. Verify its commit before a run; record any uncommitted changes. Run the installed package from this checkout instead of copying individual code files. Dataset paths remain machine-specific, and generated outputs stay ignored. Historical gallery workspaces remain evidence, not the current code checkout.

## Inspect LOKI

```bash
uv run --locked --extra cpu python -m pedestrian_behavior.inspection summary --root data/loki_data
uv run --locked --extra cpu python -m pedestrian_behavior.inspection gallery --root data/loki_data \
  --action "Crossing the road" --offset 0 --limit 8 \
  --output outputs/inspection/loki/crossing-000
```

Use exact native action names. Change `--offset` and `--output` for subsequent batches; nonempty output directories are rejected. [LOKI notes](docs/datasets/loki.md#inspection-display) explain counts, BEV conventions and missing-data display.

## Inspect ROAD-Waymo

The shared environment supplies Pillow/NumPy/PyArrow. No Waymo SDK or TensorFlow is required. Run where original Waymo files are accessible.

```bash
uv run --locked --extra cu118 python -m pedestrian_behavior.inspection.road_waymo \
  --index /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --action Wait2X --limit 8 --offset 0 \
  --output outputs/inspection/road_waymo/waiting-000
```

Use exact actions (`Stop`, `Wait2X`, `Xing`, `XingFmLft`, etc.). Selection is deterministic with `--seed`; `--clip`/`--track` restrict it. `--waymo-root` relocates original component paths without editing the index. Nonempty output directories are rejected. [ROAD-Waymo notes](docs/datasets/road-waymo.md#rgb--lidar-bev-inspection-gallery-2026-10-02) explain timing, geometry and markers.

## Inspect track lengths and gaps

Annotation-only statistics retain internal gaps inside each pedestrian's first-to-last clip-scoped extent. Run the ROAD command from the Git checkout on `aalto`, where its index is accessible. Choose a fresh output directory; existing directories are rejected.

```bash
uv run --locked --extra cpu python scripts/track-statistics.py --self-check
uv run --locked --extra cpu python scripts/track-statistics.py --dataset loki --input data/loki_data \
  --output outputs/inspection/track-statistics/loki
uv run --locked --extra cu118 python scripts/track-statistics.py --dataset road-waymo \
  --input /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --output outputs/inspection/track-statistics/road-waymo
```

Each scan writes `tracks.csv` and `summary.json`, including whole-track duration, observed/missing frames, modality masks, overlapping action cohorts and conservative 5 Hz window capacities. [LOKI](docs/datasets/loki.md#track-extents-and-gaps-2026-10-05) and [ROAD-Waymo](docs/datasets/road-waymo.md#track-extents-and-gaps-2026-10-05) own interpretation and limitations.

## Prepare and inspect complete tracks

Steps 1–2 save every native candidate's complete 5 Hz extent, kinematics, masks and unchanged native supervision. Output directories must be fresh. The [schema](docs/datasets/README.md#reader-and-prepared-track-schema) and [E001 protocol](experiments/E001-kinematic-transfer/README.md) own interpretation. This does not train a model or project targets.

```bash
uv run --locked --extra cpu python scripts/check-reader-preparation.py \
  --output outputs/experiments/E001/reader-preparation/acceptance
uv run --locked --extra cpu python scripts/verify-loki-transform.py \
  --root data/loki_data --output outputs/experiments/E001/reader-preparation/loki-transform-evidence.json
uv run --locked --extra cpu python -m pedestrian_behavior.preparation --dataset loki \
  --input data/loki_data --loki-transform configs/loki-transform.json \
  --output outputs/experiments/E001/reader-preparation/loki
```

Run ROAD-Waymo on `aalto`, from its Git checkout:

```bash
uv run --locked --extra cu118 python -m pedestrian_behavior.preparation --dataset road-waymo \
  --input /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --output outputs/experiments/E001/reader-preparation/road-waymo
```

Each collection has `manifest.json`, `audit.json` and `tracks/*.npz`; `status=complete` requires matching native/saved candidate inventories. Source roots are unnecessary to reload numeric tracks. `--scene` may repeat for an explicitly restricted smoke collection.

Inspect the frozen eight native cases per dataset; ROAD-Waymo uses `--extra cu118` and its index path on `aalto`. Original sources are needed for RGB/LiDAR previews.

```bash
uv run --locked --extra cpu python -m pedestrian_behavior.inspection.prepared_tracks \
  --collection outputs/experiments/E001/reader-preparation/loki \
  --cases outputs/experiments/E001/reader-preparation/loki-cases.json \
  --input data/loki_data --loki-transform configs/loki-transform.json \
  --output outputs/experiments/E001/reader-preparation/loki-inspector
```

Open `acceptance/index.html` for independent expected/actual fixture checks or the dataset inspector's `index.html` for synchronized native context, fixed trajectories, feature timelines, masks and annotation values. Copy the complete inspector directory for portable viewing. Missing slots stay blank; the viewer does not interpolate.
