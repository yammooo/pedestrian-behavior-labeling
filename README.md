# Pedestrian Behavior Labeling

Aalto research workspace for a general offline pedestrian-behavior labeler from existing tracks under the [multimodal study contract](docs/datasets/README.md). No labeling model yet. Start with the [research dashboard](docs/README.md); the first kinematic transfer comparison is [E001, Planned](experiments/E001-kinematic-transfer/README.md).

## Layout

- `docs/`: research definition, dataset/literature evidence, historical archive and log.
- `experiments/`: stable comparison records and future comparison-specific configs.
- `src/`: reusable native readers/inspection tools; `scripts/` and `notebooks/`: one-off inspection/exploration.
- `configs/`: shared configuration; `tests/`: checks.
- `data/`: local datasets; `outputs/`: ignored generated artifacts. Never commit either datasets or large outputs.

## Setup and validation

Run from the repository root. Inspection requires `ffmpeg` on `PATH`.

```bash
conda create -n pedestrian-behavior python=3.11 pillow pip
conda activate pedestrian-behavior
python -m pip install --no-deps -e .
python -m unittest discover -s tests
```

## Inspect LOKI

```bash
python -m pedestrian_behavior.inspection summary --root data/loki_data
python -m pedestrian_behavior.inspection gallery --root data/loki_data \
  --action "Crossing the road" --offset 0 --limit 8 \
  --output outputs/inspection/loki/crossing-000
```

Use exact native action names. Change `--offset` and `--output` for subsequent batches; nonempty output directories are rejected. [LOKI notes](docs/datasets/loki.md#inspection-display) explain counts, BEV conventions and missing-data display.

## Inspect ROAD-Waymo

Install optional Pillow/NumPy/PyArrow dependencies in `pedestrian-behavior`; the inspected `aalto` environment already has them. No Waymo SDK or TensorFlow is required. Run where original Waymo files are accessible.

```bash
python -m pip install -e '.[waymo-inspection]'
python -m pedestrian_behavior.inspection.road_waymo \
  --index /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --action Wait2X --limit 8 --offset 0 \
  --output outputs/inspection/road_waymo/waiting-000
```

Use exact actions (`Stop`, `Wait2X`, `Xing`, `XingFmLft`, etc.). Selection is deterministic with `--seed`; `--clip`/`--track` restrict it. `--waymo-root` relocates original component paths without editing the index. Nonempty output directories are rejected. [ROAD-Waymo notes](docs/datasets/road-waymo.md#rgb--lidar-bev-inspection-gallery-2026-10-02) explain timing, geometry and markers.

## Inspect track lengths and gaps

Annotation-only statistics retain internal gaps inside each pedestrian's first-to-last clip-scoped extent. Run the ROAD command where its index is accessible, using a copy of the utility and the existing readers. Choose a fresh output directory; existing directories are rejected.

```bash
python scripts/track-statistics.py --self-check
python scripts/track-statistics.py --dataset loki --input data/loki_data \
  --output outputs/inspection/track-statistics/loki
python scripts/track-statistics.py --dataset road-waymo \
  --input /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --output outputs/inspection/track-statistics/road-waymo
```

Each scan writes `tracks.csv` and `summary.json`, including whole-track duration, observed/missing frames, modality masks, overlapping action cohorts and conservative 5 Hz window capacities. [LOKI](docs/datasets/loki.md#track-extents-and-gaps-2026-10-05) and [ROAD-Waymo](docs/datasets/road-waymo.md#track-extents-and-gaps-2026-10-05) own interpretation and limitations.
