# Pedestrian Behavior Labeling

Aalto research workspace for a general offline pedestrian-behavior labeler from existing tracks under the [multimodal study contract](docs/datasets/README.md). No labeling model yet. Start with the [research dashboard](docs/README.md); the first kinematic transfer comparison is [E001, Planned](experiments/E001-kinematic-transfer/README.md).

## Layout

- `docs/`: research definition, dataset/literature evidence, historical archive and log.
- `experiments/`: stable comparison records and future comparison-specific configs.
- `src/`: package code following the [agreed architecture](ARCHITECTURE.md); `scripts/` and `notebooks/`: one-off inspection/exploration.
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

W&B **0.30.0** is included in both environments for experiment tracking. [E001's run provenance](experiments/E001-kinematic-transfer/README.md#run-provenance) records the destination and user-reported login on both laptops; training integration remains unimplemented.

Validate locally; replace `cpu` with `cu118` on `aalto`. Include the selected extra on project commands. `--locked` rejects an outdated lockfile instead of changing it.

```bash
uv lock --check
uv pip check --python .venv/bin/python
uv run --locked --extra cpu python -m unittest discover -s tests
uv run --locked --extra cpu python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
uv run --locked --extra cpu wandb --version
```

To authenticate, run `uv run --locked --extra cpu wandb login --verify` locally. On `aalto`, run `/home/user20/.local/bin/uv run --locked --extra cu118 wandb login --verify` from its repository root. Paste your W&B API key into the interactive prompt; credentials are stored outside Git. Login does not configure training logging.

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
uv run --locked --extra cpu python -m pedestrian_behavior.data.preparation --dataset loki \
  --input data/loki_data --loki-transform configs/loki-transform.json \
  --output outputs/experiments/E001/reader-preparation/loki
```

Run ROAD-Waymo on `aalto`, from its Git checkout:

```bash
uv run --locked --extra cu118 python -m pedestrian_behavior.data.preparation --dataset road-waymo \
  --input /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --output outputs/experiments/E001/reader-preparation/road-waymo
```

Each collection has `manifest.json`, `audit.json` and `tracks/*.npz`; `status=complete` requires matching native/saved candidate inventories. Source roots are unnecessary to reload numeric tracks. `--scene` may repeat for an explicitly restricted smoke collection.

Inspect the frozen eight native cases per dataset; ROAD-Waymo uses `--extra cu118` and its index path on `aalto`. Original sources are needed for RGB/LiDAR previews.

```bash
uv run --locked --extra cpu python -m pedestrian_behavior.inspection.prepared_tracks \
  --collection outputs/experiments/E001/reader-preparation/loki \
  --cases experiments/E001-kinematic-transfer/reader-cases.json \
  --input data/loki_data --loki-transform configs/loki-transform.json \
  --output outputs/experiments/E001/reader-preparation/loki-inspector
```

Open `acceptance/index.html` for independent expected/actual fixture checks or the dataset inspector's `index.html` for synchronized native context, fixed trajectories, feature timelines, masks and annotation values. Copy the complete inspector directory for portable viewing. Missing slots stay blank; the viewer does not interpolate.

Reproduce the ROAD association/context audit and review pack on `aalto`, after pulling and verifying the intended Git revision. The output directory must be fresh; the full native index and saved collection are required. E001 owns the [acceptance record](experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08).

```bash
uv run --locked --extra cu118 python scripts/audit-road-associations.py \
  --index /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --collection outputs/experiments/E001/reader-preparation/road-waymo \
  --output outputs/experiments/E001/association-review/complete
```

`audit.json` freezes source hashes, native counts, selected identities/timestamps and command/revision; `index.html` shows same-frame RGB/BEV and original-resolution box crops. Structural success is not automatic visual acceptance. Native-only extensions have no ROAD GT; images with tiny boxes or occlusion may remain inconclusive.

## E001 saved tracks to batches

Audit native supervision once, then freeze the eligible population, clip/scenario splits and source-training normalization. Setup outputs must be fresh. Native files are required only for this audit; subsequent samples/batches use the existing saved archives and setup artifacts. [E001](experiments/E001-kinematic-transfer/README.md#saved-tracks-to-batches-2026-10-08) owns targets, settings and population evidence.

```bash
uv run --locked --extra cpu python scripts/check-e001-data.py \
  --output outputs/experiments/E001/data-setup/acceptance
uv run --locked --extra cpu python -m pedestrian_behavior.experiments.e001 \
  --collection outputs/experiments/E001/reader-preparation/loki \
  --native-input data/loki_data --output outputs/experiments/E001/data-setup/loki
```

On `aalto`, pull the intended Git revision and verify `git rev-parse HEAD` before running:

```bash
uv run --locked --extra cu118 python -m pedestrian_behavior.experiments.e001 \
  --collection outputs/experiments/E001/reader-preparation/road-waymo \
  --native-input /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --output outputs/experiments/E001/data-setup/road-waymo
```

Load batches without native sensor files (run through the same uv environment):

```python
import json
from pathlib import Path
from pedestrian_behavior.data.loading import track_batches
from pedestrian_behavior.experiments.e001 import dataset_from_setup

base = Path("outputs/experiments/E001")
source_setup = base / "data-setup/loki/setup.json"
statistics = json.loads(source_setup.read_text())["normalization"]["K+T+R"]
dataset = dataset_from_setup(base / "reader-preparation/loki", source_setup,
                             "training", "K+T+R", statistics)
batches = track_batches(dataset, training=True)
batch = next(iter(batches))
```

For transfer, pass the target collection/setup and split while retaining the source statistics. A sample returns CPU `inputs[T,D]` float32, `targets[T]` int64, `gt_valid[T]` bool, locator and archive reference. A batch adds original `lengths`, a mask true only for padding, and reference lists; padding is zero/−100/false. Internal missing slots stay in the sequence.

## E001 models

The [model protocol](experiments/E001-kinematic-transfer/README.md#first-diagnostic-baseline) defines A (framewise MLP) and B (one-layer BiLSTM), with the same encoder/linear-head structure and independent weights. Generate data/model expected/actual acceptance evidence in a fresh directory:

```bash
uv run --locked --extra cpu python scripts/check-e001-data.py \
  --output outputs/experiments/E001/models/acceptance-local
```

On `aalto`, pull/verify the intended revision, use `--extra cu118` and a fresh output path. CUDA acceptance checks run when a GPU is available; the local CPU report records that check as skipped. Open the generated `index.html`.

Run this after obtaining `batch` above, through the same uv environment:

```python
import torch
from pedestrian_behavior.experiments.e001 import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("K+T+R", "B").to(device).eval()  # "A" for framewise
with torch.no_grad():
    logits = model(batch["inputs"].to(device), batch["lengths"])  # lengths stay on CPU
predictions = logits.argmax(dim=-1)
# Only non-padding predictions are meaningful; score only accepted GT slots.
```

Logits have shape `[B,T,4]`; padded logits are zero, internal missing slots remain predictions. This is model inference/acceptance only; comparison training/evaluation/checkpoint commands are not implemented yet.
