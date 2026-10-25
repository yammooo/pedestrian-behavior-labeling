# Pedestrian Behavior Labeling

Aalto research workspace for a general offline pedestrian-behavior labeler from existing tracks under the [multimodal study contract](docs/datasets/README.md). Kinematic models, training/evaluation and a brief single-seed comparison review are implemented. Start with the [research dashboard](docs/README.md); [E001](experiments/E001-kinematic-transfer/README.md) owns the comparison and repetition protocol.

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

W&B **0.30.0** is included in both environments for experiment tracking. [E001's run provenance](experiments/E001-kinematic-transfer/README.md#run-provenance) records the destination and user-reported login on both laptops; training logging is implemented; E001 records upload validation separately.

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

## Review completed E001 comparisons

The [joint E001/E001b/E001c review](experiments/E001b-2d-bbox-contribution/results-review.md) covers 130 five-seed attempts / 260 test cells. Rebuild its numerical tables and all ten PNG/SVG figure pairs from the retained metadata snapshot:

```bash
uv run --locked --extra cpu python scripts/review-e001-results.py \
  --output outputs/experiments/E001b/analysis/20261015-seed-review
```

This does not run inference or edit the Markdown conclusions. The script's `--collect --prior-verification <previous-audit.json> --output <fresh-directory>` mode runs on Aalto to snapshot the three named frozen sweeps and audit saved E001c logits against earlier identities/GT/conditions; full predictions/checkpoints stay there. The [review evidence](experiments/E001b-2d-bbox-contribution/results-review.md#e001c-extension-evidence-2026-10-15) records exact scope, source paths and reproducibility checks. Neither mode modifies original run artifacts or W&B.

## Inspect model predictions

Run the browser viewer on `aalto` from its isolated checkout. It reads completed E001/E001b/E001c test predictions, including runs that finish while it is open after **Refresh runs**. No inference, training or W&B connection is needed.

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-viewer
uv run --locked --extra cu118 python -m pedestrian_behavior.inspection.predictions \
  --loki-root /run/media/user20/F47C60057C5FC152/projects/loki_data \
  --road-index /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --waymo-root /run/media/user20/F47C60057C5FC152/projects/waymo_v2
```

From the ThinkPad, forward the loopback port and open [the viewer](http://127.0.0.1:8765):

```bash
ssh -F /home/yammo/.ssh/config -N -L 8765:127.0.0.1:8765 aalto
```

Choose the evaluation dataset and up to four runs (A–D); the run search matches experiment, training source, architecture, configuration, seed, budget and attempt name. The track table shows **one metric at a time**: accuracy, mean CE, supported-class macro-F1, or a selected GT→prediction confusion rate. Sort any model or its signed difference from A; accuracy/F1 differences use percentage points, CE differences use loss units. Smaller scores are worse for accuracy/F1; larger scores are worse for CE/confusion rate. These per-track diagnostics do not replace the experiment's four-class track-weighted primary score.

An optional observation-axis/value filter selects tracks with matching context. **Matching frames** scores only accepted GT in that condition; **Whole track** retains complete-track scores. The table always distinguishes scored, matching and total GT counts. Unscored slices show `—`; minimum scored GT defaults to zero. Specific-confusion views show numerator/support and retain tracks where at least one selected model makes that error. Only identically defined/matching condition memberships shared by selected predictions are offered; E001's annotated-2D flags are not a substitute for verified E001b/E001c boxes.

Click a track for synchronized RGB/crop/LiDAR BEV, GT and model timelines, probability values and a confusion matrix. Click a prediction row to choose the model for next/previous error navigation. Predictions on unlabeled context stay visible and unscored. Native boxes are yellow; the optional ROAD annotation overlay is cyan and never supplies missing Waymo input boxes or crops. LOKI source times are nominal; Waymo source times are physical. Missing observations are never interpolated. The trajectory view uses the fixed prepared coordinate frame separately from the native BEV.

Native previews prepare once per track in a background worker, then reuse a separate cache across models/seeds. The first native visit can take time; progress and loading-frame status remain explicit. Missing native storage leaves scores/timelines usable. **Save case** records the run/prediction hashes, track/time, browsing state, reason and note in `outputs/inspection/prediction-viewer/reviews.json`; changed predictions prevent saved-case reopening. Browser links retain the current selection. This is retrospective test inspection: disclose it in subsequent research decisions; do not use it to retrospectively select seeds/checkpoints.

`--outputs-root`, `--cache` and `--port` override local paths/port. Defaults are `outputs`, `outputs/inspection/prediction-viewer` and loopback 8765. The isolated remote checkout links its ignored `outputs` to the original machine-specific outputs; model results remain read-only. Locally use `--extra cpu` and available native roots. Neither model checkpoints nor prediction archives are copied into the viewer cache.

Acceptance (2026-10-15): **77 CPU tests, five CUDA skips; all 77 CUDA tests passed**. Fixture browser checks covered playback, four-model selection, scopes/confusions, saved notes and link restoration; real browser checks covered both datasets. The [repeatable native check](scripts/check-prediction-viewer.py) compared four runs across E001/E001b/E001c, reconstructed track accuracy and verified **366 unchanged source artifacts**. [Numerical evidence](outputs/inspection/prediction-viewer/acceptance/native-b297d5a/verification.json), [real browser checks](outputs/inspection/prediction-viewer/acceptance/real-browser.json), [CPU](outputs/inspection/prediction-viewer/acceptance/acceptance-cpu.log) / [CUDA logs](outputs/inspection/prediction-viewer/acceptance/acceptance-cuda.log) and [original-checkout verification](outputs/inspection/prediction-viewer/acceptance/checkout-verification.json) remain local/remote under ignored inspection outputs. Representative four-model index loading took 1.0–1.4 s, warm queries 0.18–0.25 s; first native preparation took 7.7 s for 42 LOKI frames and 44.2 s for 56 ROAD-Waymo frames, with cache reuse verified. These are example timings, not latency guarantees. Training checkouts/results were preserved; no new comparison training or inference was launched.

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

## E001b bbox extension

Prepare [E001b](experiments/E001b-2d-bbox-contribution/README.md) separately; this reads the frozen E001 collections/setup and writes only a fresh bbox cache/statistics directory. No training or W&B run is started. Locally:

```bash
uv run --locked --extra cpu python scripts/prepare-e001b.py \
  --collection outputs/experiments/E001/reader-preparation/loki \
  --setup outputs/experiments/E001/data-setup/loki/setup.json \
  --native-input data/loki_data --output outputs/experiments/E001b/bbox-native/loki
```

On `aalto`, use an isolated Git worktree and its own uv environment while E001 runs; do not pull/sync into the active training checkout. Pass absolute E001 collection/setup paths from `/home/user20/projects/pedestrian-behavior-labeling`, use `--extra cu118`, and set `--native-input` to the current mounted `loki_data` path (below) or `/home/user20/road_waymo_mapping/merged_pedestrians_20261002`. Write separate E001b outputs. The command requires a fresh directory, retains failure evidence and checks all original collection/setup file hashes before/after.

The dataset partition can mount at `/run/media/user20/F47C60057C5FC152/projects` instead of the older `/media/user20/...` path stored in frozen manifests. Check the actual mount before preparation. On 2026-10-09, native Waymo files were verified under `/run/media/...`. Use the explicit E001b `--waymo-root` override; it retains native checksum checks and does not rewrite E001 references. From the isolated E001b checkout on `aalto`:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e001b
/home/user20/.local/bin/uv run --locked --extra cu118 python scripts/prepare-e001b.py \
  --collection /home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001/reader-preparation/road-waymo \
  --setup /home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001/data-setup/road-waymo/setup.json \
  --native-input /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --waymo-root /run/media/user20/F47C60057C5FC152/projects/waymo_v2 \
  --output /home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001b/bbox-native/road-waymo
```

For LOKI native reads at this mount, pass `/run/media/user20/F47C60057C5FC152/projects/loki_data` as `--native-input`. Existing saved LOKI caches need no regeneration for a mount-path change. To restore an unmounted partition, run `udisksctl mount --block-device /dev/nvme0n1p3 --options ro` in a terminal on the laptop and use the mount location it reports. An `AlreadyMounted` response reports the current location; it does not mean the stored `/media/...` path exists.

For saved-cache sample construction, call `pedestrian_behavior.experiments.e001b.dataset_from_extension(collection, setup_path, extension, split, configuration, baseline_statistics, bbox_statistics)`, with configuration `availability` or `geometry`. Both statistics come from the **training source**, including for opposite-dataset evaluation. Read the extension's `baseline_normalization` and `normalization` fields. Samples use unchanged E001 targets/references, with 14 or 20 input columns. The [E001b training commands](#e001b-training-and-evaluation) reuse the shared attempt runner.

Use `outputs/experiments/E001b/bbox-native/{loki,road-waymo}/` for the accepted native-only policy: Waymo FRONT boxes without ROAD fallback, and native LOKI boxes. The original ROAD-preferred `bbox/` caches remain historical evidence and fail the current loader policy check. [Source decision and verification](experiments/E001b-2d-bbox-contribution/README.md#native-only-cache-acceptance-2026-10-09) own acceptance; both native-only caches are complete on both machines. ROAD preparation uses the verified `/run/media/...` mount override. Use the [E001b training commands](#e001b-training-and-evaluation) after acceptance checks.

## E001c bbox cache

Recompute native inputs while preserving E001/E001b. Run at the intended Git revision on `aalto`; outputs must be fresh. [Protocol/schema/acceptance](experiments/E001c-bbox-inputs/README.md).

```bash
uv run --locked --extra cu118 python scripts/prepare-e001c.py \
  --collection /home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001/reader-preparation/loki \
  --setup /home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001/data-setup/loki/setup.json \
  --native-input /run/media/user20/F47C60057C5FC152/projects/loki_data \
  --loki-transform configs/loki-transform.json \
  --output /home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001c/box-cache/loki

uv run --locked --extra cu118 python scripts/prepare-e001c.py \
  --collection /home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001/reader-preparation/road-waymo \
  --setup /home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001/data-setup/road-waymo/setup.json \
  --native-input /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --waymo-root /run/media/user20/F47C60057C5FC152/projects/waymo_v2 \
  --output /home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E001c/box-cache/road-waymo
```

These commands prepare caches only. The [native mount guidance](#e001b-bbox-extension) still applies. Local checks: `uv run --locked --extra cpu python -m unittest discover -s tests -p test_e001c.py`.

## E001c training and evaluation

Use the existing frozen E001 collections/setups and E001c `box-cache/{loki,road-waymo}/`; native storage is unnecessary. The [final protocol](experiments/E001c-bbox-inputs/README.md#inputs-and-bounded-comparison) fixes BiLSTM and `boxes` (19 inputs), `boxes-2d-size-vel` (21), `boxes-2d-3d-size-vel` (24). All share four flags, source-only normalization and E001/E001b's 13 evaluation axes.

On `aalto`, the isolated checkout `/home/user20/projects/pedestrian-behavior-labeling-e001c` links its ignored E001/E001c output roots to the existing data. After pulling/verifying the intended `e001c-box-cache` revision, run:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e001c
uv run --locked --extra cu118 python scripts/train-e001c-seeds.py --jobs 2
```

This launches **30 attempts / 60 test cells**, seeds 0–4, max 75 epochs/patience 8, with two concurrent processes and two CPU threads per process. `--jobs 1` is sequential; `--seeds 0` runs six attempts. A fresh timestamped `outputs/experiments/E001c/runs/seeds-<UTC>/` contains `sweep.json`, per-attempt logs and readable source/configuration/BiLSTM/seed directories. Optional `--output <fresh-directory>` names the batch. Failures stop the queue and propagate, preserving evidence; no resume/retry. Reuse the [E001 tmux guidance](#e001-multiple-seeds) if needed.

For one attempt:

```bash
uv run --locked --extra cu118 python scripts/train-e001c.py \
  --source road-waymo --configuration boxes --seed 0 --device cuda \
  --output outputs/experiments/E001c/runs/road-boxes-seed0
```

Single attempts default to 75/8 and accept `--seed`, `--epochs`, `--patience`. Local commands use `--extra cpu --device cpu`. `--smoke` restricts a ROAD CUDA attempt to one epoch, two training batches and one source-validation batch; it never evaluates held-out tests. Comparisons reload the source-validation-selected checkpoint, then evaluate both tests. Local checkpoints/predictions, PNG/SVG plots and tidy W&B step/epoch/final namespaces reuse the shared runner. W&B names include E001c/source/final configuration/BiLSTM/seed/UTC time; no model or prediction artifacts are uploaded. The 3D size-velocity variant remains a redundant-input control, as documented in the protocol. Full comparisons remain user-launched.

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

Logits have shape `[B,T,4]`; padded logits are zero, internal missing slots remain predictions. This is model inference/acceptance only; Use the fixed single-attempt command below for training and final evaluation.


## E001 training and evaluation

Both frozen collections/setups must be available under `outputs/experiments/E001/{reader-preparation,data-setup}/{loki,road-waymo}/`. Native files are unnecessary. The command uses the [fixed protocol](experiments/E001-kinematic-transfer/README.md#loss-and-training), selects only on source validation, reloads `best.pt`, then evaluates both held-out tests using source normalization.

```bash
uv run --locked --extra cpu python scripts/train-e001.py \
  --source loki --configuration K --variant A --device cpu \
  --output outputs/experiments/E001/runs/loki-k-a-seed0
```

Use `--extra cu118 --device cuda` on `aalto` after Git push/pull and revision verification. Configurations are `K`, `K+T`, `K+T+R`, `RAW`; variants are A/B. Single attempts accept `--seed`, `--epochs` and `--patience`; defaults retain the original 0/30/5 protocol. W&B names include E001, source, configuration, `MLP`/`BiLSTM`, requested seed and UTC attempt time; see [naming and logging](experiments/E001-kinematic-transfer/README.md#wb-workspace-and-logging). Output directories must be fresh, including after failure. No resume/retry is implemented. W&B errors propagate after saving available local evidence.

Each attempt contains config/provenance/status, epoch history, per-update `steps.jsonl`, `best.pt`, `last.pt`, dataset-specific `metrics.json` and pickle-free `predictions.npz`, and PNG/SVG plots. Checkpoints and full predictions stay local; W&B receives configuration/provenance, per-update training CE, epoch training/validation curves and final metrics/tables/PNG plots. Provenance and denominators stay in configuration rather than metric history. Before valuable runs, arrange a separate backup; external backup storage remains **TBD**.

Open the [manual E001 workspace](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling?nw=71j6lc3jv4g) for the main native curves and collapsible final-evaluation/observation-strata sections. Automatic panel generation is disabled in that saved view. Detailed tables remain available in each run; local numerical files retain all metrics. [Metric names and axes](experiments/E001-kinematic-transfer/README.md#wb-workspace-and-logging) are fixed separately from W&B's internal logging-row counter.

### E001 multiple seeds

The [accepted repetition round](experiments/E001-kinematic-transfer/README.md#multiple-seed-launcher-2026-10-09) uses **75 maximum epochs / patience 8 / training seeds 0–4**: 80 attempts and 160 evaluation cells, including fresh seed-0 runs. Splits and source normalization remain frozen. The launcher defaults to **two separate processes sharing the selected GPU**; each has two CPU compute threads and zero loader workers. `--jobs 1` runs sequentially; `--seeds 0 1 2` gives the smaller 48-attempt matrix. Increasing concurrency may increase GPU/CPU/I/O contention; no throughput gain is promised.

On `aalto`, use the isolated checkout prepared at the revision from branch `e001-multiple-seeds`; its ignored E001 output directory links to the existing frozen data/output root. The original checkout and its 50/8 working edit stay intact. Run directly on the GPU laptop, or first connect from the ThinkPad using `ssh -F ~/.ssh/config aalto`:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e001-seeds
git rev-parse HEAD
tmux new -s e001-seeds
uv run --locked --extra cu118 python scripts/train-e001-seeds.py --jobs 2
```

Detach with **Ctrl-b, then d**; reconnect with `tmux attach -t e001-seeds`. An automatically fresh `outputs/experiments/E001/runs/seeds-<UTC timestamp>/` contains `sweep.json`, per-attempt log files in `logs/`, and separate readable `<source>-<configuration>-<MLP|BiLSTM>-seed<N>/` directories. Supply `--output <fresh-directory>` to name the batch explicitly. If a child fails, the launcher stops new launches, interrupts other active children and exits with failure, preserving available evidence. No automatic retries or resume. Use a fresh batch directory for a manually restarted subset of seeds.

The separate GPU smoke command uses one epoch, two training batches and one source-validation batch. It logs to the same W&B project with `job_type=gpu-smoke`, never accesses held-out tests and is not a comparison result:

```bash
uv run --locked --extra cu118 python scripts/train-e001.py \
  --source road-waymo --configuration K --variant B --device cuda --smoke \
  --output outputs/experiments/E001/runs/road-k-b-gpu-smoke
```

Generate the extended expected/actual report with `scripts/check-e001-data.py --output <fresh-directory>` through uv. Recompute metrics without inference:

```python
from pedestrian_behavior.evaluation import load_predictions, prediction_metrics, stratified_metrics
from pedestrian_behavior.experiments.e001 import CLASSES, STRATA
tracks = load_predictions("outputs/experiments/E001/runs/loki-k-a-seed0/loki-test/predictions.npz")
primary = prediction_metrics(tracks, len(CLASSES))
strata = stratified_metrics(tracks, len(CLASSES), STRATA)
```

`experiments.e001_run.plots()` can regenerate PNG/SVG exports from these results plus saved history/selection. Saved arrays retain every non-padding prediction, targets/GT masks, offsets, identities, archive references, source/grid times, condition memberships and individual 2D provenance flags.

## E001b training and evaluation

[E001b](experiments/E001b-2d-bbox-contribution/README.md) reuses E001 training/evaluation/reporting through thin wrappers. The two inputs are `availability` (14 columns) and `geometry` (20); both use BiLSTM only. Defaults are seed 0, 75 epochs and patience 8; the sweep declares seeds 0–4, two sources and two inputs: **20 training runs / 40 evaluation cells**, reusing matching E001 K+T+R BiLSTM baselines. Numeric baseline/bbox normalization comes from the training source for both tests. Existing E001 commands/defaults stay unchanged.

Use the isolated checkout `/home/user20/projects/pedestrian-behavior-labeling-e001b` on `aalto`. Its own ignored output links point to the original checkout's generated E001/E001b artifacts; the active E001 training checkout/environment is never updated. Code transfer is Git fetch/checkout with verified revision. From the isolated checkout, one fresh attempt:

```bash
/home/user20/.local/bin/uv run --locked --extra cu118 python scripts/train-e001b.py \
  --source road-waymo --configuration geometry --device cuda --seed 0 \
  --output outputs/experiments/E001b/runs/road-geometry-seed0
```

After E001 completes, run the full E001b matrix on the laptop:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e001b
tmux new -s e001b \
  '/home/user20/.local/bin/uv run --locked --extra cu118 python scripts/train-e001b-seeds.py --jobs 2'
```

Detach with `Ctrl-b d`; reconnect with `tmux attach -t e001b`. The launcher creates a timestamped fresh directory, per-attempt logs and `sweep.json`; failures stop the queue and propagate. No resume or retries. `--output`, `--jobs`, `--seeds` and `--device` are explicit overrides. Single-attempt `--epochs`/`--patience` changes are recorded deviations; the sweep fixes 75/8. Commands read saved caches; native mounts are needed only to prepare them.

A separate ROAD CUDA smoke uses `--smoke` with either configuration and a fresh output directory: one epoch, at most two training batches and one source-validation batch, with actual W&B logging and local checkpoint/plot checks. It never evaluates held-out tests or counts as a comparison result. Full checkpoints/predictions stay local. W&B keeps E001's step/epoch/evaluation namespaces, with additional verified-bbox strata and metadata in config. [Implementation and acceptance](experiments/E001b-2d-bbox-contribution/README.md#shared-runner-and-launcher-2026-10-09) record evidence.

## E002a pretrained-feature feasibility

The [E002a behavior pilot](experiments/E002a-feature-feasibility/README.md) has separate preparation, calibration, integration and training commands below. Use its isolated CUDA extractor environment and pinned author clones/weights; the main E001 environment and caches remain unchanged. The earlier bounded probe (`scripts/probe-e002a.py --help`) checks native crops, semantic BEV/alignment and resources without behavior training, held-out scoring, W&B or full caches.

The [pilot specification](experiments/E002a-feature-feasibility/README.md#deferred-training-pilot) fixes scope, dropout, fitting and sequential execution. The implementation reuses E001 loss, validation selection, checkpoints, metrics, plots and W&B; the pilot evaluates source validation only.

### E002a feature caches

The [cache inventory](experiments/E002a-feature-feasibility/README.md#cache-inventory-and-retirement--2026-10-22) records current paths. The old450 GB point LRU was deleted at the user's request; geometry and geometry+RGB training remain independent of it. Historical LRU cache utilities below can recreate that cache; current training cannot. Indexed storage and training wiring pass bounded checks; full preparation and operational acceptance remain pending. Existing prototype shards remain usable, while new preparation extracts directly from native sweeps.

The standalone indexed builder reads native LiDAR directly and does not use the retired LRU. The verified builder checkout on aalto is `/home/user20/projects/pedestrian-behavior-labeling-indexed-build`; its outputs alias the canonical root. It leaves the other checkouts unchanged. From that checkout, prepare **one source at a time**:

```bash
probe_root="$PWD/outputs/experiments/E002a"
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/build-e002a-point-cache.py \
  --device cuda --dataset road-waymo --output "$probe_root/indexed-road-waymo"
```

For LOKI, change `--dataset` to `loki` and `--output` to `"$probe_root/indexed-loki"`. Check remaining space after the first source; the208 GB ROAD and112–251 GB LOKI projections are estimates. Each command prepares only frozen eligible source training/validation tracks, using the existing full-sweep5 cm D-DITR input and target-specific20 cm output pooling. Features are stored FP16 and loaded FP32. No RGB inference, W&B or training is invoked. `--min-free-gb` defaults to20 (decimal GB); it guards each frame write, not total-cache fit. Fresh directories only; failures propagate with local `cache.json` evidence, no automatic retry/resume. A successful root publishes `manifest.json` after scene checksum/readback and frozen-input checks. For bounded native acceptance only, add `--max-scenes 1 --max-frames-per-scene 2`; that manifest is explicitly tagged partial. **Training wiring remains pending:** the existing full-arm trainer does not consume this format yet. [Builder protocol/evidence](experiments/E002a-feature-feasibility/README.md#standalone-indexed-cache-builder--2026-10-22).

Run from the isolated aalto checkout using its pinned extractor environment. This prepares **all frozen eligible training/validation RGB**, with3× crops; it performs no behavior training or test inference:

```bash
probe_root="$PWD/outputs/experiments/E002a"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/cache-e002a.py \
  --device cuda --dataset both --kind rgb \
  --point-cache-gb 450 \
  --output "$probe_root/rgb-cache-preparation"
```

RGB defaults to`outputs/experiments/E002a/rgb-cache/`; LiDAR defaults to`outputs/experiments/E002a/point-cache/`. Set`--rgb-cache-root`, `--point-cache-root`, `--point-cache-gb` and`--min-free-gb` without editing code. GB is decimal;450 GB is one shared point-payload ceiling, excluding RGB/index/temp files. Keep the20 GB free-space guard and operating headroom. A fresh evidence`--output` is required; matching immutable feature entries may be reused. One cache owner, zero workers. [Schemas, identities and failure semantics](experiments/E002a-feature-feasibility/README.md#production-feature-caches--2026-10-21).

Point entries are supplied lazily on misses by `FeatureSupply.iter_points(dataset, locator)`; measured `point_cost` references survive eviction and remain unknown until first measurement. `--kind points` explicitly warms the source population and is optional. Avoid full warming merely to fill the LRU. For bounded native/GPU acceptance on the six existing development tracks, use fresh dedicated roots:

```bash
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/check-e002a-cache.py \
  --device cuda --selection "$probe_root/execution-probe-04/execution.json" \
  --rgb-cache-root "$probe_root/cache-check-data/rgb" \
  --point-cache-root "$probe_root/cache-check-data/points" \
  --point-cache-gb 450 --output "$probe_root/cache-check"
```

This caches six complete RGB tracks and at most two usable point frames/track, then checks warm/reopened hits, independent native re-encoding, shared-frame reuse, capacity reduction and unchanged input/RNG evidence. It also compares bounded scene-image reuse against independent per-track extraction exactly. `--selection` may bound preparation requests; it never changes union populations or permits held-out tracks. Preparation groups requests by scene and reuses at most2 GB decoded images in RAM; saved crop semantics and16-crop encoder batches stay unchanged. Cache checking logs locally, with no W&B or training.

The accepted full RGB preparation is already complete at `rgb-cache-preparation-03/cache.json`:15,925 tracks, with every entry read back successfully. To repeat the hits-only integrity/GT-support audit, choose a fresh output:

```bash
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/audit-e002a-rgb.py \
  --device cuda --preparation "$probe_root/rgb-cache-preparation-03/cache.json" \
  --output "$probe_root/my-rgb-audit"
```

For a bounded CPU-only output-pooling/storage diagnostic on existing frozen features (no encoder inference or production cache writes), choose a fresh output:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/measure-e002a-pooling.py \
  --output "$probe_root/my-output-pooling-measurement"
```

It compares original tokens, constant10/20 cm and original target-near tokens with20 cm context. Diagnostic FP16 storage does not enable FP16 training. [Measurement scope/results](experiments/E002a-feature-feasibility/README.md#output-pooling-measurement--2026-10-22).

The separate target-specific two-resolution prototype reuses those frozen outputs and six complete development tracks. It writes new scene shards, checks selections/integrity and times complete-track loading plus bounded model CE/backward/validation; it does not change production caches or launch a comparison run:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/prototype-e002a-indexed.py \
  --device cuda --output "$probe_root/my-indexed-prototype"
```

Defaults use `native-probe-07`, `output-pooling-20261022/measurement.json`, `execution-probe-04` and `geometry-revision-calibration/profile.json`; each is overridable. A fresh output and25 GB free disk are required; the GPU probe requires10 GiB free VRAM and stops on failures. [Protocol and acceptance](experiments/E002a-feature-feasibility/README.md#target-specific-indexed-prototype--2026-10-22).

### E002a source-validation pilot

The first full attempt exposed a [population-scale input-pipeline bottleneck](experiments/E002a-feature-feasibility/README.md#full-population-throughput-check--2026-10-22); meet the [agreed throughput gate](experiments/E002a-feature-feasibility/README.md#throughput-gate--agreed-2026-10-22) before further pilot execution. The gate remains a documentation prerequisite, not an automated launcher check. **Current training reads prepared caches only:** RGB NPZ/SQLite are read-only and LiDAR uses the source's completed indexed cache. No encoder loading, native feature generation, point LRU or fallback on a miss. Geometry reads neither cache; geometry+RGB never opens a point cache or asks for point costs. Enabled-only65/194/323→256 fusion, normalization, settings and source-validation selection are unchanged.

Use `/home/user20/projects/pedestrian-behavior-labeling-indexed-train` on aalto, with shared canonical outputs and explicit `PYTHONPATH`; the active dataset-builder checkout stays unchanged. Complete both source caches before the six-attempt launcher. A missing/incomplete/wrong-population cache fails before W&B initialization. The supplied source manifest and all small scene manifests are pinned and checked; binary chunks are verified on access. Packing costs come from precomputed counts, without opening feature payloads or running inference. Full attempts reject partial caches; smoke attempts may use partial roots only if every selected complete track is covered.

The accepted 23 October profile is stored separately from the checkouts. Use it with the matching execution code and extractor environment (fresh output directories):

```bash
probe_root="$PWD/outputs/experiments/E002a"
execution_root="/home/user20/projects/e002a-execution-optimizations-20261023"
execution_profile="$execution_root/calibration-chunk4-final/profile.json"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/smoke-e002a.py \
  --profile "$execution_profile" \
  --rgb-preparation "$probe_root/rgb-cache-preparation-03/cache.json" \
  --indexed-cache-root "$probe_root" --output "$probe_root/my-indexed-integration"
```

Calibration retains the six complete development tracks and labelled shape stresses, reads their new pooled points and existing3× RGB fixtures, and measures real CE/backward/AdamW plus spatial concurrency. It preserves the78% VRAM allocation ceiling; frozen-miss reserve is zero because no encoder runs during fitting. The accepted profile uses up to four spatial frames per call, 1,810,000 ordinary padded tokens and 1,460,024 raw point tokens per physical microbatch; complete oversized tracks still use the existing fallback. [Execution measurements](experiments/E002a-feature-feasibility/README.md#execution-optimization-checks--2026-10-23) retain rejected larger budgets and asynchronous transfers. CPU point checks remain enabled using NumPy; GPU checks remain Torch operations. Feature caches need no rebuild.

After execution-code changes, revalidate the limits, then use the new complete output profile:

```bash
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/calibrate-e002a.py \
  --device cuda --indexed-cache-root "$probe_root" --reference-profile "$execution_profile" \
  --output "$execution_root/my-revalidated-calibration"
```

`--reference-profile` accepts completed profiles or explicit proposed settings; proposed settings become usable only after successful calibration. Revalidation measures an ordinary-mode boundary at or above the requested limit. Training requires a complete profile with matching execution hashes/backend/GPU and indexed point protocol. The smoke command makes first/repeated prepared reads (not guaranteed physically cold/warm disk) on both sources with actual W&B and checkpoint/plot/prediction reconstruction. These are operational checks, not comparison results; full-population throughput acceptance is separate.

Before the six attempts, measure complete64-track updates on shuffled, dense and long groups using the actual prepared pipeline:

```bash
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/check-e002a-throughput.py \
  --source road-waymo --arm g-rgb-dditr --device cuda \
  --profile "$execution_profile" \
  --rgb-preparation "$probe_root/rgb-cache-preparation-03/cache.json" \
  --output "$probe_root/my-road-throughput"
```

For a quick split, add `--breakdown`: only the same shuffled training group is run twice. LiDAR loading now overlaps one prepared CPU microbatch with model work; input iteration measures **consumer wait**, while nested point/RGB timers and `loader.prepared_seconds` measure background preparation and can overlap the remainder. Add `--serial` for non-overlapping stage attribution using the optimized reader. The remainder includes H2D, forward/backward, loss and optimizer/host bookkeeping; it is not pure GPU kernel time. Repeat per source/arm with fresh outputs. This is bounded operational timing with discarded weights, no W&B or held-out scoring. Each input-only case gets advisory-cold files with freshly reset chunk verification, then repeated files with retained process-local verification; `throughput.json` retains timings, packing/read counts and a conservative sampled stress-envelope extrapolation with explicit60 s overhead. It is not a measured full epoch or guaranteed cold/worst-case bound. [Readiness evidence](experiments/E002a-feature-feasibility/README.md#completed-caches-and-gpu-readiness--2026-10-23).

For one measured complete training+source-validation epoch, add `--full-epoch` instead of `--breakdown`. This runs the full frozen populations once with fresh verification/advisory-cold source pages, fixed seed/packing/settings, local per-update timings and source-validation metrics; weights are discarded. It records input wait, background preparation, data/model remainder, peak memory and assembly/cleanup time. W&B and checkpoint/plot export overhead are excluded, so this timing alone does not establish the complete operational or worst-case30-minute gate.

LiDAR frame preparation uses **four ordered CPU workers** by default. Override with `--point-workers N` on the single-run, pilot, calibration or smoke commands. Keep the BLAS/OpenMP limits above to avoid nested numerical threading; model-side PyTorch uses two CPU threads. `--worker-sweep` replaces the ordinary throughput cases with the same seeded64-track GPU update at1/2/4/8/16/32 workers (fresh-verification/advisory-cold and repeated reads). It preserves complete tracks/tensor order, records exact loss/token/packing parity and does not launch comparison training. The [worker measurements](experiments/E002a-feature-feasibility/README.md#parallel-frame-preparation--2026-10-23) select the default. `scripts/measure-e002a-spatial-chunks.py --help` exposes the bounded chunk/validation/physical-budget diagnostic; it records temporary overrides without changing a saved profile. Resource profiles from before the execution-code changes must be regenerated.

Then the manual six-attempt interface is:

```bash
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/train-e002a-pilot.py \
  --device cuda --profile "$execution_profile" \
  --rgb-preparation "$probe_root/rgb-cache-preparation-03/cache.json" \
  --indexed-cache-root "$probe_root"
```

Six sequential seed0 attempts: ROAD-Waymo then LOKI, each `g`, `g-rgb`, `g-rgb-dditr`; names `e002a-<source>-<arm>-s0`. Fresh outputs under `outputs/experiments/E002a/runs/pilot-<time>/`;100 epochs/patience15, effective64 tracks and warmup remain fixed. Both training and selection use frozen source development splits only. Read `sweep.json`, `history.json`, `steps.jsonl`, `packing.jsonl`, runtime read counters and W&B. Final paired validation changes/PNG/SVG comparisons are generated after all six; summaries reject mixed fusion or point-storage protocols. No retry/resume.

For one LiDAR attempt, replace the launcher with:

```bash
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/train-e002a.py \
  --source road-waymo --arm g-rgb-dditr --device cuda \
  --profile "$execution_profile" \
  --rgb-preparation "$probe_root/rgb-cache-preparation-03/cache.json" \
  --indexed-point-cache "$probe_root/indexed-road-waymo" \
  --output "$probe_root/runs/my-fresh-attempt"
```

`--indexed-point-cache` defaults to `outputs-root/experiments/E002a/indexed-<source>`; the sweep's `--indexed-cache-root` is their parent. Retired training options `--point-cache-root`, `--point-cache-gb`, `--min-free-gb` are removed; storage capacity/headroom belong to preparation. Geometry (`--arm g`) needs neither RGB preparation, point cache nor encoder probe. The existing [cache inventory](experiments/E002a-feature-feasibility/README.md#cache-inventory-and-retirement--2026-10-22) distinguishes training dependencies from historical fixtures.

### Geometry + D-DITR follow-up

The post-pilot `g-dditr` diagnostic uses Geometry64 + D-DITR128 with only `geometry_valid` and `lidar_valid` flags. It does not need RGB preparation, the RGB cache or DINO weights. The six-attempt launcher above remains unchanged. The revalidated profile is already saved at `/home/user20/projects/e002a-execution-optimizations-20261023/calibration-g-dditr-followup/profile.json`; use it directly for a run. The commands below run from the new isolated checkout, whose `outputs` link points to the canonical Aalto output directory. They do not change the existing training checkout.

The profile is complete, so **skip this calibration command** unless execution code changes again. For a future revalidation, use the current completed profile as the reference and choose a fresh output directory; then point training to that new profile:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-gdditr
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
probe_root="$PWD/outputs/experiments/E002a"
execution_root="/home/user20/projects/e002a-execution-optimizations-20261023"
execution_profile="$execution_root/calibration-chunk4-final/profile.json"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/calibrate-e002a.py \
  --device cuda --indexed-cache-root "$probe_root" --reference-profile "$execution_profile" \
  --output "$execution_root/my-fresh-recalibration"
```

Run one source at a time, with a fresh output directory for each:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-gdditr
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
probe_root="$PWD/outputs/experiments/E002a"
point_only_profile="/home/user20/projects/e002a-execution-optimizations-20261023/calibration-g-dditr-followup/profile.json"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/train-e002a.py \
  --source road-waymo --arm g-dditr --device cuda \
  --profile "$point_only_profile" --indexed-point-cache "$probe_root/indexed-road-waymo" \
  --output "$probe_root/runs/my-g-dditr-road-waymo-s0"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/train-e002a.py \
  --source loki --arm g-dditr --device cuda \
  --profile "$point_only_profile" --indexed-point-cache "$probe_root/indexed-loki" \
  --output "$probe_root/runs/my-g-dditr-loki-s0"
```

These are full source train/validation runs under the existing100-epoch /15-patience recipe, not a smoke or part of the original paired six-run summary. Keep each output directory fresh. See the [follow-up protocol and limits](experiments/E002a-feature-feasibility/README.md#post-pilot-geometry--d-ditr-follow-up--2026-10-24).

### E002b geometry and geometry + RGB

The independent launcher runs the three [E002b variants](experiments/E002b-normalization-capacity/README.md) for `g` and `g-rgb`: six sequential seed0 models per source,100 epochs maximum/patience15, complete source validation and no held-out evaluation. The shared LiDAR runner is separate and remains pending. Each model has its own W&B run, checkpoints and numerical evidence; the existing feature caches stay read-only.

Use the isolated E002b checkout on `aalto`. The calibration below must be complete for the matching code/backend before training; changing execution code requires a fresh calibration output. Existing E002a profiles are not accepted as E002b profiles.

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
e002a_root="$PWD/outputs/experiments/E002a"
e002b_profile="$PWD/outputs/experiments/E002b/calibration-independent/profile.json"

uv run --no-project --python "$e002a_root/extractor-env/bin/python" python scripts/run-e002b-independent.py \
  --source road-waymo --device cuda --profile "$e002b_profile" \
  --rgb-preparation "$e002a_root/rgb-cache-preparation-03/cache.json" \
  --output outputs/experiments/E002b/runs/road-independent-s0

uv run --no-project --python "$e002a_root/extractor-env/bin/python" python scripts/run-e002b-independent.py \
  --source loki --device cuda --profile "$e002b_profile" \
  --rgb-preparation "$e002a_root/rgb-cache-preparation-03/cache.json" \
  --output outputs/experiments/E002b/runs/loki-independent-s0
```

Keep each output directory fresh and do not update this checkout while its launcher is running. Add `--arms g` or `--arms g-rgb` to run only that arm's three variants. `scripts/train-e002b.py --help` exposes a single explicit arm/variant attempt. No resume, retry, cache rebuild or encoder inference is introduced.

For later recalibration, choose a new output directory and point subsequent training at its `profile.json`:

```bash
uv run --no-project --python "$e002a_root/extractor-env/bin/python" python scripts/calibrate-e002b.py \
  --device cuda \
  --reference-profile /home/user20/projects/e002a-execution-optimizations-20261023/calibration-g-dditr-followup/profile.json \
  --output outputs/experiments/E002b/my-fresh-calibration
```

The [DINO patch visual check](experiments/E002a-feature-feasibility/README.md#dino-patch-visual-check--2026-10-19) adds PCA/similarity panels and a corrected crop gallery from the same development sample: `scripts/check-e002a-rgb.py --help`. It reuses the completed probe and original checkpoint without a segmentation head or training.

Add `--compare-crop-scales` with a fresh output directory for the [1.5×/2×/3× side-by-side diagnostic](experiments/E002a-feature-feasibility/README.md#crop-size-visual-comparison--2026-10-19); every crop remains 256×256. These earlier diagnostics retain their original scales; the E002a pilot now uses cached CLS from3× crops.

For [ten full-track window videos](experiments/E002a-feature-feasibility/README.md#whole-track-window-videos--2026-10-19), use `scripts/video-e002a-tracks.py --help` in the same isolated GPU environment. Each combines a 3× RGB crop with nested 1.5×/2×/3× outlines and target-centered semantic LiDAR with 15/20/30 m radius rings.

For the [frozen D-DITR/Utonia PCA comparison](experiments/E002a-feature-feasibility/README.md#frozen-point-feature-pca--2026-10-20), reuse those ten tracks in the isolated extractor environment:

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/check-e002a-points.py \
  --probe "$probe_root/native-probe-07" \
  --videos "$probe_root/track-window-videos" \
  --utonia-code "$probe_root/dependencies/utonia" \
  --utonia-weights "$probe_root/weights/utonia.pth" \
  --output "$probe_root/point-pca-comparison" --device cuda
```

The output directory must be fresh. PCA is fitted once per encoder on a balanced training-only sample from both datasets. RGB and both point-feature BEVs share native frame/point identities; colors are arbitrary, independently fitted per encoder and fixed across videos. No behavior training or segmentation accuracy evaluation.

For the bounded [frozen semantic-head transfer check](experiments/E002a-feature-feasibility/README.md#frozen-semantic-head-transfer-check--2026-10-20), use the same isolated GPU environment:

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/check-e002a-semantics.py \
  --probe "$probe_root/native-probe-07" \
  --output "$probe_root/semantic-transfer" --device cuda
```

This scores the existing frozen nuScenes head on seven timestamp-matched Waymo development frames, with a declared common taxonomy. It saves local numerical fixtures, confusion PNG/SVG, GT/prediction BEVs and a report. Fresh output required; no training or held-out evaluation.

For the authorized [fresh linear semantic head](experiments/E002a-feature-feasibility/README.md#frozen-feature-linear-semantic-probe--2026-10-20) over the same frozen backbone, use the same isolated environment:

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/probe-e002a-linear.py \
  --probe "$probe_root/native-probe-07" \
  --output "$probe_root/linear-semantic-probe" --device cuda
```

Fresh output required. This extracts a bounded 48-training/12-validation-frame sample from separate existing development clips, trains only a nine-class affine head, and compares it with the original head and a taxonomy-only control. No behavior model, backbone updates, LOKI supervision or held-out scoring.

The bounded [execution/feature-supply probe](experiments/E002a-feature-feasibility/README.md#execution-and-feature-supply-probe--2026-10-21) caches3× RGB CLS for six complete development tracks and compares native D-DITR extraction, shared cache I/O and ordinary/recomputed/offloaded spatial workloads:

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/probe-e002a-execution.py \
  --probe "$probe_root/native-probe-07" \
  --output "$probe_root/execution-probe" --device cuda
```

Fresh output required. No additional point subsampling, temporal truncation, behavior fitting or held-out access. Larger resource microbatches repeat a real track as a shape test; they are not independent training examples.

For a metadata-only [400 GB cache simulation](experiments/E002a-feature-feasibility/README.md#bounded-cache-simulation--2026-10-21), using local frozen train/validation archives and the completed probe's size estimates:

```bash
uv run --locked --extra cpu python scripts/simulate-e002a-cache.py \
  --output outputs/experiments/E002a/cache-simulation
```

This replays six hypothetical source epochs without native reads, feature-cache construction, GPU inference or training. Sizes remain uncertain; the report separates cold/steady behavior and size sensitivities.
