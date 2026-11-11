# E001d: temporal capacity control

## Question and scope

Does increasing framewise MLP capacity to match the parameter count of E001's K+T+R BiLSTM change source and cross-dataset performance? This development-informed follow-up adds one model; it does not change E001 populations, labels, splits, normalization or evaluation.

Original E001 MLP/BiLSTM seed0–4 results and held-out tests informed later development. E001d evaluates both held-out datasets through the shared runner, but these are not pristine confirmatory tests. Match new attempts to the [original five-seed K+T+R MLP/BiLSTM controls](../E001b-2d-bbox-contribution/results-review.md#primary-track-weighted-macro-f1) by source and seed. Those controls are preserved under `outputs/experiments/E001/runs/seeds-20261009-084417-412679/`; no control retraining is needed.

## Model and fixed settings

The input is K+T+R, 12 dimensions. Starting from E001's framewise `KinematicClassifier`, append `Linear(64,194) → ReLU → Linear(194,64)` to its encoder. The appended block has no dropout and no final ReLU. Keep the original encoder, dropout and `64 → 4` readout. There is no recurrence or state across frames. The new model has **30,342 trainable parameters**; the matched original BiLSTM has **30,340**, while the original MLP has **5,252**.

Reuse [E001's protocol](../E001-kinematic-transfer/README.md), saved collections, K+T+R source-only normalization, condition definitions, equal-track training, loss, AdamW (`lr=0.001`, `weight_decay=0.0001`), batch size64, float32, gradient clipping1.0, 75 epoch maximum and patience8. Select on source validation, reload its best checkpoint, and evaluate both held-out tests with the source normalization. Original E001 defaults and artifacts remain unchanged.

The exactly ten new attempts are ROAD-Waymo and LOKI, each at seeds0–4. Match all settings to the original E001 seed0–4 K+T+R MLP/BiLSTM attempts. No target training or target-based selection is added. Retain failures and all metric denominators using the shared attempt runner.

## Command and outputs

On the verified Aalto evaluation checkout, the existing E001 output links provide collections/setups; no native inputs or preparation are needed. From `/home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation`:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
git rev-parse HEAD
uv run --locked --extra cu118 python scripts/train-e001d-seeds.py --jobs 2 --device cuda
```

The launcher creates a fresh `outputs/experiments/E001d/runs/matched-capacity-<UTC>/` with ten readable attempt directories, logs and `sweep.json`. Use `--jobs 1` for sequential execution or `--output <fresh-directory>` to name the batch. Each attempt follows E001 artifact handling and writes its model/settings, source statistics, predictions, metrics and checkpoint. A single source/seed attempt can be launched with `scripts/train-e001d.py` and explicit `--source`, `--seed`, `--device` and `--output`.

## Results

TBD; training and W&B runs are user-launched. No E001d model training has been started during implementation.

## Implementation check

The focused CPU test verifies both parameter counts, frame independence, padding, finite backward gradients, unchanged E001 defaults and the ten-attempt matrix. CLI help and documentation links pass. The existing shared runner is reused; no full suite, W&B smoke or comparison training was run during preparation.
