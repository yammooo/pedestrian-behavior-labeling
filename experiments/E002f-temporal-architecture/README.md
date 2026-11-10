# E002f — Temporal architecture on ROAD→LOKI

**Status: implemented and accepted, 2026-11-10; full comparison ready for user launch.** This comparison asks whether a one-layer Transformer or a one-layer BiLSTM changes source-trained and target-adapted behavior labeling when the shared E002d fusion input and training budgets stay fixed. It covers ROAD-Waymo → LOKI only, seed0. No reverse transfer or additional temporal variants are in scope.

## Fixed comparison

Both models use E002d's `g-rgb-dditr` input: geometry/RGB/LiDAR branches, branch LayerNorm, RGB descriptor dropout0.5, and trainable spatial readout, fusion, temporal module, final LayerNorm and classifier. DINO and D-DITR features are precomputed, frozen inputs. ROAD statistics normalize both source training/validation and LOKI train/validation. Source training selects on ROAD validation; direct LOKI validation is then recorded before target adaptation.

| Architecture | Temporal module | Temporal + final norm + 4-class head |
|---|---|---:|
| `transformer-1layer` | One 128-wide, 4-head RoPE block; FFN512; dropout0.1 |199,044 |
| `bilstm-96` | Packed one-layer BiLSTM, hidden96/direction; output dropout0.1; 192→128 projection |199,044 |
| Existing two-layer reference | Two 128-wide, 4-head RoPE blocks; FFN512; dropout0.1 |397,316 |

The existing [E002d source model](../E002d-rgb-lidar-complementarity/README.md) and [E003c source-initialized adaptations](../E003c-temporal-adaptation/README.md#full-source-initialized-results-2026-11-06) supply the two-layer reference; the launcher does not retrain it. Its ROAD→LOKI validation F1 is46.50% before adaptation,63.90% after head-only fitting and68.17% after temporal/head fitting. Wider historical two-layer models are not additional launcher arms.

The single-layer LSTM uses ordinary PyTorch biases; complete-track lengths are packed so backward recurrence never processes padded suffixes. Both fusion outputs are128-D. Each whole source model has552,196 trainable parameters, including the shared trainable branches/readout.

Each source-best checkpoint gets two independently initialized LOKI fits:

| Arm | Target features | Trainable parameters |
|---|---|---:|
| `head` | Frozen source temporal backbone + final LayerNorm outputs |516 |
| `temporal-head` | Architecture-specific pre-temporal fusion output |199,044 |

Both arms copy the exact original source classifier; the temporal arm also copies the selected source temporal module and final norm. No target-adapted head is reused. Source branch/fusion weights and input encoders remain frozen during target adaptation. Epoch0 target logits must reproduce direct source predictions on the same validation population.

## Recipes and access

Source fits use seed0, float32, batch64, AdamW (`lr=3e-4`, weight decay`1e-4`, betas`0.9/0.999`, epsilon`1e-8`), three-epoch warmup, global clip1, maximum100 epochs and patience15. The inherited E002d input profile retains its accepted complete-track and point-token bounds. The E002f calibration separately checks both backbones at64×102 through forward, masked track CE, backward and optimizer update. Actual source smokes exercise the prepared RGB and indexed-point paths.

Target fits use the E003c recipe: all8,574 LOKI training tracks and1,861 validation tracks; whole-track batches64; AdamW with weight decay`1e-4`; head LR`3e-4`; temporal+final-norm LR`3e-5`; fresh optimizer per arm; three-epoch warmup; clip1; maximum100 epochs; patience15; dropout disabled. Epoch0 is a checkpoint candidate. Selection maximizes track-weighted four-class macro-F1, breaks exact F1 ties by lower CE and then earlier epoch; patience resets only on strict F1 improvement. Training and validation use GT accepted by the existing target policy. No test split is accessed.

This is full target-label training and target-validation selection, not zero-shot or held-out transfer. E002f arms are paired within each source checkpoint but are not matched to a different architecture's source checkpoint beyond the fixed seed/recipe. Results can support a descriptive comparison under this protocol, not a causal claim that temporal architecture alone explains transfer differences.

## Run

On Aalto, from the active checkout, the sequential launcher calibrates both temporal shapes and then performs the two source fits, each architecture's fresh LOKI cache, and four target fits:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
bash scripts/run-e002f-sequential.sh
```

Bounded integration acceptance uses the existing explicit E003a three-train/two-validation track selection and creates separate smoke outputs:

```bash
E002F_SMOKE=1 bash scripts/run-e002f-sequential.sh
```

The launcher uses `outputs/experiments/E002a/extractor-env/bin/python`, prepared read-only caches, fresh timestamped output roots and one W&B run per fit. It stops on failure; it does not resume or retry. `E002F_RUN_ROOT`, `E002F_FEATURE_ROOT`, `E002F_PROFILE`, `E002F_PYTHON`, `E002F_DEVICE`, `E002F_SMOKE_SELECTION`, `E002F_PROBE`, `E002F_RGB_CACHE_ROOT`, and `E002F_RGB_PREPARATION` can select explicit paths. A non-complete existing profile or output root is rejected.

Standalone source, extraction and adaptation commands are available through `scripts/train-e002f.py`; resource calibration is `scripts/calibrate-e002f.py`. Inputs, caches and source checkpoints remain read-only. Full training is user-launched only.

## Acceptance and results

At revision `76589ff`, the full CPU suite passes272 tests with13 expected CUDA skips; all272 CUDA tests pass. Both backbones pass64×102 forward/masked-CE/backward/AdamW probes. These synthetic probes omit point tokens; actual read-only RGB/LiDAR paths are exercised by the source smokes. They verify execution bounds, not optimal GPU utilization or full-epoch runtime.

All six one-epoch GPU/W&B smokes complete with empty failure lists, using three training and two validation tracks per dataset. Source checkpoint comparisons confirm updates in geometry, RGB, spatial readout, fusion and temporal parameters. Saved predictions reconstruct selected metrics; both adaptation arms reproduce their source model's direct target predictions at epoch0 and retain ROAD normalization. Best/last checkpoints and PNG/SVG plots exist. W&B readback includes step loss, epoch CE/F1 and aggregate tables/plots; no model/checkpoint or full-prediction uploads. Original E002d checkpoint and protected inputs remain unchanged.

| Architecture | Source smoke | Head smoke | Temporal/head smoke |
|---|---|---|---|
| Transformer ×1 |[9dgb5s09](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/9dgb5s09)|[jp9x5dr3](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/jp9x5dr3)|[2sjb6pix](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/2sjb6pix)|
| BiLSTM |[ppyqf4ey](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ppyqf4ey)|[ch2j6chl](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ch2j6chl)|[woh9j6zm](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/woh9j6zm)|

Aalto evidence: `outputs/experiments/E002f/acceptance-20261110/{cuda-tests.log,smoke.log,verification.json}`, `calibration/acceptance-20261110/profile.json`, `runs/acceptance-20261110/` and `features/acceptance-20261110/`. CLI/help, shell syntax, local documentation links and diff checks pass. No full comparison training or held-out evaluation was launched; scientific results remain **TBD**.
