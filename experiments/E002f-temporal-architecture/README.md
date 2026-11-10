# E002f — Temporal architecture on ROAD→LOKI

**Status: implementation prepared; GPU calibration/smokes and full runs not yet recorded.** This comparison asks whether a one-layer Transformer or a one-layer BiLSTM changes source-trained and target-adapted behavior labeling when the shared E002d fusion input and training budgets stay fixed. It covers ROAD-Waymo → LOKI only, seed0. No reverse transfer or additional temporal variants are in scope.

## Fixed comparison

Both models use E002d's `g-rgb-dditr` input: geometry/RGB/LiDAR branches, branch LayerNorm, RGB descriptor dropout0.5, and trainable spatial readout, fusion, temporal module, final LayerNorm and classifier. DINO and D-DITR features are precomputed, frozen inputs. ROAD statistics normalize both source training/validation and LOKI train/validation. Source training selects on ROAD validation; direct LOKI validation is then recorded before target adaptation.

| Architecture | Temporal module | Temporal + final norm + 4-class head |
|---|---|---:|
| `transformer-1layer` | One 128-wide, 4-head RoPE block; FFN512; dropout0.1 |199,044 |
| `bilstm-96` | Packed one-layer BiLSTM, hidden96/direction; output dropout0.1; 192→128 projection |199,044 |

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

Required checks: parameter counts and model specs; packed BiLSTM ignores suffix padding and returns zero padded outputs; source models train the branches/fusion end-to-end; direct target predictions reproduce epoch0 for both adaptation arms; exact source-head tensor equality; disabled adaptation dropout; identical target track order, GT, padding and batch boundaries; fresh optimizers and intended LR groups; checkpoint selection/reload; and a full64×102 CUDA update for both backbones. The bounded GPU smoke must execute the actual RGB/point path. Tests stay closed. Numerical results, run paths, checkpoint hashes, W&B references and deviations: **TBD**.
