# E004a — Head adaptation versus labeled-track budget

**Pilot direction agreed 2026-11-12; not implemented or launched.** This is the first [E004 label-efficiency](../E004-label-efficiency/README.md) comparison. It asks how target-validation performance changes as more complete target-training tracks supervise only the source classifier. Target-only controls remain necessary before claiming annotation savings.

## Initial model and data flow

Use the original ROAD-trained [E002f all-modality BiLSTM](../E002f-temporal-architecture/README.md#fixed-comparison), selected at source epoch28: `outputs/experiments/E002f/runs/temporal-architecture-20261110-091857/source/bilstm-96/best.pt`. Exact checkpoint SHA256 will be frozen before implementation. This is a fixed competitive reference, not a claim of the universally best model.

```text
LOKI complete-track inputs, normalized with frozen ROAD statistics
  → frozen geometry / DINO / D-DITR spatial branches and fusion
  → frozen BiLSTM, projection and final LayerNorm
  → cached per-timestep 128-D representation
  → original source Linear(128, 4), fine-tuned on selected target GT only
```

Freeze the complete body, including final LayerNorm, in evaluation mode. Only the existing classifier's 516 parameters learn; do not initialize a fresh random head for this curve. No dropout during frozen extraction or head fitting. Reuse E002f's saved source-only target representations where compatible; no new encoder extraction or point-cache rebuild is planned. Keep full-track observations, including unlabeled context, without providing GT to model inputs.

First direction: ROAD-Waymo → LOKI. Select subsets only from the existing 8,574-track LOKI training pool; the 1,861-track validation population has already informed development. No test access in this pilot. The reverse direction is later work, not a second initial source-training run.

## Proposed budget grid and repeats

Candidate absolute training-track counts: **0, 1, 5, 10, 25, 50, 100, 250, 500, 1,000, all**. Zero is the original source classifier without target fitting. Each positive budget labels all accepted GT timesteps in the selected complete tracks. Report their raw GT count/duration, class support and distinct recordings alongside track count.

Use five independent subset draws for reduced budgets, with increasing subsets nested within each draw. Reuse identical subsets across later methods. Fix the model-training seed for this first pilot to focus on draw variation; exact draw seeds and training seed are TBD. Zero and all-data anchors do not become independent repetitions by duplicating the same inputs/initialization.

Recording-aware subset selection is **TBD**: exact track counts and complete-recording inclusion are different policies. Preserve the existing train/validation/test group split. Do not silently balance classes or resample until rare classes appear; any GT-informed selection would be additional annotation access and a different sampling protocol.

## Evaluation and annotation accounting

Plot mean and spread of target-validation four-class track-weighted macro-F1 against absolute training-track budget. Retain all four classes even when a training subset lacks one. Also report per-class precision/recall/F1, CE, subset support and compute; preserve failures and negative transfer. Do not choose a favorable subset draw.

Validation/early-stopping policy and its annotation cost are **TBD**. Using the full existing labeled validation set is acceptable for an explicitly developmental pilot, but would add a fixed 1,861 labeled tracks /60,231 GT timesteps beyond the fitting budget. A one-track fit must not be presented as one-track total target-label access. No final label-efficiency or reliability claim follows from a training-budget curve alone.

## To settle before implementation

| Item | State |
|---|---|
| Source body / head initialization / normalization | Original ROAD E002f full-fusion BiLSTM; fine-tune its head; ROAD statistics throughout |
| Source checkpoint hash / feature-cache identity | TBD; verify once against the existing files |
| Final budget grid and recording-aware draw rule | Proposed above; exact policy TBD |
| Repeat / initialization seeds | Five reduced-budget draws; exact seeds TBD |
| Loss | Existing accepted-GT mean CE per track, averaged over labeled tracks |
| Optimizer, regularization, fitting budget and stopping | TBD; low-budget convergence/selection must be specified rather than blindly inheriting full-label epochs |
| Validation labels, selection and later independent evaluation | TBD; count all target-label access |
| Coverage-at-error protocol | Secondary E004 endpoint; definitions proposed in the overview, numerical tolerance and audit policy TBD |

No new runs, code or evaluation artifacts exist for E004a yet. Existing E002f direct/full-label head results are context; the full-label head can be reused as an anchor only if the eventual initialization, fitting and selection recipe match.
