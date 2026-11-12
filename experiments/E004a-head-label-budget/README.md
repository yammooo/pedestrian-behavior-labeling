# E004a — Head adaptation versus labeled-track budget

**Pilot direction agreed 2026-11-12; not implemented or launched.** This is the first [E004 label-efficiency](../E004-label-efficiency/README.md) comparison. It asks how target performance changes as more complete target-training tracks supervise only the source classifier. Target-only controls remain necessary before claiming annotation savings.

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

The reporting specification below is fixed; the assessment/calibration populations and fitting protocol are not yet fixed. Retain all four classes even when a training subset lacks one. Also report per-class precision/recall/F1, CE, subset support and compute; preserve failures and negative transfer. Do not choose a favorable subset draw.

Validation/early-stopping policy and its annotation cost are **TBD**. Using the full existing labeled validation set is acceptable for an explicitly developmental pilot, but would add a fixed 1,861 labeled tracks /60,231 GT timesteps beyond the fitting budget. A one-track fit must not be presented as one-track total target-label access. No final label-efficiency or reliability claim follows from a training-budget curve alone.

## Fixed reporting specification (2026-11-12)

Produce two main figures: `budget-f1.png` / `.svg` and `budget-coverage.png` / `.svg`. Use absolute labeled-training-track count on the x-axis, linear from 0 to 1 then logarithmic above 1 so the zero-fitting-label anchor is visible. Place the all-data anchor at its actual track count. Show the fixed development/calibration-label overhead in captions and numerical records; this is not total-label cost on the x-axis.

| Figure | Y-axis and grouping | Repeats |
|---|---|---|
| **Budget versus F1** | Full-coverage four-class track-weighted macro-F1 (%) on the declared assessment population; one E004a head-adaptation series | Mean ± sample SD (`ddof=1`) over the five subset draws |
| **Budget versus automatic coverage** | Track-weighted coverage (%); four colored series for calibration error targets **1%, 5%, 10%, 20%**, with 5% the central reporting point | Mean ± sample SD of each run's achieved assessment coverage; never pool predictions across models |

Zero and all-data anchors have one fixed model result, not five independent copies; omit their SD bars. Use the same assessment population for every budget/draw. SD describes subset-draw variation conditional on the fixed source model/training seed, not uncertainty over new recordings. If the reporting population is also used to select a checkpoint or threshold, label results as development results rather than independent assessment.

### Acceptance rule and cutoff selection

- Acceptance unit: individual scored GT timestep. Score: maximum softmax probability of the predicted class, without a learned calibrator or rejection head.
- Primary weights: `1 / original_track_GT_count`, retained unchanged in every accepted subset. Primary accepted error is weighted misclassification fraction, not `1 - macro-F1`. Save raw-frame companions.
- For each selected checkpoint (each budget/draw, not each training epoch), compute the full curve at every distinct score cutoff. Accept equal-score ties together; no 5%-coverage quantization. Do not assume empirical error is monotone or stop at the first constraint violation.
- For each error target, choose the cutoff with **maximum calibration coverage** among those with observed calibration error at or below the target. If no nonempty accepted set qualifies, reject all: coverage 0, error undefined. No arbitrary minimum-support exclusion is added to this descriptive pilot; save accepted frames/tracks/groups and error counts.
- Freeze that run-specific cutoff and apply it unchanged to assessment predictions. The model may need a different cutoff at each budget/draw/error target. Never tune cutoffs on assessment GT.
- Plot the resulting assessment coverage even if its achieved error exceeds the calibration target; do not hide failed draws or readjust using assessment labels. Label legend entries as **calibration error targets**, not guaranteed assessment errors. Use hollow budget markers if any nonempty draw exceeds its target; the numerical summary gives the count of violations and reject-all cases.
- Coverage means/SD include reject-all runs as zeros. Accepted-error means/SD use nonempty accepted sets only and retain their repeat count; undefined error is never replaced by zero.

The coverage figure answers how much can be accepted using a rule selected to meet a calibration target. It does not guarantee that the target is achieved on unseen data. Save actual achieved error alongside every plotted point.

### Class retention and saved data

For each GT class `c`, retain the original assessment denominator and report:

```text
GT-class retention = accepted GT-c weight / all GT-c weight
accepted precision = accepted correct-c weight / accepted predicted-c weight
correctly automated fraction = accepted correct-c weight / all GT-c weight
```

Zero accepted predicted-c weight gives undefined precision; zero original GT-c support gives undefined retention/correctly automated fraction. Do not substitute recall computed only within the accepted subset for class retention. Numerical records include class supports and natural-availability cohorts; aggregate accuracy must not hide rejection of rare behaviors.

Save per-run F1/CE, full calibration and assessment error–coverage curves, selected cutoffs, achieved coverage/error, raw and weighted counts, class retention/precision, budget/draw identity and annotation-access counts. Save aggregate means/SD and repeat counts separately. Full curves are supporting numerical evidence, not a third required main figure; 5%-coverage tables or representative curves can be produced later without changing the calculation. Keep artifacts local under ignored `outputs/experiments/E004a/`, following existing prediction/provenance conventions.

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
| Coverage reporting | Two figures, timestep acceptance, track weights, 1%/5%/10%/20% calibration targets fixed above; calibration/audit membership and any formal uncertainty guarantee TBD |

No new runs, code or evaluation artifacts exist for E004a yet. Existing E002f direct/full-label head results are context; the full-label head can be reused as an anchor only if the eventual initialization, fitting and selection recipe match.
