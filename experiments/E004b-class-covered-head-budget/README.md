# E004b — Class-covered head label budgets

**Full pilot complete on 2026-11-13; validation assessment only.** Does initial four-class support improve low-label head adaptation versus the [E004a early-check rerun](../E004a-head-label-budget/README.md#early-check-rerun-2026-11-13)? Full training stays user-launched.

## Sampling

Use only the frozen 8,574-track LOKI training pool. For each draw seed 0–4:

1. From canonical track identities, shuffle the four class roles and sample uniformly among eligible tracks using a separate RNG at the draw seed.
2. Choose **four distinct tracks from four distinct scenarios**, one supporting each class. A designated track needs **at least five accepted GT timesteps of its role's class**, not necessarily contiguous. Padding/unlabeled slots do not count. Other classes may occur; **fit all accepted labels in each selected track**.
3. Prepend these four tracks **once**, then append E004a's recording-round-robin ordering, skipping selected track IDs. The remainder is ordinary sampling; it does not exclude other tracks from the seed scenarios. Fail if a role has no eligible candidate after the distinct-scenario exclusions.
4. Use nested prefixes at **5, 10, 25, 50, 100, 250, 500, 1,000 tracks**. Thus budget 5 is four initial tracks plus one ordinary track; budget 10 is those same four plus six. **No repeated groups of four or exact class balancing.**

Save seed roles, IDs, scenarios and class support, plus each subset's exact IDs and denominators. Run five draws per reduced budget, one canonical all 8,574-track fit and one zero-fitting-label reference: **41 fits /42 result rows**. Omit the one-track arm. This support floor does not guarantee within-class diversity.

## Fixed comparison and access

Reuse E004a's original ROAD-trained all-modality BiLSTM, source normalization and compact 128-D cache; only the original 516-parameter head learns. Keep the [early-check fitting, checkpoint selection and reporting](../E004a-head-label-budget/README.md#early-check-rerun-2026-11-13), including training seed 0, checks at 0/1/5/10/25/50/100 then every 100, a 10,000-update ceiling and approximately 1,500-update strict-F1 patience. No optimizer, loss, model or input changes. Same development/calibration and assessment scenarios; assessment never selects checkpoints or cutoffs. Tests remain closed.

**Sampling uses privileged training-GT class/support metadata over the full candidate pool.** Record this access separately from dense fitting labels, the fixed 930 development/calibration tracks and 931 assessment tracks. Screening cost is **unknown**, not free or a measured dense-label equivalent. E004b tests informed selection, not five-track total annotation effort or proven savings over target-only training.

## Run

On Aalto after syncing the reviewed revision:

```bash
cd /home/user20/projects/pedestrian-behavior-labeling-e002b-evaluation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run --locked --extra cu118 python scripts/train-e004b.py --device cuda
```

Fresh outputs: `outputs/experiments/E004b/runs/<timestamp>/`; W&B names identify `e004b` / `class-covered`. Existing E004a outputs/cache stay intact. Keep the existing two budget PNG/SVG figures and local numerical/checkpoint/prediction artifacts. `--smoke` runs only budgets 5/10, draw 0, 200 updates per fit, with offline W&B; full runs use online W&B. Logging errors propagate; no retry/resume.


Checks at `d816d98`: **294 CPU tests pass (13 CUDA skips)**, including six E004b sampler checks; CLI, links and diff checks pass. Aalto's two-fit200-update CUDA/offline-W&B smoke passes. All five real-data draws satisfy the four-role/five-GT/four-scenario rule; ordinary remainders, nested budgets and canonical all-data selection are verified. Source checkpoint SHA is unchanged. Evidence: `outputs/experiments/E004b/runs/20261113-090830-754399Z/verification.json`. At preparation, full41-fit training had **not started**; the user launched it afterward.


## Results

**All 41 fits plus the zero reference completed**, using revision `c844cee`, in `outputs/experiments/E004b/runs/20261113-091208-600096Z/`. Compare with E004a's completed early-check run `20261113-082847-737608Z`. Both use the same source model/cache, normalization, optimizer, selection cadence and assessment population. Subsets intentionally differ; draw IDs are not identical labeled samples.

F1 is assessment track-weighted macro-F1 (%), mean ±sample SD across five subset draws. Zero/all anchors each have one result. Δ is E004b minus E004a in percentage points, computed before rounding.

| Fitting tracks | E004a early-check F1 ±SD | E004b class-covered F1 ±SD | Δ (pp) |
|---|---:|---:|---:|
| 0 | 41.67 | 41.67 | +0.00 |
| 5 | 47.13 ±3.48 | 45.95 ±0.89 | -1.18 |
| 10 | 47.34 ±2.41 | 47.92 ±1.92 | +0.59 |
| 25 | 50.52 ±2.14 | 49.96 ±2.20 | -0.56 |
| 50 | 50.09 ±1.69 | 49.56 ±1.19 | -0.54 |
| 100 | 52.31 ±3.03 | 51.52 ±1.89 | -0.79 |
| 250 | 55.56 ±3.01 | 55.68 ±2.66 | +0.12 |
| 500 | 55.81 ±2.75 | 55.98 ±2.77 | +0.17 |
| 1,000 | 56.84 ±1.25 | 56.99 ±1.18 | +0.15 |
| 8,574 | 55.28 | 55.28 | +0.00 |

![Ordinary versus class-covered sampling: assessment F1](../../outputs/experiments/E004b/results-review/e004a-early-comparison/e004a-vs-e004b-f1.png) ([SVG](../../outputs/experiments/E004b/results-review/e004a-early-comparison/e004a-vs-e004b-f1.svg))

**Minimum class coverage does not yield a consistent mean F1 improvement.** Five-track F1 is lower by 1.18 pp, although draw-to-draw SD falls from 3.48 to 0.89 pp. Ten tracks improves slightly; 25–100 tracks is lower, and gains at 250–1,000 are below 0.2 pp. The zero and all-data anchors match. This tests the specific four-track/five-GT support floor; it does not establish that broader class diversity or different selection policies cannot help.

![Coverage and achieved error at the 5% calibration target](../../outputs/experiments/E004b/results-review/e004a-early-comparison/e004a-vs-e004b-coverage-error.png) ([SVG](../../outputs/experiments/E004b/results-review/e004a-early-comparison/e004a-vs-e004b-coverage-error.svg))

**Confidence-based automation remains below the intended usefulness target.** At five tracks, coverage falls **19.79%→7.68%**, while achieved assessment error falls **8.41%→6.78%**. At 1,000 tracks, E004b gives **56.79% coverage /10.30% error**, versus 57.51% /10.69% for E004a. All-data remains 54.83% /8.99%. The 5% calibration target is exceeded in 40/42 E004b checkpoint results, versus 41/42 on the same E004a budget grid. These are correlated model results on one assessment pool, not independent reliability trials.

**Cost qualification:** matched track counts do not guarantee identical GT volume. At five tracks, average accepted fitting GT is 178.6 timesteps for E004a versus 178.0 for E004b; per-fit counts are saved. E004b additionally uses privileged GT support metadata for all 8,574 training candidates, with unknown screening effort. Both use 930 development/calibration tracks and 931 assessment tracks. No total annotation-efficiency or target-only savings claim follows.

Report aggregates and sampling/reference metadata were checked against the saved result rows; the two plots are linked as PNG/SVG. Numerical evidence: [comparison JSON](../../outputs/experiments/E004b/results-review/e004a-early-comparison/comparison.json), [CSV](../../outputs/experiments/E004b/results-review/e004a-early-comparison/comparison.csv), [full E004b report](../../outputs/experiments/E004b/results-review/20261113-091208-600096Z/report/budget-report.json), [subset plan](../../outputs/experiments/E004b/results-review/20261113-091208-600096Z/subset-plans.json), [provenance](../../outputs/experiments/E004b/results-review/20261113-091208-600096Z/provenance.json). No new training, inference or test access for this review.
