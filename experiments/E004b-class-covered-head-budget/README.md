# E004b — Class-covered head label budgets

**Agreed 2026-11-13; implemented and checked.** Does initial four-class support improve low-label head adaptation versus the [E004a early-check rerun](../E004a-head-label-budget/README.md#early-check-rerun-2026-11-13)? Full training stays user-launched.

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


Checks at `d816d98`: **294 CPU tests pass (13 CUDA skips)**, including six E004b sampler checks; CLI, links and diff checks pass. Aalto's two-fit200-update CUDA/offline-W&B smoke passes. All five real-data draws satisfy the four-role/five-GT/four-scenario rule; ordinary remainders, nested budgets and canonical all-data selection are verified. Source checkpoint SHA is unchanged. Evidence: `outputs/experiments/E004b/runs/20261113-090830-754399Z/verification.json`. Full41-fit training has **not started**.
