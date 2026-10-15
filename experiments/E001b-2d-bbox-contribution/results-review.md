# E001, E001b and E001c — five-seed results

Reviewed 2026-10-10; extended **2026-10-15** with E001c. **130 completed attempts / 260 held-out evaluation cells**, seeds 0–4. Earlier seed-0 30/5 and 50/8 rounds and validation-only smokes are excluded from these averages. Designs: [E001](../E001-kinematic-transfer/README.md), [E001b](README.md), [E001c](../E001c-bbox-inputs/README.md).

- **BiLSTM improves on MLP in every seed for every matched E001 input/direction.** The improvement does not isolate temporal context from additional recurrent capacity.
- **RAW versus K+T+R depends on direction.** For BiLSTM, K+T+R beats RAW by 1.63 pp on ROAD within-domain, is nearly tied on ROAD → LOKI (+0.07 pp), and averages lower for LOKI within-domain (−0.82 pp) and LOKI → ROAD (−2.03 pp). The last disadvantage appears in four of five seeds.
- **Geometry helps LOKI → ROAD consistently:** +5.51 ± 0.50 pp versus availability, positive in all five seeds. Most of the gain is Stopped; Waiting worsens. Other directions show mixed seed-level gains. Availability alone has no consistent benefit across directions.
- **E001c does not improve the earlier geometry reference on average in any direction.** Basic `boxes` loses 3.50 ± 1.28 pp on ROAD→LOKI, negative in every seed. Higher Stopped F1 in some directions comes with worse Waiting/Crossing recovery. This is a changed input package, not an isolated center/dimension ablation.
- **2D size velocities show no systematic benefit.** Their mean effect is nonpositive in all four directions, with mixed seed effects. Added 3D size velocities are effectively zero/noise; differences from that control cannot demonstrate useful 3D size dynamics.
- **The epoch cap did not bind.** All 130 attempts stopped after eight non-improving source-validation epochs. This review provides no reason to raise the cap; it does not establish that patience 8 is optimal.

## Reading the numbers

R = ROAD-Waymo; L = LOKI. Direction is **training source → held-out test dataset**. Unless stated otherwise, values are percentages, **mean ± sample standard deviation across five seeds** (`ddof=1`). Each seed contributes one metric; no confusion matrices, tracks or frames are pooled across seeds to compute the reported mean F1. Differences are same-seed differences in **percentage points (pp)**. Same seed organizes replicates; differing input dimensions do not guarantee identical shared-weight initialization. No best-seed selection, significance claim or confidence interval is used.

Primary F1 is computed from a confusion matrix in which every track has total weight one (`1 / accepted_GT_count` per GT frame), then averages all four class F1s. It is not the mean of independently computed track F1s. Padding and unlabeled context are unscored. Five training seeds share one frozen group split: SD describes training variability, not uncertainty over new scenes or alternative splits.

All runs use the accepted 75-epoch / patience-8 budget, float32 AdamW and source-only normalization/selection. Frozen collection/setup/policy references and shared settings match across runs; E001b baseline statistics equal E001 K+T+R statistics exactly. Native Waymo FRONT geometry is used without ROAD fallback. Full populations remain eligible. Test scores did not select checkpoints; this retrospective test/transfer inspection must be disclosed in later development.

| Sweep | Attempts / test cells | Selected epoch range | Completed epoch range | Actual concurrent jobs | Code revision |
| --- | --- | --- | --- | --- | --- |
| E001 | 80 / 160 | 2–62 | 10–70 | 4 | `28a0b48b73c5704872eabb7063f08a17b5a555df` |
| E001b | 20 / 40 | 18–45 | 26–53 | 6 | `202c332a414613fb4f724ebaef529f956f72d606` |
| E001c | 30 / 60 | 10–48 | 18–56 | 8 | `25a27632736bbea280b8621fef03197bcd7a4cb2` |

E001/E001b completed on 2026-10-09; E001c completed on 2026-10-15. All attempt/sweep statuses are complete with no recorded failures or uncommitted source diffs. Selection and patience evidence were reconstructed from epoch histories; checkpoint tensors were not reloaded during this review. Concurrency differs, so runtime is not a controlled model-speed comparison.

| Test population | Groups | Tracks | Context slots | Accepted GT frames | GT frames M / S / W / C |
| --- | --- | --- | --- | --- | --- |
| ROAD-Waymo | 78 | 1,118 | 86,380 | 41,488 | 21,239 / 10,631 / 3,523 / 6,095 |
| LOKI | 93 | 1,929 | 71,558 | 69,267 | 44,718 / 5,768 / 8,953 / 9,828 |

## Primary track-weighted macro-F1

| Model/input | R→R | R→L | L→L | L→R |
| --- | --- | --- | --- | --- |
| MLP RAW | 43.82 ± 0.26 | 42.17 ± 0.31 | 55.23 ± 0.79 | 41.47 ± 0.13 |
| MLP K | 42.24 ± 0.18 | 41.98 ± 0.15 | 37.33 ± 2.65 | 26.93 ± 0.77 |
| MLP K+T | 42.76 ± 0.52 | 42.78 ± 0.48 | 43.01 ± 3.46 | 32.10 ± 1.44 |
| MLP K+T+R | 44.75 ± 0.24 | 42.72 ± 0.71 | 53.30 ± 2.45 | 39.51 ± 2.65 |
| BiLSTM RAW | 51.22 ± 1.91 | 46.83 ± 1.14 | 62.01 ± 0.82 | 44.25 ± 0.93 |
| BiLSTM K | 50.86 ± 2.11 | 47.79 ± 0.70 | 55.19 ± 2.00 | 37.58 ± 0.67 |
| BiLSTM K+T | 50.45 ± 1.98 | 45.77 ± 1.42 | 54.92 ± 3.06 | 36.79 ± 1.21 |
| BiLSTM K+T+R | 52.84 ± 1.04 | 46.90 ± 0.83 | 61.19 ± 1.94 | 42.22 ± 1.59 |
| BiLSTM availability | 51.75 ± 1.35 | 46.45 ± 1.84 | 61.21 ± 1.50 | 42.02 ± 1.08 |
| BiLSTM geometry | 52.18 ± 1.55 | 45.57 ± 1.10 | 62.03 ± 1.61 | 47.53 ± 0.91 |
| BiLSTM boxes | 51.12 ± 0.96 | 42.07 ± 1.81 | 60.24 ± 1.29 | 45.26 ± 1.45 |
| BiLSTM boxes-2d-size-vel | 51.10 ± 1.58 | 40.19 ± 2.50 | 59.35 ± 1.88 | 45.03 ± 0.90 |
| BiLSTM boxes-2d-3d-size-vel | 50.74 ± 0.61 | 41.53 ± 2.57 | 60.08 ± 1.55 | 46.36 ± 2.35 |

K = pedestrian velocity; T = displacement from first valid position; R = pedestrian–ego relative position and **relative velocity**. Availability and geometry extend exact K+T+R and use BiLSTM only. RAW is the agreed raw-state control, not a selected final representation.

[All seed values and summary statistics](../../outputs/experiments/E001b/analysis/20261015-seed-review/summary.csv) · [Seed plot PNG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/primary-f1.png) / [SVG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/primary-f1.svg).

### Bbox contribution: paired differences

| Direction | Availability − baseline (pp) | Positive seeds | Geometry − availability (pp) | Positive seeds | Geometry − baseline (pp) | Positive seeds |
| --- | --- | --- | --- | --- | --- | --- |
| R→R | -1.10 ± 1.23 | 2/5 | 0.43 ± 1.74 | 3/5 | -0.67 ± 2.17 | 1/5 |
| R→L | -0.45 ± 1.95 | 3/5 | -0.88 ± 2.84 | 2/5 | -1.33 ± 1.38 | 1/5 |
| L→L | 0.02 ± 1.82 | 2/5 | 0.82 ± 2.47 | 3/5 | 0.84 ± 3.39 | 3/5 |
| L→R | -0.21 ± 1.23 | 2/5 | 5.51 ± 0.50 | 5/5 | 5.30 ± 1.24 | 5/5 |

The principal comparison is geometry minus availability. The positive L→R result is directional, not evidence of universally transferable geometry. It also does not identify the responsible numeric feature or exclude camera/annotation dependence. R→L geometry is below the original baseline in four of five seeds.

[All paired comparisons, including MLP/BiLSTM and RAW/K+T+R](../../outputs/experiments/E001b/analysis/20261010-seed-review/paired-differences.csv) · [Geometry-minus-availability seed plot](../../outputs/experiments/E001b/analysis/20261010-seed-review/plots/bbox-differences.png).

## E001c: native box inputs

`boxes` uses 3D center XYZ, native dimensions and center velocity XYZ; image-normalized 2D geometric center UV, width/height and center velocity UV; plus four validity flags. `boxes-2d-size-vel` adds 2D width/height velocities; `boxes-2d-3d-size-vel` also adds all three 3D size velocities. Inputs are 19/21/24 columns, using the same BiLSTM widths, optimizer, source-only normalization/selection and frozen populations. No heading or explicit ego-state inputs.

**Interpret the comparison at the correct level.** E001b geometry retains K+T+R and bottom-v geometry. E001c replaces that representation, adds XYZ/native dimensions, changes bottom-v to geometric center and changes masks/input dimension. It is not a clean test of any one component. E001c was designed after the earlier test inspection: training/normalization/checkpoint selection remain source-only, but the representation is a **target-informed follow-up**, not an untouched confirmatory transfer design.

### Same-seed primary differences

Mean ± sample SD in pp; parentheses count positive differences out of five seeds. No seed is selected from test results.

| Comparison | R→R | R→L | L→L | L→R |
| --- | --- | --- | --- | --- |
| BiLSTM boxes − BiLSTM RAW | -0.10 ± 1.39 (2/5 ↑) | -4.76 ± 2.12 (0/5 ↑) | -1.77 ± 1.71 (1/5 ↑) | 1.01 ± 2.21 (3/5 ↑) |
| BiLSTM boxes − BiLSTM K+T+R | -1.72 ± 1.97 (1/5 ↑) | -4.83 ± 2.01 (0/5 ↑) | -0.95 ± 2.89 (3/5 ↑) | 3.04 ± 2.47 (5/5 ↑) |
| BiLSTM boxes − BiLSTM geometry | -1.05 ± 1.29 (1/5 ↑) | -3.50 ± 1.28 (0/5 ↑) | -1.79 ± 1.49 (1/5 ↑) | -2.26 ± 1.97 (1/5 ↑) |
| BiLSTM boxes-2d-size-vel − BiLSTM boxes | -0.02 ± 1.61 (2/5 ↑) | -1.87 ± 3.65 (1/5 ↑) | -0.88 ± 1.97 (2/5 ↑) | -0.24 ± 1.70 (3/5 ↑) |
| BiLSTM boxes-2d-3d-size-vel − BiLSTM boxes-2d-size-vel | -0.36 ± 1.45 (1/5 ↑) | 1.34 ± 3.11 (3/5 ↑) | 0.72 ± 3.05 (3/5 ↑) | 1.34 ± 2.82 (3/5 ↑) |
| BiLSTM boxes-2d-3d-size-vel − BiLSTM boxes | -0.38 ± 1.07 (2/5 ↑) | -0.53 ± 2.96 (2/5 ↑) | -0.16 ± 2.30 (3/5 ↑) | 1.10 ± 2.08 (3/5 ↑) |

- **Basic boxes versus earlier references:** ROAD→LOKI loses **3.50 ± 1.28 pp versus geometry** and **4.76 ± 2.12 pp versus RAW**, both negative in all five seeds. Within-domain means are also lower. LOKI→ROAD improves versus K+T+R in all seeds (+3.04 ± 2.47 pp), but remains below the earlier geometry mean (−2.26 ± 1.97 pp).
- **2D size velocity:** the incremental means are −0.02 / −1.87 / −0.88 / −0.24 pp in R→R / R→L / L→L / L→R. Seed effects are mixed and variability is substantial; this does not support promoting these extra inputs.
- **3D size-velocity control:** the cache audit found constant within-track 3D dimensions (LOKI velocities exactly zero; ROAD maximum 1.49e−10 m/s). Input-layer parameter count and initialization change with 21→24 inputs. The observed differences, including positive means in three directions, cannot be attributed to new physical size-dynamics information.

[Paired values and positive-seed counts](../../outputs/experiments/E001b/analysis/20261015-seed-review/paired-differences.csv) · [Comparison plot PNG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/e001c-differences.png) / [SVG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/e001c-differences.svg).

### Class trade-offs

Track-weighted class F1, mean ± SD. The following rows extend the E001/E001b class reference below.

| Direction | BiLSTM input | Moving | Stopped | Waiting | Crossing |
| --- | --- | --- | --- | --- | --- |
| R→R | boxes | 76.66 ± 1.79 | 65.71 ± 2.21 | 20.76 ± 5.22 | 41.36 ± 1.14 |
| R→R | boxes-2d-size-vel | 76.83 ± 0.69 | 64.97 ± 2.05 | 21.31 ± 5.02 | 41.31 ± 2.88 |
| R→R | boxes-2d-3d-size-vel | 77.09 ± 0.85 | 66.01 ± 0.64 | 19.11 ± 3.37 | 40.77 ± 2.24 |
| R→L | boxes | 81.08 ± 1.99 | 36.41 ± 3.26 | 20.54 ± 6.23 | 30.24 ± 0.85 |
| R→L | boxes-2d-size-vel | 78.96 ± 1.49 | 32.69 ± 4.43 | 18.20 ± 8.31 | 30.91 ± 2.81 |
| R→L | boxes-2d-3d-size-vel | 79.87 ± 1.43 | 34.87 ± 2.73 | 22.81 ± 7.58 | 28.58 ± 2.10 |
| L→L | boxes | 90.34 ± 0.34 | 55.88 ± 4.17 | 53.64 ± 2.93 | 41.08 ± 0.56 |
| L→L | boxes-2d-size-vel | 90.44 ± 0.22 | 51.47 ± 4.80 | 54.89 ± 1.33 | 40.61 ± 3.23 |
| L→L | boxes-2d-3d-size-vel | 90.53 ± 0.49 | 52.87 ± 3.19 | 56.11 ± 2.75 | 40.80 ± 2.69 |
| L→R | boxes | 74.90 ± 1.24 | 58.56 ± 5.17 | 14.18 ± 4.16 | 33.42 ± 3.39 |
| L→R | boxes-2d-size-vel | 74.84 ± 0.70 | 53.69 ± 3.03 | 17.83 ± 0.95 | 33.75 ± 1.61 |
| L→R | boxes-2d-3d-size-vel | 76.03 ± 0.93 | 56.36 ± 4.97 | 21.05 ± 3.10 | 32.02 ± 3.43 |

The largest ROAD→LOKI loss is **Stopped: 52.44 → 36.41% F1**, geometry→boxes. GT Stopped predicted Moving rises **15.5 → 46.8%** of row-normalized track-weighted mass. Waiting F1 improves (13.98 → 20.54%), but does not offset the other losses. This is not a uniform deterioration of every class.

In LOKI→ROAD, boxes raises Stopped F1 **51.96 → 58.56%**, but lowers Waiting **21.50 → 14.18%** and Crossing **39.83 → 33.42%**. Crossing predicted Moving rises **64.8 → 72.8%**. Within LOKI, Stopped also improves while Waiting/Crossing worsen. More bbox fields do not resolve the existing class trade-offs under this model/protocol.

[All 13 model/input class summaries, both weightings and GT support](../../outputs/experiments/E001b/analysis/20261015-seed-review/per-class-summary.csv).

### Secondary metrics and observation conditions

| Direction | BiLSTM input | Pooled macro-F1 (%) | Track accuracy (%) | Track CE |
| --- | --- | --- | --- | --- |
| R→R | boxes | 55.34 ± 1.96 | 66.98 ± 1.88 | 0.90 ± 0.06 |
| R→R | boxes-2d-size-vel | 55.63 ± 1.31 | 66.96 ± 0.92 | 0.94 ± 0.04 |
| R→R | boxes-2d-3d-size-vel | 55.26 ± 0.80 | 67.37 ± 0.67 | 0.89 ± 0.03 |
| R→L | boxes | 44.60 ± 1.86 | 67.02 ± 2.08 | 0.90 ± 0.05 |
| R→L | boxes-2d-size-vel | 43.04 ± 2.92 | 64.75 ± 1.81 | 0.99 ± 0.04 |
| R→L | boxes-2d-3d-size-vel | 44.72 ± 2.93 | 65.81 ± 2.13 | 0.95 ± 0.06 |
| L→L | boxes | 63.18 ± 1.18 | 79.69 ± 0.52 | 0.53 ± 0.01 |
| L→L | boxes-2d-size-vel | 62.09 ± 1.87 | 79.42 ± 0.43 | 0.52 ± 0.01 |
| L→L | boxes-2d-3d-size-vel | 63.33 ± 1.89 | 79.64 ± 0.93 | 0.53 ± 0.01 |
| L→R | boxes | 47.52 ± 1.93 | 63.80 ± 1.62 | 1.36 ± 0.09 |
| L→R | boxes-2d-size-vel | 46.81 ± 1.57 | 62.79 ± 1.05 | 1.33 ± 0.09 |
| L→R | boxes-2d-3d-size-vel | 48.40 ± 2.33 | 63.26 ± 1.41 | 1.33 ± 0.06 |

Accuracy alone gives a different impression in LOKI→ROAD: boxes has **63.80%** track accuracy versus geometry's **62.74%**, despite lower macro-F1 and worse CE (**1.36 versus 1.29**). Keep class recovery and the declared primary metric visible.

The following slices reuse exactly the same GT populations and per-slice track reweighting. Full results for all thirteen axes, with class/frame/track/group support, are in [stratum summaries](../../outputs/experiments/E001b/analysis/20261015-seed-review/strata-summary.csv).

| Direction | Slice | G/T/F; classes | boxes | + 2D size vel | + 2D/3D size vel |
| --- | --- | --- | --- | --- | --- |
| R→R | frame-bbox missing | 0/0/0; — | — | — | — |
| R→R | frame-bbox valid | 78/1118/41488; MSWC | 51.12 ± 0.96 | 51.10 ± 1.58 | 50.74 ± 0.61 |
| R→R | track-bbox never | 0/0/0; — | — | — | — |
| R→R | track-bbox partial | 78/1028/35359; MSWC | 51.49 ± 0.68 | 51.75 ± 1.61 | 51.34 ± 0.58 |
| R→R | transition near | 34/117/504; MSWC | 37.74 ± 2.64 | 37.54 ± 2.02 | 37.39 ± 2.83 |
| R→R | transition steady | 78/1110/40887; MSWC | 51.46 ± 0.99 | 51.38 ± 1.64 | 51.03 ± 0.67 |
| R→L | frame-bbox missing | 93/1813/38236; MSWC | 39.19 ± 2.29 | 37.24 ± 2.96 | 39.09 ± 2.83 |
| R→L | frame-bbox valid | 89/1386/31031; MSWC | 45.41 ± 2.05 | 44.28 ± 1.94 | 44.30 ± 1.84 |
| R→L | track-bbox never | 81/534/14255; MSWC | 35.50 ± 2.89 | 33.52 ± 3.04 | 36.60 ± 4.56 |
| R→L | track-bbox partial | 89/1281/52861; MSWC | 43.00 ± 1.99 | 41.69 ± 2.49 | 42.02 ± 1.69 |
| R→L | transition near | 59/332/1675; MSWC | 35.62 ± 2.24 | 34.43 ± 1.67 | 35.37 ± 1.98 |
| R→L | transition steady | 93/1909/67572; MSWC | 42.61 ± 1.86 | 40.74 ± 2.59 | 42.04 ± 2.56 |
| L→L | frame-bbox missing | 93/1813/38236; MSWC | 56.18 ± 1.54 | 55.71 ± 1.67 | 56.53 ± 1.81 |
| L→L | frame-bbox valid | 89/1386/31031; MSWC | 65.34 ± 1.49 | 64.18 ± 2.17 | 64.54 ± 1.14 |
| L→L | track-bbox never | 81/534/14255; MSWC | 51.22 ± 3.08 | 50.22 ± 1.39 | 52.73 ± 2.72 |
| L→L | track-bbox partial | 89/1281/52861; MSWC | 62.24 ± 1.51 | 61.39 ± 2.12 | 61.65 ± 1.37 |
| L→L | transition near | 59/332/1675; MSWC | 39.57 ± 1.20 | 38.24 ± 2.11 | 40.25 ± 3.23 |
| L→L | transition steady | 93/1909/67572; MSWC | 60.76 ± 1.29 | 59.88 ± 1.92 | 60.60 ± 1.53 |
| L→R | frame-bbox missing | 0/0/0; — | — | — | — |
| L→R | frame-bbox valid | 78/1118/41488; MSWC | 45.26 ± 1.45 | 45.03 ± 0.90 | 46.36 ± 2.35 |
| L→R | track-bbox never | 0/0/0; — | — | — | — |
| L→R | track-bbox partial | 78/1028/35359; MSWC | 45.73 ± 1.47 | 45.72 ± 0.93 | 46.93 ± 2.18 |
| L→R | transition near | 34/117/504; MSWC | 26.97 ± 2.05 | 28.03 ± 2.59 | 27.76 ± 3.93 |
| L→R | transition steady | 78/1110/40887; MSWC | 45.65 ± 1.45 | 45.41 ± 0.95 | 46.73 ± 2.37 |

ROAD still has no missing-bbox/never-bbox **scored** slice. LOKI retains all **534 never-box tracks**, rather than restricting evaluation to visible pedestrians. ROAD→LOKI boxes loses versus geometry on both missing-box frames (**39.19 versus 43.75%**) and valid-box frames (**45.41 versus 49.93%**); the regression is not confined to missing 2D input. Native bbox availability is not verified RGB visibility.

Near-boundary LOKI→ROAD F1 is **26.97 ± 2.05%** for boxes versus **31.27 ± 2.74%** for geometry. This remains frame-state scoring near boundaries, not transition-event/type F1. Small/unsupported slices retain the limitations described below; natural strata do not establish causal feature effects.

## Class results and confusion matrices

The E001/E001b reference values below are track-weighted class F1, mean ± SD; E001c rows are above. M/S/W/C = Moving / Stopped / Waiting to cross / Crossing. Full precision, recall, pooled-frame class results and supporting track counts are retained in [per-class summaries](../../outputs/experiments/E001b/analysis/20261010-seed-review/per-class-summary.csv) and [per-seed class results](../../outputs/experiments/E001b/analysis/20261010-seed-review/per-class.csv).

| Direction | Model/input | Moving | Stopped | Waiting | Crossing |
| --- | --- | --- | --- | --- | --- |
| R→R | MLP RAW | 76.70 ± 0.58 | 64.78 ± 0.34 | 0.00 ± 0.00 | 33.78 ± 1.15 |
| R→R | MLP K | 74.64 ± 0.46 | 64.65 ± 0.25 | 0.00 ± 0.00 | 29.66 ± 0.60 |
| R→R | MLP K+T | 74.94 ± 0.46 | 65.01 ± 0.49 | 0.31 ± 0.58 | 30.78 ± 1.58 |
| R→R | MLP K+T+R | 77.39 ± 0.51 | 65.00 ± 0.28 | 0.01 ± 0.03 | 36.60 ± 1.11 |
| R→R | BiLSTM RAW | 78.28 ± 0.86 | 69.15 ± 0.72 | 17.50 ± 7.91 | 39.95 ± 0.96 |
| R→R | BiLSTM K | 77.14 ± 0.69 | 69.81 ± 0.27 | 23.14 ± 6.51 | 33.35 ± 1.30 |
| R→R | BiLSTM K+T | 77.35 ± 0.11 | 70.08 ± 0.31 | 22.05 ± 6.79 | 32.31 ± 2.51 |
| R→R | BiLSTM K+T+R | 77.73 ± 0.90 | 70.11 ± 0.71 | 23.03 ± 4.43 | 40.51 ± 2.12 |
| R→R | BiLSTM availability | 77.90 ± 0.96 | 68.90 ± 1.13 | 19.52 ± 4.97 | 40.65 ± 2.05 |
| R→R | BiLSTM geometry | 77.84 ± 0.71 | 68.91 ± 1.67 | 20.74 ± 5.15 | 41.22 ± 1.86 |
| R→L | MLP RAW | 84.29 ± 1.06 | 55.05 ± 0.29 | 0.00 ± 0.00 | 29.34 ± 0.34 |
| R→L | MLP K | 84.12 ± 0.41 | 55.65 ± 0.40 | 0.00 ± 0.00 | 28.16 ± 0.66 |
| R→L | MLP K+T | 85.02 ± 0.40 | 57.20 ± 0.23 | 0.96 ± 1.85 | 27.95 ± 0.81 |
| R→L | MLP K+T+R | 84.12 ± 1.53 | 54.99 ± 0.88 | 0.55 ± 1.23 | 31.21 ± 1.31 |
| R→L | BiLSTM RAW | 85.68 ± 1.32 | 55.24 ± 2.05 | 13.18 ± 5.13 | 33.19 ± 2.11 |
| R→L | BiLSTM K | 86.34 ± 1.25 | 55.43 ± 2.51 | 19.37 ± 4.14 | 30.03 ± 2.02 |
| R→L | BiLSTM K+T | 86.94 ± 0.57 | 54.95 ± 2.33 | 13.19 ± 4.16 | 28.00 ± 3.17 |
| R→L | BiLSTM K+T+R | 84.17 ± 1.72 | 54.51 ± 1.45 | 15.31 ± 1.43 | 33.62 ± 1.20 |
| R→L | BiLSTM availability | 85.23 ± 1.54 | 51.75 ± 3.11 | 13.94 ± 4.12 | 34.87 ± 1.44 |
| R→L | BiLSTM geometry | 83.17 ± 0.44 | 52.44 ± 2.12 | 13.98 ± 4.71 | 32.66 ± 1.74 |
| L→L | MLP RAW | 89.62 ± 0.21 | 39.87 ± 2.79 | 53.92 ± 1.44 | 37.52 ± 0.70 |
| L→L | MLP K | 88.89 ± 0.25 | 4.91 ± 10.97 | 54.44 ± 0.58 | 1.07 ± 1.60 |
| L→L | MLP K+T | 87.95 ± 0.42 | 15.96 ± 17.08 | 51.47 ± 2.71 | 16.65 ± 3.68 |
| L→L | MLP K+T+R | 89.68 ± 0.43 | 35.15 ± 5.95 | 52.11 ± 0.47 | 36.27 ± 4.31 |
| L→L | BiLSTM RAW | 90.82 ± 0.41 | 51.50 ± 1.19 | 59.08 ± 0.65 | 46.64 ± 2.13 |
| L→L | BiLSTM K | 89.96 ± 0.19 | 42.47 ± 5.75 | 57.35 ± 0.61 | 31.00 ± 3.42 |
| L→L | BiLSTM K+T | 89.84 ± 0.70 | 42.56 ± 8.28 | 56.87 ± 1.44 | 30.42 ± 3.08 |
| L→L | BiLSTM K+T+R | 91.35 ± 0.29 | 46.77 ± 8.27 | 57.92 ± 0.39 | 48.71 ± 1.29 |
| L→L | BiLSTM availability | 91.11 ± 0.42 | 47.83 ± 4.14 | 57.25 ± 1.76 | 48.64 ± 1.16 |
| L→L | BiLSTM geometry | 91.04 ± 0.27 | 48.24 ± 4.31 | 59.62 ± 1.94 | 49.20 ± 1.55 |
| L→R | MLP RAW | 76.30 ± 0.21 | 35.62 ± 3.29 | 23.40 ± 1.78 | 30.56 ± 1.60 |
| L→R | MLP K | 77.04 ± 0.10 | 1.18 ± 2.65 | 28.54 ± 0.16 | 0.98 ± 1.34 |
| L→R | MLP K+T | 76.45 ± 0.36 | 2.97 ± 3.75 | 27.92 ± 0.46 | 21.06 ± 4.26 |
| L→R | MLP K+T+R | 76.81 ± 0.34 | 22.21 ± 9.56 | 27.40 ± 2.33 | 31.61 ± 2.06 |
| L→R | BiLSTM RAW | 75.16 ± 0.80 | 38.28 ± 3.92 | 27.31 ± 0.74 | 36.27 ± 3.89 |
| L→R | BiLSTM K | 75.90 ± 0.91 | 14.25 ± 2.10 | 29.76 ± 0.49 | 30.42 ± 1.63 |
| L→R | BiLSTM K+T | 76.03 ± 0.74 | 11.98 ± 5.08 | 29.39 ± 1.24 | 29.75 ± 0.95 |
| L→R | BiLSTM K+T+R | 75.65 ± 1.02 | 28.33 ± 7.05 | 25.90 ± 3.36 | 39.01 ± 1.50 |
| L→R | BiLSTM availability | 74.61 ± 1.26 | 27.24 ± 4.57 | 26.40 ± 1.67 | 39.82 ± 1.87 |
| L→R | BiLSTM geometry | 76.81 ± 1.06 | 51.96 ± 2.23 | 21.50 ± 2.29 | 39.83 ± 4.14 |

### What the errors explain

- **ROAD-trained models rarely recover Waiting.** K+T+R Waiting recall averages 14.3% within ROAD and 8.9% on LOKI; 64.8% and 77.8% of the corresponding GT Waiting track-weighted mass is predicted Stopped. Moving remains much easier than Waiting/Crossing.
- **LOKI→ROAD geometry changes the Stopped/Waiting balance.** Stopped F1 rises **27.24 → 51.96%**, with recall **17.5 → 43.1%**, versus availability. Stopped predicted Waiting falls **52.3 → 25.0%**. Waiting F1 falls **26.40 → 21.50%**, recall **52.2 → 25.8%**, and Waiting predicted Stopped rises **19.4 → 42.8%**. The overall improvement is not an improvement in every class.
- **Crossing is often predicted Moving.** Across these four BiLSTM inputs and directions, roughly 58–68% of GT Crossing weighted mass is predicted Moving. Geometry does not resolve that failure: L→R Crossing F1 is essentially unchanged versus availability (39.82 → 39.83%), while recall falls 38.1 → 33.6%.
- **The LOKI K+T+R baseline has variable Stopped F1:** 46.77 ± 8.27% within-domain. Its overall 61.19 ± 1.94% can hide that class variability. RAW averages better Stopped F1 but worse Crossing F1, so a single feature ranking hides a tradeoff.

Mean precision/recall percentages for the bbox comparison are below; their SDs and all classes are in the linked complete numerical file. F1 is averaged separately, not calculated from these averaged P/R values.

| Direction | BiLSTM input | Stopped P/R | Waiting P/R | Crossing P/R |
| --- | --- | --- | --- | --- |
| R→R | K+T+R | 65.8/75.0 | 60.6/14.3 | 58.0/31.4 |
| R→R | availability | 66.0/72.1 | 49.4/12.4 | 60.4/30.8 |
| R→R | geometry | 66.2/71.9 | 45.6/13.9 | 59.8/31.5 |
| R→L | K+T+R | 40.8/82.2 | 55.1/8.9 | 32.8/35.4 |
| R→L | availability | 38.4/80.5 | 53.0/8.1 | 41.2/30.4 |
| R→L | geometry | 39.5/78.1 | 45.4/8.4 | 30.0/36.0 |
| L→L | K+T+R | 57.4/40.2 | 52.2/65.8 | 61.4/40.4 |
| L→L | availability | 58.3/40.6 | 52.2/63.5 | 61.3/40.4 |
| L→L | geometry | 62.0/39.6 | 52.8/68.5 | 60.4/41.7 |
| L→R | K+T+R | 64.0/18.6 | 17.6/49.9 | 45.0/34.9 |
| L→R | availability | 62.5/17.5 | 17.7/52.2 | 42.2/38.1 |
| L→R | geometry | 65.5/43.1 | 18.6/25.8 | 50.1/33.6 |

Confusion plots have **GT rows / prediction columns** and show the mean of five independently row-normalized track-weighted matrices. The off-diagonal entries above are classification errors, not temporal transitions. Numerical count matrices, row means and SDs for **all thirteen model/input rows, all four directions, both weighting schemes** are in [confusion summaries](../../outputs/experiments/E001b/analysis/20261015-seed-review/confusion-summary.json).

| Direction | Confusion comparison plot |
| --- | --- |
| R→R | [PNG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/confusions-r-r.png) / [SVG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/confusions-r-r.svg) |
| R→L | [PNG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/confusions-r-l.png) / [SVG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/confusions-r-l.svg) |
| L→L | [PNG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/confusions-l-l.png) / [SVG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/confusions-l-l.svg) |
| L→R | [PNG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/confusions-l-r.png) / [SVG](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/confusions-l-r.svg) |

### Secondary metrics

These do not replace the primary metric. Pooled-frame F1 gives long tracks more influence; track-weighted accuracy is equivalent to mean track accuracy. The disparity between high accuracy and lower macro-F1 reflects uneven class recovery. In L→R, geometry's pooled F1 and track accuracy also improve, so its gain is not confined to the primary aggregation; CE changes much less.

| Direction | BiLSTM input | Pooled-frame macro-F1 (%) | Track-weighted accuracy (%) | Track CE |
| --- | --- | --- | --- | --- |
| R→R | RAW | 55.78 ± 2.86 | 69.21 ± 0.69 | 0.85 ± 0.04 |
| R→R | K+T+R | 58.46 ± 1.15 | 69.18 ± 0.74 | 0.85 ± 0.02 |
| R→R | availability | 57.41 ± 2.29 | 68.92 ± 0.85 | 0.83 ± 0.02 |
| R→R | geometry | 58.02 ± 1.75 | 68.88 ± 0.92 | 0.83 ± 0.02 |
| R→L | RAW | 48.72 ± 1.69 | 72.34 ± 1.51 | 0.75 ± 0.05 |
| R→L | K+T+R | 48.89 ± 1.26 | 70.56 ± 1.85 | 0.79 ± 0.06 |
| R→L | availability | 48.26 ± 1.72 | 71.65 ± 1.94 | 0.77 ± 0.04 |
| R→L | geometry | 46.72 ± 1.65 | 69.14 ± 0.64 | 0.79 ± 0.01 |
| L→L | RAW | 65.13 ± 0.87 | 80.38 ± 0.46 | 0.50 ± 0.01 |
| L→L | K+T+R | 64.88 ± 1.78 | 80.46 ± 0.55 | 0.50 ± 0.02 |
| L→L | availability | 64.67 ± 1.40 | 80.35 ± 0.66 | 0.50 ± 0.01 |
| L→L | geometry | 66.02 ± 1.29 | 80.56 ± 0.61 | 0.50 ± 0.01 |
| L→R | RAW | 45.06 ± 1.64 | 58.80 ± 1.16 | 1.26 ± 0.06 |
| L→R | K+T+R | 43.60 ± 2.21 | 57.13 ± 1.58 | 1.31 ± 0.10 |
| L→R | availability | 43.28 ± 0.98 | 55.98 ± 1.56 | 1.30 ± 0.11 |
| L→R | geometry | 50.00 ± 0.81 | 62.74 ± 1.17 | 1.29 ± 0.06 |

[Complete secondary summaries for every model/input, including support-weighted F1 and median track accuracy](../../outputs/experiments/E001b/analysis/20261010-seed-review/secondary-summary.csv).

## Observation strata

Every slice recomputes each represented track's weight as `1 / accepted_GT_count_in_slice`. Slice macro-F1 averages only **GT-supported classes**. `—` means empty/unscored, never zero. G/T/F = supporting groups / tracks / accepted GT frames; class codes show which classes enter that slice's macro. Full class-frame and class-track supports are in [complete stratum summaries](../../outputs/experiments/E001b/analysis/20261010-seed-review/strata-summary.csv).

Compare configurations inside the same slice. Scores between slices with different classes are not directly comparable, and natural cohorts also differ in class composition, duration and scenes. Tracks/groups may appear in several frame-level bins, so those counts do not add to the full population. Native union duration, resampled context coverage and internal gaps retain their agreed definitions; range is physical and unnormalized.

### All ten original axes: K+T+R BiLSTM reference

ROAD support applies to R→R and L→R; LOKI support applies to R→L and L→L. This compact reference covers every original bin. The linked complete file contains all inputs/models and, for bbox models/reference, all thirteen axes.

| Axis/bin | ROAD G/T/F; classes | LOKI G/T/F; classes | R→R | R→L | L→L | L→R |
| --- | --- | --- | --- | --- | --- | --- |
| duration [0,1) | 2/3/12; MSC | 48/149/459; MSWC | 55.56 ± 0.00 | 27.68 ± 0.79 | 36.18 ± 2.44 | 56.70 ± 3.58 |
| duration [1,5) | 21/58/768; MSWC | 82/690/10133; MSWC | 29.79 ± 1.81 | 41.29 ± 1.34 | 48.94 ± 1.71 | 29.41 ± 2.80 |
| duration [5,10) | 37/143/3044; MSWC | 78/591/21295; MSWC | 42.56 ± 0.81 | 45.04 ± 0.51 | 56.10 ± 2.42 | 42.72 ± 2.20 |
| duration [10,15) | 45/229/7158; MSWC | 43/244/14313; MSWC | 52.42 ± 1.27 | 49.59 ± 0.97 | 57.78 ± 2.76 | 43.22 ± 2.19 |
| duration [15,inf) | 64/685/30506; MSWC | 30/255/23067; MSWC | 55.51 ± 1.44 | 44.05 ± 1.89 | 67.49 ± 1.57 | 40.99 ± 1.42 |
| position-coverage [0,.25) | 15/27/1212; MSWC | 6/11/120; MS | 18.60 ± 0.45 | 47.05 ± 1.08 | 47.62 ± 0.00 | 22.14 ± 4.02 |
| position-coverage [.25,.75) | 49/157/8550; MSWC | 29/62/1383; MSWC | 57.68 ± 1.90 | 43.63 ± 2.53 | 54.52 ± 6.42 | 41.77 ± 4.82 |
| position-coverage [.75,1] | 68/934/31726; MSWC | 93/1856/67764; MSWC | 52.46 ± 1.08 | 46.93 ± 0.83 | 61.13 ± 2.03 | 42.40 ± 1.72 |
| gt-count 1 | 5/8/8; MSC | 12/20/20; MSW | 25.84 ± 8.76 | 3.21 ± 0.07 | 29.63 ± 0.00 | 25.66 ± 14.93 |
| gt-count 2-4 | 19/44/140; MSWC | 44/111/339; MSWC | 29.72 ± 9.74 | 31.59 ± 0.77 | 38.82 ± 2.75 | 39.55 ± 4.14 |
| gt-count 5-19 | 46/332/3997; MSWC | 77/554/6681; MSWC | 40.69 ± 1.58 | 39.44 ± 1.81 | 46.63 ± 1.51 | 38.72 ± 3.10 |
| gt-count 20+ | 72/734/37343; MSWC | 87/1244/62227; MSWC | 57.27 ± 1.00 | 49.51 ± 1.13 | 62.57 ± 2.19 | 43.72 ± 2.21 |
| track-2d never | 0/0/0; — | 81/534/14255; MSWC | — | 38.33 ± 0.80 | 58.60 ± 1.45 | — |
| track-2d partial | 78/1028/35359; MSWC | 89/1281/52861; MSWC | 53.06 ± 0.91 | 49.73 ± 0.96 | 61.12 ± 2.28 | 42.22 ± 1.67 |
| track-2d complete | 28/90/6129; MSWC | 50/114/2151; MSWC | 50.08 ± 3.30 | 46.86 ± 3.92 | 60.46 ± 3.46 | 40.16 ± 2.69 |
| internal-position-gap none | 75/1105/40990; MSWC | 93/1929/69267; MSWC | 52.83 ± 1.08 | 46.90 ± 0.83 | 61.19 ± 1.94 | 42.16 ± 1.62 |
| internal-position-gap (0,1] | 4/13/498; MSW | 0/0/0; — | 60.19 ± 1.34 | — | — | 46.85 ± 7.23 |
| internal-position-gap (1,inf) | 0/0/0; — | 0/0/0; — | — | — | — | — |
| position missing | 63/364/6801; MSWC | 0/0/0; — | 52.18 ± 4.77 | — | — | 34.60 ± 2.93 |
| position valid | 78/1105/34687; MSWC | 93/1929/69267; MSWC | 53.44 ± 0.88 | 46.90 ± 0.83 | 61.19 ± 1.94 | 43.38 ± 1.66 |
| velocity missing | 63/364/6804; MSWC | 12/20/20; MSW | 52.18 ± 4.77 | 3.21 ± 0.07 | 29.63 ± 0.00 | 34.61 ± 2.94 |
| velocity valid | 78/1103/34684; MSWC | 93/1909/69247; MSWC | 53.53 ± 0.87 | 47.49 ± 0.84 | 61.34 ± 1.95 | 43.41 ± 1.67 |
| frame-2d missing | 0/0/0; — | 93/1813/38236; MSWC | — | 44.98 ± 0.96 | 58.77 ± 1.97 | — |
| frame-2d valid | 78/1118/41488; MSWC | 89/1386/31031; MSWC | 52.84 ± 1.04 | 52.00 ± 0.64 | 62.97 ± 2.63 | 42.22 ± 1.59 |
| range [0,10) | 11/23/388; MSC | 87/672/7579; MSWC | 73.65 ± 2.82 | 48.45 ± 2.85 | 57.51 ± 3.41 | 49.85 ± 14.69 |
| range [10,20) | 43/184/1742; MSWC | 90/1077/16269; MSWC | 50.65 ± 2.93 | 48.48 ± 0.91 | 59.84 ± 3.25 | 48.21 ± 1.63 |
| range [20,40) | 71/732/11530; MSWC | 89/1281/26621; MSWC | 50.82 ± 1.21 | 48.88 ± 0.39 | 61.11 ± 1.84 | 43.66 ± 2.01 |
| range [40,inf) | 75/877/21027; MSWC | 86/836/18798; MSWC | 53.98 ± 1.92 | 40.29 ± 1.54 | 59.04 ± 1.66 | 41.43 ± 1.75 |
| range unknown | 63/364/6801; MSWC | 0/0/0; — | 52.18 ± 4.77 | — | — | 34.60 ± 2.93 |
| transition near | 34/117/504; MSWC | 59/332/1675; MSWC | 41.12 ± 1.47 | 41.34 ± 1.45 | 39.79 ± 0.73 | 33.55 ± 2.40 |
| transition steady | 78/1110/40887; MSWC | 93/1909/67572; MSWC | 53.26 ± 1.07 | 47.68 ± 0.85 | 61.70 ± 1.92 | 42.55 ± 1.57 |
| transition unknown | 30/85/97; MSWC | 12/20/20; MSW | 52.56 ± 6.81 | 3.21 ± 0.07 | 29.63 ± 0.00 | 40.33 ± 7.06 |

### Verified bbox strata: baseline, availability and geometry

Baseline predictions were retrospectively sliced using the matching saved E001b condition masks after checking identities, targets and GT masks; no inference or checkpoint selection was repeated. On these evaluations, original annotated-2D and verified-bbox frame/track strata match exactly. That equality is dataset-specific evidence, not a general equivalence between annotation presence and numeric usability.

| Direction | Axis/bin | G/T/F; classes | Baseline | Availability | Geometry | Geometry − availability (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| R→R | frame-bbox missing | 0/0/0; — | — | — | — | — |
| R→R | frame-bbox valid | 78/1118/41488; MSWC | 52.84 ± 1.04 | 51.75 ± 1.35 | 52.18 ± 1.55 | 0.43 ± 1.74 |
| R→R | bbox-velocity missing | 27/78/84; MSWC | 52.79 ± 6.86 | 51.97 ± 8.11 | 53.53 ± 6.11 | 1.56 ± 11.19 |
| R→R | bbox-velocity valid | 78/1111/41404; MSWC | 52.98 ± 1.04 | 51.94 ± 1.38 | 52.27 ± 1.58 | 0.33 ± 1.77 |
| R→R | track-bbox never | 0/0/0; — | — | — | — | — |
| R→R | track-bbox partial | 78/1028/35359; MSWC | 53.06 ± 0.91 | 51.56 ± 1.05 | 51.98 ± 1.32 | 0.41 ± 1.21 |
| R→R | track-bbox complete | 28/90/6129; MSWC | 50.08 ± 3.30 | 51.34 ± 4.26 | 50.75 ± 3.65 | -0.59 ± 6.17 |
| R→L | frame-bbox missing | 93/1813/38236; MSWC | 44.98 ± 0.96 | 43.71 ± 2.41 | 43.75 ± 1.53 | 0.04 ± 3.50 |
| R→L | frame-bbox valid | 89/1386/31031; MSWC | 52.00 ± 0.64 | 52.91 ± 1.15 | 49.93 ± 1.28 | -2.99 ± 2.27 |
| R→L | bbox-velocity missing | 93/1813/38517; MSWC | 44.99 ± 0.96 | 43.74 ± 2.41 | 43.75 ± 1.52 | 0.02 ± 3.50 |
| R→L | bbox-velocity valid | 89/1352/30750; MSWC | 52.21 ± 0.61 | 53.26 ± 1.11 | 50.00 ± 1.31 | -3.26 ± 2.23 |
| R→L | track-bbox never | 81/534/14255; MSWC | 38.33 ± 0.80 | 34.27 ± 2.56 | 35.65 ± 2.18 | 1.38 ± 4.56 |
| R→L | track-bbox partial | 89/1281/52861; MSWC | 49.73 ± 0.96 | 50.13 ± 1.45 | 48.12 ± 0.96 | -2.01 ± 2.28 |
| R→L | track-bbox complete | 50/114/2151; MSWC | 46.86 ± 3.92 | 51.50 ± 3.97 | 47.45 ± 3.87 | -4.06 ± 5.10 |
| L→L | frame-bbox missing | 93/1813/38236; MSWC | 58.77 ± 1.97 | 58.01 ± 2.14 | 59.63 ± 1.63 | 1.61 ± 3.09 |
| L→L | frame-bbox valid | 89/1386/31031; MSWC | 62.97 ± 2.63 | 63.08 ± 1.75 | 64.62 ± 2.04 | 1.54 ± 3.45 |
| L→L | bbox-velocity missing | 93/1813/38517; MSWC | 58.78 ± 1.97 | 58.03 ± 2.13 | 59.66 ± 1.65 | 1.63 ± 3.09 |
| L→L | bbox-velocity valid | 89/1352/30750; MSWC | 63.17 ± 2.71 | 63.34 ± 1.74 | 64.81 ± 1.95 | 1.47 ± 3.34 |
| L→L | track-bbox never | 81/534/14255; MSWC | 58.60 ± 1.45 | 59.05 ± 1.82 | 57.57 ± 2.77 | -1.47 ± 1.69 |
| L→L | track-bbox partial | 89/1281/52861; MSWC | 61.12 ± 2.28 | 60.79 ± 1.73 | 62.52 ± 1.26 | 1.74 ± 2.73 |
| L→L | track-bbox complete | 50/114/2151; MSWC | 60.46 ± 3.46 | 65.04 ± 5.86 | 60.13 ± 6.42 | -4.91 ± 7.55 |
| L→R | frame-bbox missing | 0/0/0; — | — | — | — | — |
| L→R | frame-bbox valid | 78/1118/41488; MSWC | 42.22 ± 1.59 | 42.02 ± 1.08 | 47.53 ± 0.91 | 5.51 ± 0.50 |
| L→R | bbox-velocity missing | 27/78/84; MSWC | 40.66 ± 5.85 | 41.19 ± 3.67 | 39.28 ± 3.91 | -1.91 ± 5.03 |
| L→R | bbox-velocity valid | 78/1111/41404; MSWC | 42.37 ± 1.57 | 42.11 ± 1.14 | 47.78 ± 0.91 | 5.67 ± 0.57 |
| L→R | track-bbox never | 0/0/0; — | — | — | — | — |
| L→R | track-bbox partial | 78/1028/35359; MSWC | 42.22 ± 1.67 | 42.10 ± 1.23 | 47.73 ± 0.89 | 5.63 ± 0.64 |
| L→R | track-bbox complete | 28/90/6129; MSWC | 40.16 ± 2.69 | 38.99 ± 1.06 | 42.56 ± 2.46 | 3.57 ± 3.41 |

- **Coverage is asymmetric:** every ROAD test GT frame has a usable native bbox. ROAD has no missing-box or never-box GT slice; LOKI has 38,236 missing-box GT frames and 534 never-box tracks. This review cannot measure missing-box ROAD behavior performance.
- **R→L geometry loses on observed boxes:** −2.99 ± 2.27 pp versus availability on bbox-valid frames, versus approximately zero change on missing-box frames. Never-box track performance remains low (35.65%) relative to the original baseline (38.33%).
- **L→L has no uniform geometry benefit:** partial-box tracks gain +1.74 ± 2.73 pp versus availability, while never-box tracks lose −1.47 ± 1.69 pp. The 114 complete-box tracks show −4.91 ± 7.55 pp, with substantial variability.
- **L→R gain is supported mainly by velocity-valid and partial-box observations.** Only 84 ROAD GT frames lack bbox velocity; their −1.91 ± 5.03 pp comparison is small-support evidence. Missing-box frames in a BiLSTM can use bbox context elsewhere in their track; never-box tracks cannot use their own bbox geometry.

### Geometry minus availability inside other key slices

These are paired mean ± SD differences in pp, using the same support shown in the ten-axis table. All bins, individual seed differences and positive-seed counts are in [stratum differences](../../outputs/experiments/E001b/analysis/20261010-seed-review/stratum-differences.csv).

| Axis/bin | R→R | R→L | L→L | L→R |
| --- | --- | --- | --- | --- |
| duration [5,10) | -0.51 ± 3.40 | 0.48 ± 2.35 | 2.14 ± 4.43 | -3.09 ± 3.04 |
| duration [15,inf) | -0.03 ± 1.77 | -1.52 ± 5.06 | 0.59 ± 2.62 | 8.15 ± 1.14 |
| position-coverage [0,.25) | 6.10 ± 8.28 | -2.40 ± 1.48 | -0.11 ± 0.31 | -4.49 ± 7.33 |
| position-coverage [.75,1] | 0.06 ± 1.25 | -0.78 ± 2.87 | 1.20 ± 2.49 | 6.52 ± 0.87 |
| position missing | 0.55 ± 4.72 | — | — | 3.67 ± 4.71 |
| position valid | 0.48 ± 1.39 | -0.88 ± 2.84 | 0.82 ± 2.47 | 5.10 ± 0.73 |
| range [20,40) | -0.87 ± 2.10 | -0.35 ± 3.83 | 1.06 ± 4.01 | 6.72 ± 1.89 |
| range [40,inf) | 1.50 ± 1.69 | 2.15 ± 1.82 | 2.73 ± 2.08 | 4.59 ± 1.23 |
| transition near | 0.24 ± 1.76 | -0.27 ± 1.06 | 1.73 ± 4.06 | -1.15 ± 3.27 |
| transition steady | 0.34 ± 1.80 | -0.88 ± 3.05 | 0.85 ± 2.45 | 5.72 ± 0.55 |

L→R gains are concentrated in **longer tracks and good 3D coverage**, not uniformly in difficult/missing-input observations. Tracks ≥15 s gain +8.15 ± 1.14 pp, while 5–10 s tracks average −3.09 ± 3.04 pp. The 27 low-position-coverage ROAD tracks average −4.49 ± 7.33 pp. These retrospective strata do not identify a causal mechanism.

Internal-gap conclusions are particularly limited: LOKI has no scored internal-position-gap tracks, and ROAD has only **13 tracks / four clips / 498 GT frames** with gaps ≤1 s, supporting M/S/W only. Neither test population supports gaps >1 s. The apparently high ROAD <10 m F1 averages only M/S/C and uses 23 tracks, so it cannot establish that four-class performance is better at close range.

### What “transition F1” currently measures

The existing `transition` axis measures **frame-state F1 near GT boundaries**, not transition detection, onset timing, or an event such as Moving→Crossing. Near = nominal grid time within 0.4 s of an adjacent different-label midpoint, inside one continuously labeled run. Steady = other labeled frames in runs of at least two observations; a steady frame need not belong to a completely constant-label track. Unknown = singleton labeled runs. GT gaps are never bridged.

Only **504 ROAD frames / 117 tracks** and **1,675 LOKI frames / 332 tracks** are near boundaries (1.21% / 2.42% of test GT). All four classes are supported there. Unknown LOKI has only 20 singleton GT frames and no Crossing support.

Below is the class F1 underlying the near-boundary aggregate. These are state classes, not named transition types.

| Direction | BiLSTM input | Moving | Stopped | Waiting | Crossing |
| --- | --- | --- | --- | --- | --- |
| R→R | RAW | 52.57 ± 2.12 | 34.65 ± 6.32 | 27.23 ± 10.30 | 36.17 ± 2.11 |
| R→R | K+T+R | 51.68 ± 1.75 | 37.81 ± 3.09 | 36.06 ± 4.87 | 38.95 ± 2.60 |
| R→R | availability | 50.38 ± 1.47 | 38.85 ± 5.91 | 36.77 ± 7.81 | 37.05 ± 0.94 |
| R→R | geometry | 50.57 ± 0.42 | 36.79 ± 8.26 | 38.21 ± 3.78 | 38.44 ± 1.11 |
| R→L | RAW | 54.70 ± 0.89 | 33.85 ± 3.66 | 28.21 ± 4.74 | 43.31 ± 3.03 |
| R→L | K+T+R | 51.78 ± 1.50 | 34.13 ± 3.80 | 34.37 ± 3.71 | 45.06 ± 1.99 |
| R→L | availability | 53.71 ± 1.51 | 31.54 ± 3.31 | 29.61 ± 3.47 | 43.93 ± 1.35 |
| R→L | geometry | 52.51 ± 0.61 | 30.19 ± 0.97 | 29.18 ± 3.18 | 45.80 ± 1.96 |
| L→L | RAW | 55.67 ± 1.32 | 17.08 ± 5.01 | 33.91 ± 3.00 | 46.34 ± 2.53 |
| L→L | K+T+R | 54.82 ± 1.83 | 16.71 ± 2.75 | 40.76 ± 2.56 | 46.84 ± 2.26 |
| L→L | availability | 55.95 ± 1.13 | 16.74 ± 4.55 | 38.28 ± 4.67 | 45.32 ± 2.11 |
| L→L | geometry | 55.00 ± 1.03 | 20.74 ± 4.53 | 40.09 ± 6.10 | 47.36 ± 1.68 |
| L→R | RAW | 48.25 ± 2.12 | 16.92 ± 10.95 | 30.30 ± 5.76 | 37.77 ± 3.22 |
| L→R | K+T+R | 47.80 ± 1.61 | 12.66 ± 9.55 | 33.97 ± 1.88 | 39.78 ± 1.82 |
| L→R | availability | 46.45 ± 2.58 | 10.79 ± 11.08 | 31.50 ± 3.27 | 40.93 ± 2.76 |
| L→R | geometry | 49.06 ± 1.24 | 15.10 ± 5.43 | 23.59 ± 5.52 | 37.34 ± 3.12 |

The main L→R geometry gain is in **steady** states (+5.72 ± 0.55 pp), not near boundaries (−1.15 ± 3.27 pp). Near-boundary Stopped remains difficult; L→L reference near macro-F1 is 39.79%, versus 61.70% steady. This identifies a diagnostic to inspect, not a measured transition-timing failure. Specific transition/event F1 would require a separately agreed event definition, matching/tolerance and gap/aggregation policy; it was excluded from E001/E001b and was not added here.

Updated thirteen-axis plots compare K+T+R, geometry and all three E001c inputs, with support and classes: [R→R](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/strata-r-r.png), [R→L](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/strata-r-l.png), [L→L](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/strata-l-l.png), [L→R](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/strata-l-r.png); SVG exports: [R→R](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/strata-r-r.svg), [R→L](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/strata-r-l.svg), [L→L](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/strata-l-l.svg), [L→R](../../outputs/experiments/E001b/analysis/20261015-seed-review/plots/strata-l-r.svg).

## Artifacts and W&B navigation

The [manual E001 workspace](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling?nw=71j6lc3jv4g) was read back on 2026-10-10. Expand **Final evaluation · loki / road-waymo** to find **Track-weighted confusion (row-normalized)** and class results. Both sections are collapsed by default. The plot key is `details/plots/<dataset>-track_weighted-row-normalized`; the numerical table key is `details/<dataset>/track_weighted/confusion`. Raw/count confusion views also exist under `details/plots/` and remain local.

The saved workspace contains the original **ten** observation axes; E001b additionally logs `strata/<dataset>/frame-bbox`, `bbox-velocity`, and `track-bbox`, which are **not present in that saved layout**. Its transition panel is `strata/<dataset>/transition` with near/steady/unknown, exactly as defined above. Run-summary/file readback confirmed both dataset confusion images/tables and transition plots on four representative completed source runs, with primary scores matching local results:

| Source / input, seed 0 | Existing W&B run |
| --- | --- |
| ROAD K+T+R | [E001 run](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/128ws5z9) |
| LOKI K+T+R | [E001 run](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/l37fygme) |
| ROAD geometry | [E001b run](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/7xy86qez) |
| LOKI geometry | [E001b run](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/nwrxyqmd) |

All **130 W&B URLs, remote attempt paths, revisions, selected epochs and source-validation scores** are in [combined run inventory](../../outputs/experiments/E001b/analysis/20261015-seed-review/runs.csv). The workspace/readback observations above remain dated 2026-10-10; E001c dashboards were not re-inspected. Remote originals are beneath `/home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/`:

- `E001/runs/seeds-20261009-084417-412679/`.
- `E001b/runs/seeds-20261009-140118-403993/`.
- `E001c/runs/seeds-20261015-114954-843530/`.

Each attempt contains original configuration/provenance/history/checkpoints and both `<dataset>-test/{metrics.json,predictions.npz}`. This review downloaded numerical aggregate metadata, retained its hashes and read the predictions remotely; checkpoints/full predictions remain on Aalto.

The ignored local review directory is `outputs/experiments/E001b/analysis/20261010-seed-review/`. Artifact links resolve in this checkout; these numerical files/plots are outside Git and need separate backup. It retains:

- [Per-seed primary metrics](../../outputs/experiments/E001b/analysis/20261010-seed-review/primary.csv), [complete class metrics](../../outputs/experiments/E001b/analysis/20261010-seed-review/per-class.csv), [original per-seed strata](../../outputs/experiments/E001b/analysis/20261010-seed-review/strata.csv) and all summary/difference files linked above. The additional baseline bbox slices are retained in the prediction audit and complete `strata-summary.csv`; they do not overwrite original metrics.
- [Metadata/settings/selection verification](../../outputs/experiments/E001b/analysis/20261010-seed-review/verification.json) and [saved-prediction reconstruction and file hashes](../../outputs/experiments/E001b/analysis/20261010-seed-review/prediction-verification.json): **all 200 primary evaluations reconstructed exactly**, all 200 identity/target/original-condition checks passed, 700 downloaded metadata files match remote hashes, eight representative E001b evaluations reconstructed all thirteen axes, and twenty baseline evaluations gained matching bbox slices offline. Aggregates in every stored primary/slice confusion were independently recalculated.
- [W&B readback](../../outputs/experiments/E001b/analysis/20261010-seed-review/wandb-readback.json) and [workspace readback](../../outputs/experiments/E001b/analysis/20261010-seed-review/workspace-readback.json). No logging, uploads or dashboard changes were performed during this review.
- [Analysis/verification scripts](../../outputs/experiments/E001b/analysis/20261010-seed-review/analyze.py), [remote prediction audit](../../outputs/experiments/E001b/analysis/20261010-seed-review/verify-predictions-aalto.py), [table/plot regeneration](../../outputs/experiments/E001b/analysis/20261010-seed-review/report-tables.py) and [report generation](../../outputs/experiments/E001b/analysis/20261010-seed-review/write-report.py). The earlier audit cached each NPZ array once. The E001c extension uses the shared reader subsequently optimized for the prediction viewer.
- [CPU acceptance log](../../outputs/experiments/E001b/analysis/20261010-seed-review/acceptance-cpu.log): 56 tests, 52 passes / four CUDA skips. Lock/environment checks and local documentation links/anchors passed; ten PNG/SVG plot pairs were exported and inspected for file completeness.

### E001c extension evidence (2026-10-15)

The earlier `20261010-seed-review/` evidence remains unchanged. The combined snapshot is `outputs/experiments/E001b/analysis/20261015-seed-review/` on both machines. Checkpoints/full predictions remain on Aalto; only metadata, numerical summaries and plots were copied locally.

- [Prediction audit](../../outputs/experiments/E001b/analysis/20261015-seed-review/prediction-verification.json): all **60 new primary evaluations** reconstructed exactly; all 60 identity/time/GT/13-condition checks match the E001b reference, including bbox validity/velocity masks. All 13 axes reconstructed from logits for **12 seed-0 cells** (three configurations × two sources × two tests). The **200 original prediction hashes and 700 original metadata hashes** still match the prior audit; they were not re-inferred. All stored primary/slice confusion-derived metrics were independently recalculated.
- [Protocol/selection verification](../../outputs/experiments/E001b/analysis/20261015-seed-review/verification.json): shared settings, frozen setup/collection/policy references and populations agree; E001c's three configurations use the same source-training box moments, checked against frozen cache-manifest hashes. Source-validation selection/patience was reconstructed from histories. No run reached the epoch cap. The old **40 summary rows / 200 seed scores** remain exactly unchanged; [table/export checks](../../outputs/experiments/E001b/analysis/20261015-seed-review/review-verification.json) retain exact comparisons and PNG/SVG hashes.
- [Numerical inventory](../../outputs/experiments/E001b/analysis/20261015-seed-review/runs.csv), primary/class/secondary/stratum summaries, paired and per-stratum differences, source metadata and all ten new **PNG/SVG pairs** are retained under the combined snapshot. The [tracked analysis script](../../scripts/review-e001-results.py) collects/audits through `--collect --prior-verification <prior audit> --output <fresh directory>` on Aalto, then regenerates tables/plots with `--output <snapshot>` locally. Full source/target/run provenance is in the metadata snapshot.
- [CPU acceptance](../../outputs/experiments/E001b/analysis/20261015-seed-review/acceptance-cpu.log): 78 tests, five CUDA skips; the added hand-calculated test checks sample SD, same-seed subtraction and empty slices. Plot pairs were exported and representative primary/difference/confusion/stratum figures visually inspected. No new GPU inference or W&B API/logging was needed for this extension.

No training, native reprocessing, checkpoint selection, event scoring or W&B layout changes occurred. No fresh GPU acceptance suite or model-inference replay was run; reconstruction used saved logits. Separate backup storage remains TBD.

## Next inspection

Use the [prediction viewer](../../README.md#inspect-model-predictions) to inspect a small, deliberately selected set: ROAD→LOKI Stopped→Moving failures for geometry versus boxes; LOKI→ROAD Stopped gains versus Waiting/Crossing losses; and near-boundary errors. Retain both improvements and regressions, rather than picking a favorable seed. Preserve identities, source direction, GT support and selection reasons. Use that evidence to define the next bounded comparison; these held-out diagnostics must not retrospectively select a seed or checkpoint. No additional training is recommended solely because of the 75-epoch ceiling. E001c provides no basis for promoting size-velocity inputs; any later ablation must isolate a particular representation change before attributing a mechanism.
