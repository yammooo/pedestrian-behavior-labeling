# E004 — Target-label efficiency

**Study direction agreed 2026-11-12; E004a complete; E004b class-coverage diagnostic agreed, later comparisons remain open.** E004 asks whether source behavior training reduces the target annotation needed for offline pedestrian labeling, which components need adaptation, and what automatic-label coverage is supported at a declared error tolerance.

## Scope and motivation

[E003b](../E003b-target-head-adaptation/README.md#results), [E003c](../E003c-temporal-adaptation/README.md#full-source-initialized-results-2026-11-06) and [E002f](../E002f-temporal-architecture/README.md#full-results-2026-11-11) show substantial recovery after full-target-label adaptation. They do not establish label efficiency. E004 moves from architecture/regularization diagnostics to annotation-budget curves. The candidate contribution is empirical evidence, not a claim that head adaptation or selective prediction is a new method; see the [literature overview](../../docs/literature/README.md).

The task, ontology, frozen populations, group splits, natural missingness and full-track inference contract stay unchanged. Use one fixed architecture across methods. DINO and D-DITR remain externally pretrained and frozen; their exposure is retained from E002. Target-only training excludes source **behavior training**, not external encoder pretraining. Preserve prior target-validation development and E001 test inspection in reporting.

| Comparison | Target-trained components | Purpose / status |
|---|---|---|
| Original source model | None | Zero-training-label reference; development-label access is accounted separately |
| [E004a: head-only adaptation](../E004a-head-label-budget/README.md) | Original source classifier only | Completed ROAD→LOKI pilot: [results and budget figures](../E004a-head-label-budget/README.md#results) |
| [E004b: class-covered head budgets](../E004b-class-covered-head-budget/README.md) | Original source classifier only | Four initial class-support tracks, ordinary remainder; privileged screening metadata disclosed |
| Temporal + head adaptation | Source temporal block, final norm and classifier | Planned comparison; detailed settings/ID TBD |
| Target-only training | Randomly initialized task branches, spatial readout, fusion, temporal block and head | Essential control for annotation-savings claims; same pretrained encoders and target subsets |
| Full task-model adaptation | All source-trained task components above | Optional follow-up if justified; does not mean encoder fine-tuning |

The completed E004a pilot finds diminishing F1 gains beyond a few hundred fitting tracks under its fixed recipe, while simple confidence cutoffs exceed their calibration error targets on assessment. These results motivate matched target-only/adaptation-depth comparisons and caution about automatic-label reliability; they do not establish annotation savings or guarantees. The experiment owner retains the numerical evidence and qualifiers.

## Budget and comparison rules

- First budget unit: complete labeled pedestrian tracks. Report unique recordings, tracks, accepted GT timesteps and labeled duration; one track is not one frame label or a fixed amount of human effort.
- Prefer absolute counts. E004a fixes absolute track counts, five nested recording-diverse draws and a scenario-disjoint development/calibration–assessment split. Earlier percentage/complete-scenario budget proposals were provisional and are superseded for this pilot.
- Give methods the same labeled subsets and permitted target observations. Nest increasing budgets within each draw, repeat subset selection and distinguish draw variation from optimization-seed variation. Report absent classes and retain negative transfer; no undisclosed GT-balanced sampling.
- Freeze initialization, normalization, trainable components, fit/selection budgets and source checkpoint before fitting. Source normalization stays frozen for source-model adaptation; the target-only normalization/access rule is TBD.
- Count fitting, validation/stopping, tuning, subset selection and later threshold-calibration labels. Show fixed development overhead and total unique target-label access; do not describe a tiny training subset as total annotation cost.
- Main output: four-class track-weighted macro-F1 versus label budget, with per-class results, CE, support and measured compute. Compare labels required at matched performance only where curves support it. Head-only curves alone do not establish savings over target-only training.

## Secondary endpoint: error versus automatic coverage

Define this before interpreting automatic-annotation usefulness, but do not make it a prerequisite for the E004a fitting pilot. E004a fixes individual scored timesteps as its acceptance unit; this does not imply whole-track acceptance.

For prediction `y_hat[t] = argmax p[t]`, E004a's baseline confidence score is `s[t] = max p[t]`. Accept when `s[t] >= threshold`. Neither the GT label nor `p[t, GT[t]]` may enter this acceptance rule: GT is unavailable during annotation. GT measures the accepted predictions' actual error on labeled calibration/evaluation data.

For scored GT timesteps, let `w[t] = 1 / track_GT_count` and `a[t]` indicate confidence acceptance. E004a uses these weighted measures:

```text
coverage = sum(w * a) / sum(w)
accepted error = sum(w * a * (y_hat != GT)) / sum(w * a)
```

Keep weights from the original complete track; do not renormalize each track after acceptance. No accepted weight means coverage zero and error undefined. E004a uses these track-weighted measures as primary, with raw frame-count versions alongside them; other comparisons must declare changes explicitly. Coverage over all non-padding, potentially unlabeled context is a separate acceptance fraction: without GT, its error cannot be measured.

Select a threshold using permitted labeled calibration/development data, then freeze it and assess its achieved coverage/error on separate evaluation groups not used for fitting, checkpoint selection or threshold tuning. Plotting the full risk–coverage curve is descriptive; choosing the best threshold and claiming its error on the same observations is optimistic. A softmax threshold of 0.95 does not guarantee 5% error. Report support, uncertainty, GT-class retention and accepted error by predicted class, plus missing-observation cohorts; high aggregate coverage/accuracy can hide rejection or failure of rare behaviors. See [SelectiveNet notes](../../docs/literature/papers/selectivenet.md).

### Worked example: confidence is not correctness

Five equally weighted scored timesteps:

| GT | Prediction | Predicted-class confidence | Correct? |
|---|---|---:|---|
| Moving | Moving | 0.99 | Yes |
| Stopped | Stopped | 0.97 | Yes |
| Waiting | Stopped | 0.96 | No |
| Crossing | Crossing | 0.80 | Yes |
| Moving | Moving | 0.55 | Yes |

At threshold 0.95, three predictions are accepted: coverage 60%, observed accepted error 1/3 =33.3%. At threshold 0.97, two are accepted: coverage 40%, observed error 0/2. The model can be confidently wrong, and zero observed errors in two cases do not establish zero underlying risk. The threshold is a score cutoff, not the tolerated error itself. Use calibration labels to choose a candidate cutoff for the desired error, then measure its achieved error/coverage independently.

**E004a reporting fixed:** budget–F1 and budget–coverage figures; track-weighted timestep error/coverage; 1%, 5%, 10%, 20% calibration error targets; largest qualifying calibration accepted set and separate achieved-error reporting. Its [reporting specification](../E004a-head-label-budget/README.md#fixed-reporting-specification-2026-11-12) owns details. **E004a assessment:** a fixed 46/47-scenario split of existing LOKI validation, with fitting/development label access and assessment effort separately counted. **TBD before reliability guarantees:** uncertainty/formal risk bounds accounting for correlated observations and assessment beyond historically inspected development data. No arbitrary minimum-support filter is imposed for the descriptive pilot. These are observed benchmark coverage/error estimates unless a separate guarantee is justified. Calibration/audit label access is additional annotation cost. Human hours saved are not established by benchmark label counts.

## Ownership and current boundary

E004 owns common comparison/accounting and coverage concepts. [E004a](../E004a-head-label-budget/README.md) owns the first head-only protocol. E004a is implemented over the existing frozen features; no source-model training, test preparation or encoder changes are included. All46 E004a fits are complete. No later comparison was launched by the results review.

On 2026-11-12 the user assigned E004 to label efficiency. The earlier unexecuted E004 complementary-supervision reservation is superseded; that idea remains deferred in [research history](../../docs/research.md#tentative-experiment-direction). No past run or result was renamed.
