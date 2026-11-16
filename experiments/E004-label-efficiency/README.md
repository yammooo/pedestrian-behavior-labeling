# E004 — Target-label efficiency

**Study direction agreed 2026-11-12; E004a original/early-check pilots complete; E004b complete; later comparisons remain open.** E004 asks whether source behavior training reduces the target annotation needed for offline pedestrian labeling, which components need adaptation, and what automatic-label coverage is supported at a declared error tolerance.

## Scope and motivation

[E003b](../E003b-target-head-adaptation/README.md#results), [E003c](../E003c-temporal-adaptation/README.md#full-source-initialized-results-2026-11-06) and [E002f](../E002f-temporal-architecture/README.md#full-results-2026-11-11) show substantial recovery after full-target-label adaptation. They do not establish label efficiency. E004 moves from architecture/regularization diagnostics to annotation-budget curves. The candidate contribution is empirical evidence, not a claim that head adaptation or selective prediction is a new method; see the [literature overview](../../docs/literature/README.md).

The task, ontology, frozen populations, group splits, natural missingness and full-track inference contract stay unchanged. Use one fixed architecture across methods. DINO and D-DITR remain externally pretrained and frozen; their exposure is retained from E002. Target-only training excludes source **behavior training**, not external encoder pretraining. Preserve prior target-validation development and E001 test inspection in reporting.

| Comparison | Target-trained components | Purpose / status |
|---|---|---|
| Original source model | None | Zero-training-label reference; development-label access is accounted separately |
| [E004a: head-only adaptation](../E004a-head-label-budget/README.md) | Original source classifier only | Completed original/early-check ROAD→LOKI pilots: [paired results and budget figures](../E004a-head-label-budget/README.md#results) |
| [E004b: class-covered head budgets](../E004b-class-covered-head-budget/README.md) | Original source classifier only | [Completed](../E004b-class-covered-head-budget/README.md#results): no consistent mean gain from four initial class-support tracks; privileged screening access disclosed |
| [E004c: temporal + head](../E004c-adaptation-depth-budgets/README.md) | Source temporal block, final norm and classifier | Prepared 25/100/1,000-track matched comparisons; CPU/CUDA checked, user-launched |
| [E004c: target-only](../E004c-adaptation-depth-budgets/README.md) | Randomly initialized task branches, spatial readout, fusion, temporal block and head | Same selected target labels and frozen pretrained encoders; subset-only normalization |
| Full task-model adaptation | All source-trained task components above | Optional follow-up if justified; does not mean encoder fine-tuning |

The E004a early-check follow-up captures 11 earlier checkpoints without a broad assessment gain; from 250 tracks onward results match the original pilot. Both find diminishing F1 gains beyond a few hundred fitting tracks under its fixed recipe, while simple confidence cutoffs exceed their calibration error targets on assessment. These results motivate matched target-only/adaptation-depth comparisons and caution about automatic-label reliability; they do not establish annotation savings or guarantees. The experiment owner retains the numerical evidence and qualifiers.

## Budget and comparison rules

- First budget unit: complete labeled pedestrian tracks. Report unique recordings, tracks, accepted GT timesteps and labeled duration; one track is not one frame label or a fixed amount of human effort.
- Prefer absolute counts. E004a fixes absolute track counts, five nested recording-diverse draws and a scenario-disjoint development/calibration–assessment split. Earlier percentage/complete-scenario budget proposals were provisional and are superseded for this pilot.
- Give methods the same labeled subsets and permitted target observations. Nest increasing budgets within each draw, repeat subset selection and distinguish draw variation from optimization-seed variation. Report absent classes and retain negative transfer; no undisclosed GT-balanced sampling.
- Freeze initialization, normalization, trainable components, fit/selection budgets and source checkpoint before fitting. Source normalization stays frozen for source-model adaptation; target-only fits normalization only on its selected training tracks in E004c.
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

E004 owns common comparison/accounting and coverage concepts. [E004a](../E004a-head-label-budget/README.md) owns the first head-only protocol. E004a is implemented over the existing frozen features; no source-model training, test preparation or encoder changes are included. Both 46-fit E004a pilots are complete. E004b is complete; its minimum class-support floor gives no consistent mean F1 gain under privileged GT-informed sampling. No new comparison was launched by the results review.

On 2026-11-12 the user assigned E004 to label efficiency. The earlier unexecuted E004 complementary-supervision reservation is superseded; that idea remains deferred in [research history](../../docs/research.md#tentative-experiment-direction). No past run or result was renamed.


## Track-difficulty diagnostic (2026-11-16)

Saved predictions compare the same **931 LOKI tracks /47 scenarios /30,605 GT timesteps**. All three models use the E002d geometry + RGB + D-DITR architecture with two temporal Transformer layers. The head-only adaptation is [E003c `head`](../E003c-temporal-adaptation/README.md#full-source-initialized-results-2026-11-06), initialized from the original ROAD classifier and fitted with all target-training labels. The target-trained reference is the existing E002d LOKI run. These are **not the E004 BiLSTM budget fits**; the optional 25-track comparison is omitted to keep architectures matched.

The track identities are E004's assessment half, but the adapted and LOKI-trained checkpoints were selected using the original full LOKI validation set, including these tracks. This is descriptive analysis of previously inspected validation data, not independent assessment. No training, encoder inference, confidence filtering or test access was performed.

| Model on common LOKI subset | Track-weighted macro-F1 (%) | Track-weighted error (%) |
|---|---:|---:|
| ROAD source, direct |44.04|48.92|
| ROAD body + full-label target head |60.81|21.06|
| LOKI-trained task model |71.25|16.93|

Error is `1 - accuracy`, not `1 - macro-F1`. Each scored timestep retains weight `1 / complete-track GT count`, including within slices. Within-GT-class error below is `1 - recall`; it does not measure class precision.

| GT class | ROAD direct error (%) | Adapted-head error (%) | LOKI-trained error (%) |
|---|---:|---:|---:|
| Moving |49.0|6.1|9.3|
| Stopped |24.5|54.7|45.3|
| Waiting to cross |62.6|38.1|30.3|
| Crossing |53.5|69.5|31.3|

### What the conditions reveal

- **No general distance rule:** aggregate error after adaptation is fairly flat across distance bins. Within GT Crossing, however, LOKI-trained error rises from **15.7% below 10 m to 49.8% at 40 m or beyond** (695 /1,649 GT timesteps); direct-transfer error rises 18.0%→64.0%. Distance is a timestep condition, not each track's median distance.
- **RGB availability is class-specific:** among GT Crossing in tracks with usable RGB in less than 25% of slots, direct/head/LOKI-trained error is **82.5/92.8/42.9%**, versus **14.0/36.6/9.8%** with at least 75% usable RGB (2,161 /1,817 GT timesteps). For low-RGB Moving, head adaptation instead reduces error 62.8%→3.1%. Availability therefore does not define one uniformly difficult population, and these associations do not establish a causal RGB benefit.
- **Very short tracks are difficult before transfer adaptation:** the 89 tracks shorter than 1 s have 74.0% direct error, 23.5% head-adapted error and 25.6% LOKI-trained error. After adaptation, longer duration does not consistently imply lower overall error. Removing short tracks alone would not resolve the remaining breakdown.
- **Transitions are only a small part of total error:** near-transition frames contribute **1.95/3.68/4.45%** of all weighted errors for direct/head/LOKI-trained models. Some within-class near-transition error rates are high, but most errors lie elsewhere. Stopped-near has only 29 GT timesteps across 15 tracks /8 scenarios; do not generalize from its rate.
- **LiDAR missingness has little support here:** 906 of 931 tracks have usable LiDAR in at least 75% of their slots. Only 3 tracks fall below 25%; those bins cannot support a general reliability claim.

Duration uses saved native timestamp extent, not labeled duration or a guarantee of continuous observations. RGB/LiDAR fractions use actual usable model-input flags over every unpadded track slot, including unlabeled context. Transition proximity uses the existing 0.4 s rule without bridging GT gaps. Counts, scenario support, within-class condition rates and each slice's contribution to total errors are retained in numerical files.

![Error by duration, distance and usable modality fractions](../../outputs/experiments/E004/track-difficulty-20261116/difficulty.png)

[SVG](../../outputs/experiments/E004/track-difficulty-20261116/difficulty.svg).

![Within-class error near and away from GT transitions](../../outputs/experiments/E004/track-difficulty-20261116/class-transition.png)

[SVG](../../outputs/experiments/E004/track-difficulty-20261116/class-transition.svg).

**Interpretation:** this supports investigating class-specific observation failures and the head's trade-offs before imposing a broad distance/duration eligibility rule. It does not identify the cause of transfer failure or justify changing the population yet. Source-validation context is 66.07% ROAD F1 and 74.68% full-LOKI F1 on their respective original populations; those are not paired scores on the 931-track subset.

[Analysis script](../../scripts/analyze-e004-track-difficulty.py) · [summary JSON](../../outputs/experiments/E004/track-difficulty-20261116/summary.json) · [subgroup CSV](../../outputs/experiments/E004/track-difficulty-20261116/summary.csv) · [per-track CSV](../../outputs/experiments/E004/track-difficulty-20261116/track-errors.csv). Input paths and checkpoint families are fixed in the script. Conditions come from the ROAD-direct archive; optional metadata axes differ across older exports. Track IDs, lengths, GT and masks align; timestamps match across the three model archives, while the E004 identity archive omits timestamps. All condition/class partitions reconstruct the complete weighted support and errors. The focused weight self-check passes; the CPU suite completed 294 tests with 13 expected skips, with its sandbox-blocked localhost test passing on isolated rerun outside the sandbox.


### Error concentration and GT composition

On the same 931 tracks, the worst 187 tracks (20%, rounded up) account for 41.1% of direct-transfer weighted errors, 81.3% after head adaptation and 90.3% for the LOKI-trained model. These are shares of each model's total errors, not those tracks' error rates. Direct transfer has 319 entirely wrong tracks: any 187 of them give the same 41.1% error share, so its exact worst 20% class composition is not uniquely defined.

Counts below distinguish single-class tracks (all scored GT has that class) from mixed-class tracks; unlabeled slots do not establish constant behavior. The complete population has 572 Moving-only, 67 Stopped-only, 75 Waiting-only, 64 Crossing-only and 153 mixed-class tracks.

| Entirely correct tracks | Moving-only | Stopped-only | Waiting-only | Crossing-only | Mixed | Total |
|---|---:|---:|---:|---:|---:|---:|
| ROAD direct |226|48|12|18|0|304|
| Adapted head |530|12|23|2|0|567|
| LOKI-trained |515|32|48|33|2|630|

| Hard tracks | Moving-only | Stopped-only | Waiting-only | Crossing-only | Mixed | Total |
|---|---:|---:|---:|---:|---:|---:|
| ROAD direct: all tied entirely wrong tracks |198|10|40|42|29|319|
| Adapted head: worst 20% |23|36|29|53|46|187|
| LOKI-trained: worst 20% |45|30|25|26|61|187|

Perfect head-adapted tracks are 93.5% Moving-only, versus 81.7% for the LOKI-trained model. Only 2/153 mixed tracks are entirely correct even with target training; their mean within-track error is 26.5%, versus 16.9% across all tracks. Perfect-track accuracy is stricter for longer, mixed tracks, so this does not establish that transition errors dominate. The composition qualifies the earlier concentration result: much of the perfect-track count comes from the common Moving-only population, and rare/mixed behaviors are overrepresented in the hard cohorts. These cohorts were selected retrospectively using GT and are not deployable rejection rules.

[Numerical breakdown, including boundary ties and track-weighted GT class proportions](../../outputs/experiments/E004/track-difficulty-20261116/track-error-concentration.json).


### Track quality by error range

The same 931 tracks are grouped by their fraction of misclassified accepted-GT timesteps: **0–5%, >5–20%, >20–50%, >50–80%, >80–95%, >95–100%**. Boundaries are right-inclusive; these groups partition all tracks. Each track counts once. A single error in a five-GT-frame track is 20%, so labeled-frame support is reported alongside duration.

The first figure separates single-GT-class and mixed-class tracks; each bar sums to 100% within that track type. This prevents the common Moving-only tracks from dominating every profile.

| Tracks with at most 20% error (%) | All (931) | Moving-only (572) | Stopped-only (67) | Waiting-only (75) | Crossing-only (64) | Mixed (153) |
|---|---:|---:|---:|---:|---:|---:|
| ROAD direct |42.1|47.7|74.6|22.7|31.2|20.9|
| Adapted head |70.9|95.1|40.3|53.3|10.9|27.5|
| LOKI-trained |77.7|92.0|53.7|65.3|59.4|48.4|

![Per-track error ranges within each GT track type](../../outputs/experiments/E004/track-error-ranges-20261116/error-composition.png)

[SVG](../../outputs/experiments/E004/track-error-ranges-20261116/error-composition.svg).

For each model/error bin, numerical summaries retain track count/fraction, GT composition, duration, labeled-frame count and pedestrian–ego distance. Points in the second figure are medians and bars span the 25th–75th percentiles (the middle 50%). Distance is calculated in meters from the existing unnormalized saved positions, using E001's `norm(ped_position - ego_position)` wherever pedestrian position and ego pose are valid. First compute each track's median distance over all valid unpadded context slots, including unlabeled slots, then summarize those track medians within each error group. Tracks without a valid distance remain unknown rather than being assigned zero. Duration remains nominal native timestamp span.

![Duration and distance across track-error ranges](../../outputs/experiments/E004/track-error-ranges-20261116/error-ranges.png)

[SVG](../../outputs/experiments/E004/track-error-ranges-20261116/error-ranges.svg).

**Interpretation:** mixed tracks are not uniformly lost: 74/153 have at most 20% error for the LOKI-trained model, compared with only 2/153 that are entirely correct. Mixed tracks are also longer: median 15.8 s (middle 50% 9.2–19.6 s), versus 3.7 s (1.8–6.6 s) for Moving-only tracks. This helps explain why perfect-track counts were restrictive. The adapted head's improvement remains concentrated in Moving; 50/64 Crossing-only tracks have more than 95% error, versus 42/64 under direct transfer and 18/64 under LOKI training. Length/distance summaries mix behaviors and cannot establish causality; class composition must accompany their interpretation.

For the LOKI-trained model, the 0–5% error group and the >95% error group have almost identical median track distances: **25.5 m** in both, with middle 50% **17.2–38.6 m** and **17.6–41.3 m** respectively. There is no simple overall separation by distance. Their median durations are 3.8 s and 2.6 s, while the >5–20% error group has median 17.0 s and 86 labeled timesteps. This is consistent with short tracks often being almost wholly right or wrong and longer tracks admitting partial errors; it does not isolate duration as the cause. All 931 tracks have at least one valid range measurement.

These are retrospective GT-based quality groups, **not confidence-selected automatic coverage or risk guarantees**. Historical checkpoint-selection exposure is unchanged. No retraining, encoder inference or native sensor decoding is needed: only the existing 931 small prepared geometry archives are read once. [Script](../../scripts/analyze-e004-track-error-ranges.py), [summary CSV](../../outputs/experiments/E004/track-error-ranges-20261116/summary.csv), [summary JSON](../../outputs/experiments/E004/track-error-ranges-20261116/summary.json), and [cached per-track geometry summaries](../../outputs/experiments/E004/track-error-ranges-20261116/per-track-metadata.csv). The cached CSV allows figure regeneration without SSH or additional archive reads.


Regenerate these two figures locally from the small cached metadata file:

```bash
uv run --locked --extra cpu python scripts/analyze-e004-track-error-ranges.py \
  --metadata-csv outputs/experiments/E004/track-error-ranges-20261116/per-track-metadata.csv
```

The original saved-geometry extraction and report generation took 6.5 s on Aalto. Boundary/partition checks and the 294-test CPU suite pass (13 expected skips). Both figures were visually checked; all error-bin and class-composition counts reconstruct 931 tracks per model.
