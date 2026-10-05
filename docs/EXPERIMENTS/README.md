# Experiment records

Status: Recording convention; no model runs registered

Last updated: 2026-10-02

## Purpose and ownership

- Contains: a compact comparison index and reproducible records of feasibility runs and experiments.
- Links out: global protocol to [EVALUATION_PLAN.md](../EVALUATION_PLAN.md), hypotheses to [RESEARCH_DIRECTION.md](../RESEARCH_DIRECTION.md), and measured dataset facts to [DATASETS/](../DATASETS/DATASET_MATRIX.md).

The scientific method is central to the research artifact. Declare the question, access rules and comparison before execution; record what actually ran and what failed. Do not describe a retrospective choice as predeclared.

## Index

No experiments are registered yet. Existing LOKI/ROAD-Waymo galleries are inspection media, not trained-model results. The proposed first bidirectional kinematic diagnostic is defined in [EVALUATION_PLAN.md](../EVALUATION_PLAN.md#first-two-dataset-diagnostic); no run record or implementation is created by the handoff.

Add one row per meaningful experiment or tightly related run group. Link its record and any baseline/prerequisite record.

| ID / date | Hypothesis or feasibility question | Status | Record | Main result / limitation |
|---|---|---|---|---|

## Record and artifact locations

Use a descriptive `YYYY-MM-DD-short-description.md` record in this directory. Keep versioned effective configurations in `configs/`; record the exact command and config path/revision. Large logs, predictions, plots, checkpoints and manifests belong under ignored `outputs/` or a documented external location.

Record immutable IDs/checksums for release/split/subset artifacts where practical. Do not commit raw datasets or credentials. Each run needs an identifiable output location, including failed runs. A tightly related group can share a record, but retain each run's seed, configuration, status and result.

## Minimum record template

```markdown
# Experiment / feasibility check

ID:
Date:
Status: Planned | Running | Complete | Failed | Inconclusive
Hypothesis / question:
Baseline / prerequisite record:
Protocol reference and version:

## Before execution

- Comparison and expected evidence; what would support or weaken the hypothesis.
- Dataset sources/releases, acquisition provenance, association version/quality.
- Native annotation/output semantics and any evaluation projection.
- Split and subset manifests/IDs; scene grouping; inspected-scene policy.
- Data access regime: source validation, strict zero-shot, low-shot, or target diagnostic.
- Label budget unit/denominator and absolute counts; sampling and validation-label accounting.
- Available/derived modalities, missing-input/GT policy, and evaluated cohorts.
- Primary/secondary metrics, aggregation, temporal definitions, and selection rule.
- Seeds/repeated draws, relevant controls, training exposure, and planned compute.

## Run provenance

- Code revision and any uncommitted diff.
- Effective config path/revision, full commands, dependency/environment versions.
- Hardware/host identifier without credentials; start/end times.
- Run IDs, seeds and subset IDs; output/log/checkpoint/prediction locations.
- Selected checkpoint and the validation evidence used to select it.

## Results

- Per-run status and metric artifact; retain failures and negative results.
- Aggregate comparison and variability, with cohort/class denominators.
- Runtime/training exposure/compute and observed association or data failures.
- Deviations from the declared protocol and their consequences.

## Interpretation

- What the evidence supports; uncertainty, confounds and failure examples.
- Whether source additions helped or hurt and which comparison establishes it.
- Next justified step, if any; links to canonical understanding/questions changed.
```

Use `unknown` or explain non-applicability rather than inventing fields for feasibility checks. Metric and budget definitions are still open globally; settle the applicable definitions before the run.

## Reporting discipline

The evaluation plan owns protocol, experiment records own observed results, dataset notes own measured release facts, and the research log contains short pointers. Store detailed metrics once and link them. Record configurations that did not help; do not silently drop failed seeds or tune against a held-out target result.

Report transfer direction, target-label/model-selection access and compute alongside improvements. If that run's target evidence changes its source design, disclose target-informed development instead of relabeling it strict zero-shot. Generated labels do not replace independent validation.
