# AGENTS.md

## Project state

This repository is currently a research workspace for an Aalto University
project on automatic pedestrian-behavior annotation across driving datasets.

The scientific problem, label ontology, minimum input contract, and final
model architecture are still being refined.

Do not treat working hypotheses as settled decisions.

## Before starting work

Read `docs/README.md` for document ownership, then read the canonical documents
relevant to the task.

Primary sources of truth:

- `docs/RESEARCH_DIRECTION.md`
- `docs/PROBLEM_DEFINITION.md`
- `docs/LABEL_ONTOLOGY.md`
- `docs/DATASET_CONTRACT.md`
- `docs/PIPELINE_DESIGN.md`
- `docs/EVALUATION_PLAN.md`
- `docs/ROADMAP.md`
- `docs/OPEN_QUESTIONS.md`

Use `docs/DATASETS/` for dataset-specific facts and
`docs/LITERATURE/` for literature evidence.

## Documentation model

There are three different forms of project documentation.

### Canonical documents
Contain the current project understanding.
Edit them when the understanding changes.

### Decision records
`docs/DECISIONS/` records important settled choices and why they were made.

Do not create a decision record for unresolved questions.

### Research log
`docs/LOG/` records what happened chronologically.

Do not duplicate detailed research conclusions in the log.
Instead, reference the canonical document that was updated.

### Experiments
When experiments begin, use `docs/EXPERIMENTS/` for reproducible run records and
a compact results index. Keep the protocol in `EVALUATION_PLAN.md`; record
observed results once in experiment records and link to them elsewhere.

## Research rules

- Prefer evidence-driven decisions.
- Start from the simplest meaningful baseline.
- Inspect failures before selecting more complex methods.
- Add modalities or learned models only when they address an identified gap.
- Do not assume multimodal fusion is necessary.
- Distinguish observable pedestrian behavior from latent intention.
- Treat cross-dataset generalization as an important design consideration.
- Preserve uncertainty where evidence is incomplete.
- Do not claim novelty until supported by literature comparison.

## Current project boundary

The old `zod-ped` / ZOD-IAC project is a reference, not the foundation of
this repository.

Do not copy its architecture wholesale.

Potentially reusable ideas include:
- trajectory representations;
- temporal utilities;
- event/onset derivation;
- provenance;
- confidence/abstention concepts;
- evaluation patterns.

ZOD-specific detection, frustum lifting, GOLD/SILVER logic, ego-road
assumptions, and the PV-LSTM committee are not part of the new foundation
unless explicitly justified later.

## Implementation policy

The project is currently in the research-definition stage.

Do not introduce substantial model implementations or rigid software
abstractions before the relevant problem definition and dataset contract
are sufficiently settled.

Small utilities for inspecting datasets and annotations are encouraged.

Keep reusable code in `src/`.
Keep one-off inspection tools in `scripts/`.
Keep exploratory notebooks in `notebooks/`.
Keep configuration in `configs/`.

## Python environment

Run project Python commands in the `pedestrian-behavior` Conda environment
(Python 3.11). Activate it with `conda activate pedestrian-behavior`.
The inspection commands also require `ffmpeg` on `PATH`.

## Compute resources

- Local laptop: ThinkPad T14 Gen 4, Intel i7, integrated graphics.
- Training can also run on an MSI Raider GE78 HX 13V laptop with an RTX 4080,
  reachable through the `aalto` SSH host. Do not put SSH credentials in this repository.

## Repository hygiene

- Never commit datasets or large generated artifacts.
- Do not place temporary files in the repository root.
- Avoid duplicate documentation.
- Do not create files such as `final_v2_latest.md`; update canonical files
  and rely on Git history.
- Keep literature sources/URLs attached to factual claims.
- Mark unverified information explicitly.
- Prefer `unknown` over guessing.

## When completing research work

If a task changes our understanding:

1. update the relevant canonical document;
2. update `OPEN_QUESTIONS.md` if a question was resolved or created;
3. create a decision record only if an important choice became settled;
4. add a short entry to the current research log describing what changed.

## Validation

Run `python -m unittest discover -s tests` from the repository root in the
project Conda environment.
