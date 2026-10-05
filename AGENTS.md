# AGENTS.md

## Before work

Read `docs/README.md`, then only the owners relevant to the task. This Aalto pedestrian-behavior workspace is in the research-definition stage: problem, ontology, minimum inputs and final architecture remain provisional. Do not present hypotheses as decisions.

## Document responsibilities

- `README.md`: setup and runnable commands.
- `docs/README.md`: approximately one-screen dashboard, status, next work and navigation.
- `docs/research.md`: current task, provisional questions/methods, priorities and schedule.
- `docs/datasets/README.md`: shared conventions/comparison; individual notes own native labels, schema, counts, access, provenance, sources and limitations.
- `docs/literature/README.md`: literature/reference overview and qualified contribution comparison; separate reviews only for substantial detail.
- `experiments/README.md`: shared evaluation safeguards and comparison index; each `E###-topic/README.md` owns its question, design, settings, variants/seeds/failures/results and reproducibility references. IDs stay stable; dates stay inside records. Comparison-specific configs live beside the record; shared configs in `configs/`.
- `docs/archive/`: dated investigations and former decision reasoning; historical context, not current instructions.
- `docs/log/`: short monthly chronology with pointers, not repeated conclusions.

Use concise, simple language, lowercase hyphenated names except standard `README.md` and `AGENTS.md`, and links instead of repeated explanations. Update only affected owners when evidence changes understanding; put concrete unset settings there as `TBD` or `unknown`. Update the dashboard when priorities/status change and log meaningful milestones. No separate decisions ledger, question catalogue or ideas backlog. Preserve evidence, qualifiers, uncertainty and external artifact paths; do not discard evidence to shorten a document.

## Research and implementation

Prefer evidence-driven choices and the simplest meaningful baseline. Inspect failures before adding modalities, sources or learned models. Do not assume multimodal fusion helps. Separate observable behavior from latent intention and offline full-record annotation from causal prediction. Treat cross-dataset generalization seriously; claim novelty only after literature comparison.

No substantial model implementation or rigid abstraction before problem/input and feasibility/protocol gates are sufficiently settled and implementation is authorized. Small native inspection utilities are encouraged: reusable code in `src/`, one-off tools in `scripts/`, exploration in `notebooks/`.

Predeclare comparisons, native/output semantics, releases/associations, populations/splits, missing-input/GT policy, metrics, model-selection access, budgets and seeds. Strict zero-shot excludes target training (including unlabeled adaptation) and target-based selection; disclose prior inspection and target-informed changes. Keep correlated frames/tracks together. Compare low-shot/scratch with equal labels/access and report denominators/compute. Retain failures and negative transfer. Generated labels need independent validation. Details belong in `experiments/README.md` and the affected comparison.

The old `zod-ped`/ZOD-IAC is a reference, not the foundation. Generic trajectories, temporal/event utilities, provenance, confidence/abstention and evaluation ideas may be deliberately reused. Do not copy its detection, frustum lifting, GOLD/SILVER, cut/stitch, ego-road/keyframe assumptions or PV-LSTM committee without explicit later justification.

## Environment and hygiene

Run project Python commands in Conda `pedestrian-behavior` (Python 3.11): `conda activate pedestrian-behavior`. Inspection also requires `ffmpeg` on `PATH`.

Local compute: ThinkPad T14 Gen 4, Intel i7, integrated graphics. Training may use the MSI Raider GE78 HX 13V with RTX 4080 through SSH host `aalto`; never commit credentials.

Never commit datasets or large generated artifacts. Use ignored `outputs/experiments/E###/` or documented external storage for experiment outputs. Keep temporary files out of the repository root. Avoid duplicate documents/version-suffixed files; update owners and rely on Git history. Attach literature sources/URLs to factual claims; mark unverified information and prefer `unknown` over guessing.

## Validation

Run `python -m unittest discover -s tests` from the repository root in `pedestrian-behavior`. For documentation changes, check local links/anchors and `git diff --check`.
