# Research definition and direction

Status: research-definition stage. Updated 2026-10-09: E001’s two seed-0 rounds received a brief budget/metric review; the five-seed 75/8 repetition launcher is accepted; follow-on experiments remain tentative. Empirical answers, broader ontology/head choices and final method remain open.

## Task and scope

Given an existing pedestrian track over a recorded sequence, assign a dense per-frame behavior-state sequence. A tracked temporal segment may contain several states. This is offline annotation: complete past and future observations, bidirectional processing and non-causal post-processing are permitted. Online future-action/intention prediction is a separate causal task. The [tracking pivot](archive/2026-09-22-tracking-pivot.md) records why detection, lifting and association reconstruction are excluded.

The intended artifact is a general offline pedestrian-behavior labeler for autonomous-driving datasets, supported by reproducible experiments and traceable results. Reliable automatic coverage and minimal human review are end goals; no numeric target or human-label quota is accepted. Outputs may later include derived segments/transitions/crossing onset and end, plus uncertainty for acceptance/review, only with explicit definitions and quality evidence.

The study is framed under a concrete contract: **RGB + 2D pedestrian tracks + LiDAR/3D point clouds + 3D pedestrian tracks + ego motion + timestamps/calibration**. [Dataset conventions](datasets/README.md) distinguish this contract from per-observation availability and heterogeneous source supervision; exact alignment, fields and usable-observation rules still need verification. Missing RGB or GT never implies a behavior class. Synthetic private state and behavior labels are supervision, not deployable observations.

The current target ontology is motivated by [LOKI](datasets/loki.md): Moving, Stopped, Waiting to cross and Crossing. [E001's four-state projection](../experiments/E001-kinematic-transfer/README.md#accepted-four-state-projection) is accepted for that diagnostic, with native semantic differences retained. The longer-term endpoint may be a transferable shared representation with lightweight dataset-specific behavior heads; no universal hard-coded ontology is required. Native semantics stay in dataset notes.

## Research questions

**RQ1: Given RGB, LiDAR, 2D/3D pedestrian tracks, and ego motion, which pedestrian behaviors can be reliably labeled offline, what does each modality contribute, and under which observation conditions do the labels become ambiguous?**

RQ1 characterizes empirical recoverability, rather than asking whether the data is sufficient. Moving/Stopped may be strongly observable from kinematics; Stopped/Waiting may need road geometry, orientation, barriers, traffic context or the complete future trajectory. Distant, occluded, short or 3D-only observations may make labels weakly identifiable. Poor classification alone cannot prove sensor insufficiency or intrinsic ambiguity: model, protocol and annotation failures remain alternatives.

Start with [E001](../experiments/E001-kinematic-transfer/README.md): compare motion, added trajectory, added ego-relative interaction and a raw-state control, each with a framewise MLP and whole-track BiLSTM. Progressively test scene/LiDAR evidence and RGB/context after inspecting failures. Analyze class-wise performance and observation conditions, not just one global score. A later comparison with human judgments from exactly the same evidence may help distinguish model limitations from ambiguity; it needs a predeclared protocol and is not conclusive by itself. Future observations are limited to the complete observed track, not unknown future behavior.

**RQ2: Can heterogeneous partial supervision across datasets be used to learn representations that improve cross-dataset pedestrian behavior labeling compared with single-dataset end-task training?**

RQ2 tests whether complementary supervision teaches a more transferable representation, rather than assuming that more datasets help. ROAD-Waymo supplies real action/location labels with partial verified Waymo correspondence; nuScenes offers another multimodal setting with motion attributes and scene/maps; ROAD or IDD-PeD may add visual action/context without corresponding 3D state; LOKI supplies a 3D-first behavior setting with missing RGB evidence. [Dataset notes](datasets/README.md) own release-specific evidence and limitations.

A shared representation with partly separable motion, scene-relation and crossing factors (candidate `z_motion / z_scene / z_crossing`), plus native heads where semantics differ, is a likely direction, not a fixed architecture. Compare single-source end-task training with progressively justified combinations, preserving partial-label masks and reporting conflicting signals, dataset shortcuts and negative transfer. Factorization, source order and loss balancing remain methodological choices under RQ2.

The progression is **characterize recoverability → identify useful evidence → learn transferable representations from complementary supervision**. Kinematics, temporal context, visible/3D-only cohorts, factorization and extra datasets are experiments under these questions, not separate research themes. The proposed contribution is understanding recoverable offline behavior, evidence limits and whether heterogeneous supervision improves transfer beyond dataset-specific training. Higher F1 or cross-dataset evaluation alone is not a novelty claim; see the [literature comparison](literature/README.md).

Neither transfer direction is permanently primary (former Q18). E001 compares ROAD-Waymo ↔ LOKI and both within-dataset evaluations. Camera-visible-first versus 3D-track-first annotation is useful to study, but geography, sensors, scene structure, semantics and class frequencies also change. Directional asymmetry cannot isolate any one cause. Strict zero-shot and target-informed characterization are separate [access regimes](../experiments/README.md); disclose existing inspection and later target-informed changes. Excellent zero-shot performance is not required for useful label-efficiency results.

## Priorities and gates

1. Run the accepted five-seed 75/8 E001 launcher after the [brief single-seed review](../experiments/E001-kinematic-transfer/README.md#brief-comparison-and-budget-review-2026-10-09): both 30/5 and user-modified 50/8 rounds completed; longer patience helped some BiLSTMs, with mixed feature rankings. Training/evaluation/checkpoints passed CPU/CUDA acceptance and the separate W&B GPU smoke. The separate [E001b availability/geometry diagnostic](../experiments/E001b-2d-bbox-contribution/README.md) has prepared/verified caches on frozen E001 tracks; its accepted native-only boxes and shared-runner/launcher reuse the E001 protocol. Complete implementation checks, then execute the frozen BiLSTM-only availability/geometry matrix. Inspect E001 failures before broader methods.
2. Arrange separate backup storage for valuable runs (**TBD**). The frozen LOKI collection/setup are checksum-verified on `aalto` for both-dataset runs. [E001 populations, frozen clip splits and source normalization](../experiments/E001-kinematic-transfer/README.md#real-data-setup-evidence-2026-10-08) are implemented; clip/scenario independence is the accepted user assumption and needs no further approval/audit.
3. Preserve [ROAD's qualified association acceptance](../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08): a full source/ID/context audit and purposive 33-track review, with inconclusive cases disclosed. This closes the E001 gate, without establishing population-wide visual matching accuracy.
4. Alongside E001, verify broader physical timing, independent sensor accuracy, RGB projection and map/context usability. [Dataset notes](datasets/README.md) own these unknowns and semantic limitations; they are not new prerequisites for the accepted kinematic diagnostic.

E001's protocol and data/association gates are accepted with recorded limitations. [Models and numerical acceptance](../experiments/E001-kinematic-transfer/README.md#model-implementation-and-acceptance-2026-10-08) are implemented; training/evaluation and separate GPU smoke verification are complete; user-run comparisons received a brief review, with the longer-budget deviation retained. E001b is an agreed lightweight extension, with a separate preparation gate. Broader methods and architecture remain provisional.

## Tentative experiment direction

Keep both RQs as core objectives: characterize useful evidence, then test complementary-supervision transfer. Finish E001 as the bounded reference, then use a compact bidirectional Transformer as the preferred backbone for later comparisons. Its higher performance is a hypothesis; switching does not depend on beating every E001 score. Persistent training or resource problems may require revising that choice.

| Comparison | Question / initial design |
|---|---|
| [E001](../experiments/E001-kinematic-transfer/README.md) — agreed baseline | Four kinematic feature sets × MLP/BiLSTM, within each dataset and in both transfer directions. |
| E002 — Transformer reference | Use the same kinematic evidence and evaluation reference with a full-track Transformer. Reserve one full-versus-restricted-context comparison with consistent feature construction. |
| E003 — input evidence | On the Transformer, compare kinematics alone, +RGB, +local 3D scene evidence, and +both. Initial candidates are one pedestrian/context RGB crop and one pedestrian-centered LiDAR BEV; exact encodings remain TBD. |
| E004 — complementary supervision | Fix inference inputs and compare source behavior training, +one additional source with a generic auxiliary objective, and the same observations +semantic supervision. First candidate: nuScenes scene-relation targets, subject to native feasibility. |

E002–E004 are tentative comparison families, not execution-ready protocols. Retain within-dataset and bidirectional transfer evaluation where applicable. Define each comparison's architecture, preprocessing, availability, objectives, budgets and seeds before execution; detailed records are TBD in the [experiment index](../experiments/README.md). Use a suitable published pipeline as a reference where feasible; the [literature comparison](literature/README.md#closest-neighboring-work) qualifies the contribution.

Hold the temporal backbone fixed during E003, then the inference representation fixed during E004. Keep evaluated populations/GT comparable, including missing-RGB tracks, and report class/condition support. Auxiliary scene labels are supervision, not invented behavior GT or extra inference inputs. Preserve source-only selection and disclose target-informed changes under the [shared safeguards](../experiments/README.md).

The core questions remain worth testing after negative results; execution depends on valid data and functioning training. Prepare representations and verify the additional source alongside E001. Avoid a full architecture × representation × source search; retain failures and reserve repetition for the central comparisons.

### Possible follow-up branches

Choose at most one substantial follow-up after the core comparisons, if time permits:

| Finding | Possible follow-up |
|---|---|
| Modalities help within datasets but hurt transfer | One missing-modality or regularization comparison. |
| Additional supervision hurts | One auxiliary-loss or source-sampling diagnostic. |
| Stopped/Waiting remains difficult across models | Focused, blinded human judgments from the same evidence; not automatic replacement GT or proof of intrinsic ambiguity. |

If results are consistent, prioritize repeat runs and validation. Cut optional follow-ups first if the schedule slips; revise infeasible data/source plans explicitly rather than searching for a positive score.

## Broader hypotheses and unresolved direction

The empirical answers and contribution require results, advisor alignment and a [closest-work comparison](literature/README.md) (former Q9). The hypotheses below extend beyond the bounded core; they are not additional required deliverables. E001 is diagnostic, not the final contribution.

| Provisional hypothesis | Evidence needed before adoption |
|---|---|
| H1: heterogeneous real sources improve transfer | Single-source versus the first feasible addition; further ROAD/IDD-PeD inclusion and order remain open (Q15) |
| H2: native supervision heads support a shared representation | Audit incompatible taxonomies/missing labels; compare without a forced universal mapping |
| H3: 3D road/scene understanding adds value | Kinematics versus scene evidence for an identified road-relation failure |
| H4: paired multimodal evidence contributes under incomplete observations | Matched RGB-visible/3D-only populations and validated pairing; inference versus privileged-training use remains open (Q7) |
| H5: pretraining saves target annotations | Pretrained versus scratch at exactly equal target-label budgets and selection access |
| H6: source additions can hurt | Keep negative source/task ablations and report exposure/compute |

If justified, candidate flow is verified native observations → kinematic/visual/3D-scene representations → fusion if useful → offline temporal representation → native supervision/target heads → dense states → later uncertainty/review/export. Derived pose, pedestrian-centered clouds, bbox/orientation, road-relative features, barriers/accessibility and ego/maps remain alternatives to the initial crop/BEV candidates; the study contract does not require every model to consume every modality. Recurrence, temporal convolutions or ASFormer-like encoders/chunking remain alternatives if the preferred Transformer proves impractical.

For justified multi-source work, start with ordinary joint training, dataset-balanced sampling, native-label loss masks and explicit missing modalities; balancing/loss weights remain open. Do not assume a sequential curriculum retains knowledge. Paired dropout, consistency or teacher/student distillation need a demonstrated 3D-only benefit; complementary embeddings need not be identical. Gradient-similarity diagnostics, PCGrad, GradNorm, CAGrad or task/source adapters require observed conflict and failure of simpler changes (Q16).

[Motion-representation evidence](literature/transferable-motion-representations.md) motivates separating temporal setup from domain shift and, after E001 if justified, testing auxiliary motion/scene supervision with frozen-encoder behavior probes before more complex adaptation. These are candidate comparisons, not selected models.

Sensor stress tests require a diagnosed shift: LiDAR ring/beam subsampling, angular/range sparsification, point dropout, FOV/noise/sweep/intensity changes; RGB resolution/blur/crop/lighting/weather/FOV perturbations for observed shortcuts. Canonical coordinates and verified road-relative quantities may reduce sensor dependence. Do not assume multimodal fusion is necessary.

Confidence, calibration, abstention, selective acceptance, independent audit and human effort remain conditional after error analysis (Q10). Unlabeled-target adaptation would be a separate future regime, excluded from strict zero-shot. [PedSynth++ investigation](archive/2026-10-01-synthetic-supervision.md) preserves the demoted rich-synthetic label-efficiency hypothesis; future synthetic controlled road/scene supervision, rare cases or sensor stress tests need a measured gap. No new CARLA FSM is planned. EMT is background (former Q11); ECP2.0 is a possible later application, not a traineeship dependency.

## Schedule

The traineeship ends **17 December 2026** (`2026-12-17`). These are planning windows, not measured runtime estimates or guaranteed outcomes; training throughput is unknown. Write up each comparison as it finishes.

| Window | Work / gate |
|---|---|
| 6–9 October | Finish E001 native verification and population/split checks; assess remaining work |
| 12–23 October | After authorization, implement/run E001 and inspect failures; in parallel define RGB/3D representations and check the auxiliary source |
| 26–30 October | E002: establish the Transformer reference and declare the context comparison |
| 2–13 November | E003: selected evidence representations and modality comparisons |
| 16–27 November | E004: complementary-supervision comparison; at most one optional follow-up if ahead of schedule |
| 30 November–4 December | Repeat central comparisons, finish essential controls/context analysis, consolidate conclusions and freeze scope |
| 7–17 December | Protected buffer: delayed runs, essential ablations, report, documentation, cleanup, handover and presentation |

Scale claims to evidence and preserve the buffer. Low-shot/scratch, uncertainty, additional sources, synthetic generation and ECP2.0 deployment remain possible later work, not required December deliverables. Narrow scope and claims explicitly if core work cannot fit; do not plan required work beyond the traineeship.
