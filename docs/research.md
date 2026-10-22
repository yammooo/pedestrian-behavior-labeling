# Research definition and direction

Status: research-definition stage. Updated **2026-10-22**: [E002](../experiments/E002-multimodal-reference/README.md) is the multimodal direction overview; [E002a](../experiments/E002a-feature-feasibility/README.md) owns the agreed first-pilot scope, tensor flow, fitting and450 GB bounded point-cache specification. Completed native-feature, visual, semantic and resource probes remain preparatory evidence. Production caches, the behavior model/trainer and manual launcher are implemented; bounded CUDA integration and full RGB preparation/readback are verified. The2026-10-22 snapshot of build6050233 records the first ROAD geometry attempt active with poor input throughput and no completed full epoch. The [joint five-seed review](../experiments/E001b-2d-bbox-contribution/results-review.md) preserves all 130 completed E001-family attempts and earlier rounds.

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

A shared representation with native heads where semantics differ remains a candidate. Partly separable factors (`z_motion / z_scene / z_crossing`) are a hypothesis, not a preferred architecture: [MulCPred](literature/papers/mulcpred.md) and [partial pedestrian multitask learning](literature/papers/detecting-32-attributes.md) already establish relevant precedents. Compare single-source behavior training with one justified addition, preserving partial-label masks and reporting conflict, shortcuts and negative transfer. Matched extra-observation/generic-objective controls are necessary to attribute a gain to complementary semantics. Factorization, source order and loss balancing remain open.

The progression is **characterize recoverability → identify useful evidence → learn transferable representations from complementary supervision**. Kinematics, temporal context, visible/3D-only cohorts, factorization and extra datasets are experiments under these questions, not separate research themes. The proposed contribution is understanding recoverable offline behavior, evidence limits and whether heterogeneous supervision improves transfer beyond dataset-specific training. Higher F1 or cross-dataset evaluation alone is not a novelty claim; see the [literature comparison](literature/README.md).

Neither transfer direction is permanently primary (former Q18). E001 compares ROAD-Waymo ↔ LOKI and both within-dataset evaluations. Camera-visible-first versus 3D-track-first annotation is useful to study, but geography, sensors, scene structure, semantics and class frequencies also change. Directional asymmetry cannot isolate any one cause. Strict zero-shot and target-informed characterization are separate [access regimes](../experiments/README.md); disclose existing inspection and later target-informed changes. Excellent zero-shot performance is not required for useful label-efficiency results.

## Priorities and gates

1. Resolve the [E002a population-scale feature-supply bottleneck](../experiments/E002a-feature-feasibility/README.md#full-population-throughput-check--2026-10-22) before further pilot execution and meet the [agreed worst-case epoch gate](../experiments/E002a-feature-feasibility/README.md#throughput-gate--agreed-2026-10-22). The inspected run remains unchanged by these documentation updates. The [literature recommendation](literature/offline-labeling-research-direction.md#essential-next-experiments-and-stopping-rules) motivates establishing useful RGB/context/local-3D evidence with an ordinary bidirectional model first. Complete-track-conditioned scene retrieval is conditional; [CrossFeat](literature/papers/crossfeat.md) and [TubeDETR](literature/papers/tubedetr.md) are close mechanism precedents. Preserve the [five-seed reference](../experiments/E001b-2d-bbox-contribution/results-review.md), earlier budget rounds and target-inspection history; no E001-family epoch-cap increase is motivated by this review.
2. Arrange separate backup storage for valuable runs (**TBD**). The frozen LOKI collection/setup are checksum-verified on `aalto` for both-dataset runs. [E001 populations, frozen clip splits and source normalization](../experiments/E001-kinematic-transfer/README.md#real-data-setup-evidence-2026-10-08) are implemented; clip/scenario independence is the accepted user assumption and needs no further approval/audit.
3. Preserve [ROAD's qualified association acceptance](../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08): a full source/ID/context audit and purposive 33-track review, with inconclusive cases disclosed. This closes the E001 gate, without establishing population-wide visual matching accuracy.
4. Before a new representation, check native crop/local-cloud usability and its exact checkpoint exposure. [Utonia](literature/papers/utonia.md) supports missing color/normals but includes Waymo/nuScenes pretraining; [OmniVGGT](literature/papers/omnivggt.md) takes RGB with optional aligned depth/cameras, not RGB-free raw LiDAR. Known or unknown target/source-test exposure must be disclosed separately from source-only behavior training. [Dataset notes](datasets/README.md) retain physical timing, calibration and map/context unknowns; these are not reopened prerequisites for E001.

E001's protocol and data/association gates are accepted with recorded limitations. Training/evaluation, GPU acceptance and the E001/E001b/E001c five-seed comparisons are complete; [the joint report](../experiments/E001b-2d-bbox-contribution/results-review.md) retains their results and earlier budget deviations. Broader ontology, paired sensor projection, final models and auxiliary-source protocols remain provisional. E002a's first-pilot scope/architecture/fitting/cache specification is agreed; preparatory probes, the behavior implementation and bounded integration checks exist; full pilot comparisons remain pending.

## Tentative experiment direction

**Recommendation refined 2026-10-16; pilot specification agreed2026-10-21.** Keep both RQs, prioritizing a complete empirical result and useful labeler by December. The literature review removed the earlier Transformer-first prerequisite. The user chose one multimodal reference with a bidirectional Transformer, rather than a separate temporal architecture search. E002a owns the agreed first-pilot specification and controls; retain BiLSTM as the historical reference and hold the new temporal architecture fixed across modality comparisons. Choose at most one substantive retrieval or auxiliary-supervision branch after development evidence and cost gates; do not promise both before feasibility is known.

| Comparison | Question / initial design |
|---|---|
| [E001](../experiments/E001-kinematic-transfer/README.md) — agreed baseline | Four kinematic feature sets × MLP/BiLSTM, within each dataset and in both transfer directions. |
| [E002 — multimodal working reference](../experiments/E002-multimodal-reference/README.md) | Direction overview. [E002a](../experiments/E002a-feature-feasibility/README.md) owns the first geometry/RGB/D-DITR pilot (Utonia deferred), matched availability controls, full-track Transformer and preparatory evidence. Later comparisons need their own architecture definitions. |
| E003 — scene retrieval, conditional | One matched pooling/frame-query/full-track-query comparison if E002 indicates useful scene evidence and a consequential compression limit. Same observations/features and a capacity control; no automatic architectural expansion. |
| E004 — complementary supervision, still tentative | Fixed inference representation/initialization: source behavior training, +one source with a generic objective, +the same observations with native semantic supervision. nuScenes remains a candidate, subject to usable semantics/observations and checkpoint-exposure controls. |

E002–E004 retain their IDs; E002a owns the first behavior pilot, with completed feasibility probes preserved as preparation. Its agreed specification is implemented with caches, calibrated execution and source-validation smoke evidence; full pilot runs remain pending. The literature recommendation changes dependencies/priority, not E001's accepted settings or past results. Retain within-dataset and bidirectional transfer where semantics permit; define preprocessing, availability, exposure, objectives, budgets and seeds before execution. The [experiment index](../experiments/README.md) links the E002 overview and E002a pilot; E003/E004 detailed records remain TBD. The [closest-work comparison](literature/offline-labeling-research-direction.md#closest-work-comparison) identifies useful adaptations and their protocol differences.

Hold the temporal backbone fixed during E002 modality comparisons and any E003 mechanism comparison, and inference representation/initialization fixed during E004. Keep populations/GT comparable, including missing target-RGB tracks, and report class/condition support. For a mechanism comparison, give every offline control the same full observations and frozen spatial features; match a pooled-summary/capacity control. Auxiliary labels are supervision, not invented behavior GT or extra inference inputs. Preserve source-only selection, qualify external pretraining and disclose target-informed changes under the [shared safeguards](../experiments/README.md).

The questions remain valuable after negative results. The literature review alone authorizes no preparation or training; E002a executions include documented preparatory probes and separately authorized bounded integration checks; six full pilot runs remain manual. Before a new increment, agree its data flow/acceptance and verify the narrow native/checkpoint gates. If an auxiliary source is not viable by the November gate, explicitly narrow the December contribution and report RQ2 unresolved. Avoid an architecture × encoder × source search; repeat only central comparisons and retain failures.

### Possible follow-up branches

Choose at most one substantial follow-up after the core comparisons, if time permits:

| Finding | Possible follow-up |
|---|---|
| Modalities help within datasets but hurt transfer | One missing-modality or regularization comparison. |
| Additional supervision hurts | One auxiliary-loss or source-sampling diagnostic. |
| Stopped/Waiting remains difficult across models | Focused, blinded human judgments from the same evidence; not automatic replacement GT or proof of intrinsic ambiguity. |

If results are consistent, prioritize repeat runs and validation. Cut optional follow-ups first if the schedule slips; revise infeasible data/source plans explicitly rather than searching for a positive score.

## Broader hypotheses and unresolved direction

The [2026-10-13 review](literature/offline-labeling-research-direction.md#strongest-contribution-candidates) ranks empirical/benchmark evidence first, a conditional complete-track retrieval mechanism second and complementary semantic supervision third. None is established novelty. Empirical findings and advisor alignment still determine the contribution (former Q9). The hypotheses below remain possibilities, not extra required deliverables; E001 is diagnostic, not the final method.

| Provisional hypothesis | Evidence needed before adoption |
|---|---|
| H1: heterogeneous real sources improve transfer | Single-source versus the first feasible addition; further ROAD/IDD-PeD inclusion and order remain open (Q15) |
| H2: native supervision heads support a shared representation | Audit incompatible taxonomies/missing labels; compare without a forced universal mapping |
| H3: 3D road/scene understanding adds value | Kinematics versus scene evidence for an identified road-relation failure |
| H4: paired multimodal evidence contributes under incomplete observations | Matched RGB-visible/3D-only populations and validated pairing; inference versus privileged-training use remains open (Q7) |
| H5: pretraining saves target annotations | Pretrained versus scratch at exactly equal target-label budgets and selection access |
| H6: source additions can hurt | Keep negative source/task ablations and report exposure/compute |

If justified, candidate flow is verified native observations → kinematic/visual/3D-scene representations → fusion if useful → offline temporal representation → native supervision/target heads → dense states → later uncertainty/review/export. Derived pose, pedestrian-centered clouds, bbox/orientation, road-relative features, barriers/accessibility and ego/maps remain alternatives to the initial crop/BEV candidates; the study contract does not require every model to consume every modality. Recurrence, temporal convolutions and ASFormer-like encoders/chunking are established alternatives; no temporal family is mandatory.

For justified multi-source work, start with ordinary joint training, dataset-balanced sampling, native-label loss masks and explicit missing modalities; balancing/loss weights remain open. Do not assume a sequential curriculum retains knowledge. Paired dropout, consistency or teacher/student distillation need a demonstrated 3D-only benefit; complementary embeddings need not be identical. Gradient-similarity diagnostics, PCGrad, GradNorm, CAGrad or task/source adapters require observed conflict and failure of simpler changes (Q16).

[Motion-representation evidence](literature/transferable-motion-representations.md) motivates separating temporal setup from domain shift and, after E001 if justified, testing auxiliary motion/scene supervision with frozen-encoder behavior probes before more complex adaptation. These are candidate comparisons, not selected models.

Sensor stress tests require a diagnosed shift: LiDAR ring/beam subsampling, angular/range sparsification, point dropout, FOV/noise/sweep/intensity changes; RGB resolution/blur/crop/lighting/weather/FOV perturbations for observed shortcuts. Canonical coordinates and verified road-relative quantities may reduce sensor dependence. Do not assume multimodal fusion is necessary.

Confidence, calibration, abstention, selective acceptance, independent audit and human effort remain conditional after error analysis (Q10). Unlabeled-target adaptation would be a separate future regime, excluded from strict zero-shot. [PedSynth++ investigation](archive/2026-10-01-synthetic-supervision.md) preserves the demoted rich-synthetic label-efficiency hypothesis; future synthetic controlled road/scene supervision, rare cases or sensor stress tests need a measured gap. No new CARLA FSM is planned. EMT is background (former Q11); ECP2.0 is a possible later application, not a traineeship dependency.

## Schedule

The traineeship ends **17 December 2026** (`2026-12-17`). Revised planning windows follow the literature recommendation; multimodal feature throughput and new-model costs remain unknown, rather than being inferred from kinematic sweeps. Write up each comparison as it finishes.

| Window | Work / gate |
|---|---|
| 6–10 October — completed | E001/E001b native/runner gates, user-run five-seed sweeps and joint results review |
| 13–23 October | Literature-based direction agreement; development error/semantic review and narrow representation/exposure feasibility gates |
| 26 October–13 November | E002 multimodal reference and matched evidence comparisons; choose one justified follow-up |
| 16–27 November | One complete-track retrieval or E004 one-source supervision comparison; attempt both only if measured remaining cost permits |
| 30 November–4 December | Repeat central comparisons, finish essential controls/context analysis, consolidate conclusions and freeze scope |
| 7–17 December | Protected buffer: delayed runs, essential ablations, report, documentation, cleanup, handover and presentation |

Scale claims to evidence and preserve the buffer. Low-shot/scratch, uncertainty, additional sources, synthetic generation and ECP2.0 deployment remain possible later work, not required December deliverables. Narrow scope and claims explicitly if core work cannot fit; do not plan required work beyond the traineeship.
