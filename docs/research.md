# Research definition and direction

Status: research-definition stage. Updated **2026-10-25**: [E002](../experiments/E002-multimodal-reference/README.md) is the multimodal direction overview. [E002a](../experiments/E002a-feature-feasibility/README.md#completed-source-validation-runs--2026-10-25) completed eight seed0 source-validation runs; geometry + LiDAR performed best on both sources in that seed. Its original preparation, retired cache, timing and failure evidence remains preserved. [E002b](../experiments/E002b-normalization-capacity/README.md) now owns the agreed branch-normalization/capacity follow-up and proposed shared LiDAR preparation; model definitions are prepared; grouped execution and benchmark are pending. The [joint five-seed review](../experiments/E001b-2d-bbox-contribution/results-review.md) preserves all 130 completed E001-family attempts and earlier rounds. No E002 held-out or transfer result is established.

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

1. Prepare [E002b's normalization/capacity comparison](../experiments/E002b-normalization-capacity/README.md), first validating a bounded shared-preparation benchmark for LiDAR models. Check independent model/RNG/optimizer/selection semantics and actual speed/memory before full execution. E002a's [timing evidence and worst-case epoch gate](../experiments/E002a-feature-feasibility/README.md#throughput-gate--agreed-2026-10-22) remain build-specific; no new runtime is assumed. The [literature recommendation](literature/offline-labeling-research-direction.md#essential-next-experiments-and-stopping-rules) motivates useful scene evidence with an ordinary bidirectional model first. Complete-track-conditioned scene retrieval remains conditional; [CrossFeat](literature/papers/crossfeat.md) and [TubeDETR](literature/papers/tubedetr.md) are close mechanism precedents. Preserve the [five-seed reference](../experiments/E001b-2d-bbox-contribution/results-review.md), earlier budget rounds and target-inspection history; no E001-family epoch-cap increase is motivated.
2. Arrange separate backup storage for valuable runs (**TBD**). The frozen LOKI collection/setup are checksum-verified on `aalto` for both-dataset runs. [E001 populations, frozen clip splits and source normalization](../experiments/E001-kinematic-transfer/README.md#real-data-setup-evidence-2026-10-08) are implemented; clip/scenario independence is the accepted user assumption and needs no further approval/audit.
3. Preserve [ROAD's qualified association acceptance](../experiments/E001-kinematic-transfer/README.md#road-association-acceptance-2026-10-08): a full source/ID/context audit and purposive 33-track review, with inconclusive cases disclosed. This closes the E001 gate, without establishing population-wide visual matching accuracy.
4. Before a new representation, check native crop/local-cloud usability and its exact checkpoint exposure. [Utonia](literature/papers/utonia.md) supports missing color/normals but includes Waymo/nuScenes pretraining; [OmniVGGT](literature/papers/omnivggt.md) takes RGB with optional aligned depth/cameras, not RGB-free raw LiDAR. Known or unknown target/source-test exposure must be disclosed separately from source-only behavior training. [Dataset notes](datasets/README.md) retain physical timing, calibration and map/context unknowns; these are not reopened prerequisites for E001.

E001's protocol and data/association gates are accepted with recorded limitations. Training/evaluation, GPU acceptance and the E001/E001b/E001c five-seed comparisons are complete; [the joint report](../experiments/E001b-2d-bbox-contribution/results-review.md) retains their results and earlier budget deviations. Broader ontology, paired sensor projection, final models and auxiliary-source protocols remain provisional. E002a's source-validation pilot is complete; E002b's model definitions are prepared; grouped execution and GPU calibration remain pending. These development results do not establish held-out generalization or explain an RGB shortcut.

## Tentative experiment direction

**Recommendation refined 2026-10-16; pilot specification agreed2026-10-21.** Keep both RQs, prioritizing a complete empirical result and useful labeler by December. The literature review removed the earlier Transformer-first prerequisite. The user chose one multimodal reference with a bidirectional Transformer, rather than a separate temporal architecture search. E002a owns the agreed first-pilot specification and controls; retain BiLSTM as the historical reference and hold the new temporal architecture fixed across modality comparisons. Choose at most one substantive retrieval or auxiliary-supervision branch after development evidence and cost gates; do not promise both before feasibility is known.

| Comparison | Question / initial design |
|---|---|
| [E001](../experiments/E001-kinematic-transfer/README.md) — agreed baseline | Four kinematic feature sets × MLP/BiLSTM, within each dataset and in both transfer directions. |
| [E002 — multimodal working reference](../experiments/E002-multimodal-reference/README.md) | Direction overview. [E002a](../experiments/E002a-feature-feasibility/README.md) owns the completed first pilot and preparatory evidence. [E002b](../experiments/E002b-normalization-capacity/README.md) owns the agreed LayerNorm/original-width versus temporal128 versus compact32/64/64 comparison across geometry / +RGB / +LiDAR. Adding a modality includes availability; Utonia remains deferred. |
| E003 — scene retrieval, conditional | One matched pooling/frame-query/full-track-query comparison if E002 indicates useful scene evidence and a consequential compression limit. Same observations/features and a capacity control; no automatic architectural expansion. |
| E004 — complementary supervision, still tentative | Fixed inference representation/initialization: source behavior training, +one source with a generic objective, +the same observations with native semantic supervision. nuScenes remains a candidate, subject to usable semantics/observations and checkpoint-exposure controls. |

E002–E004 retain their IDs; E002a owns the completed first behavior pilot and its original feasibility evidence, while E002b owns the normalization/capacity follow-up. Its initial seed0 matrix and proposed shared preparation do not establish a measured runtime or authorize a broader seed/architecture search. E001's accepted settings and past results remain unchanged. Retain within-dataset and bidirectional transfer where semantics permit; define preprocessing, availability, exposure, objectives, budgets and seeds before execution. The [experiment index](../experiments/README.md) links the E002 comparisons; E003/E004 detailed records remain TBD. The [closest-work comparison](literature/offline-labeling-research-direction.md#closest-work-comparison) identifies useful adaptations and their protocol differences.

Hold the temporal backbone fixed across modalities within each E002 configuration. E002b explicitly varies capacity between its declared configurations; do not attribute that contrast to modality alone. Hold the temporal backbone fixed in any E003 mechanism comparison, and inference representation/initialization fixed during E004. Keep populations/GT comparable, including missing target-RGB tracks, and report class/condition support. For a mechanism comparison, give every offline control the same full observations and frozen spatial features; match a pooled-summary/capacity control. Auxiliary labels are supervision, not invented behavior GT or extra inference inputs. Preserve source-only selection, qualify external pretraining and disclose target-informed changes under the [shared safeguards](../experiments/README.md).

The questions remain valuable after negative results. The literature review alone authorizes no preparation or training; E002a's completed runs were user-launched. E002b now has the agreed model definitions and CPU acceptance; grouped execution and training have not started. Before a new increment, agree its data flow/acceptance and verify the narrow native/checkpoint gates. If an auxiliary source is not viable by the November gate, explicitly narrow the December contribution and report RQ2 unresolved. Avoid an architecture × encoder × source search; repeat only central comparisons and retain failures.

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
