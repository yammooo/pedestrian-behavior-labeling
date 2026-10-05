# Research definition and direction

Status: research-definition stage; questions and final method provisional. Updated 2026-10-05; scientific direction from the 2026-10-02 handoff.

## Task and scope

Given an existing pedestrian track over a recorded sequence, assign a dense per-frame behavior-state sequence. A tracked temporal segment may contain several states. This is offline annotation: complete past and future observations, bidirectional processing and non-causal post-processing are permitted. Online future-action/intention prediction is a separate causal task. The [tracking pivot](archive/2026-09-22-tracking-pivot.md) records why detection, lifting and association reconstruction are excluded.

The intended artifact is a working offline research labeler with reproducible experiments and traceable results. Reliable automatic coverage and minimal human review are end goals; no numeric target or human-label quota is accepted. Outputs may later include derived segments/transitions/crossing onset and end, plus uncertainty for acceptance/review, only with explicit definitions and quality evidence.

Recorded inputs may include identity/time, 3D boxes/trajectory, point clouds, ego pose, RGB/2D and scene/maps where available. RGB is not universally required. Source modalities and taxonomies differ; the minimum usable target sensing contract remains open in [dataset conventions](datasets/README.md). Missing RGB or GT never implies a behavior class. Synthetic private state and behavior labels are supervision, not deployable observations.

Moving and stopping are largely observable; crossing needs road relation. Waiting to cross may contain inferred intention that limited sensing cannot identify uniquely. Future context can help without guaranteeing identifiability. [LOKI](datasets/loki.md) supplies the tentative four-state output ontology for [E001](../experiments/E001-kinematic-transfer/README.md), subject to a defensible ROAD-Waymo projection. Native semantics stay in dataset notes.

## Provisional questions

**RQ1: Can behavior supervision available only for camera-visible pedestrians be transferred into a 3D-centric temporal representation that can label pedestrians without visual observations?**

ROAD-Waymo's front-camera annotation selects a different population from LOKI's broader 3D-first tracks. ROAD-Waymo → LOKI also changes geography, sensors, scene structure, annotation conventions and class frequencies. These shifts are confounded; this question does not isolate generic sensor/geographic generalization.

**RQ2: Does factorizing pedestrian behavior into transferable motion, scene-relation, and crossing representations improve transfer to the 3D-only setting?**

Candidate `z_motion / z_scene / z_crossing` factors could accommodate complementary supervision without equating taxonomies. Motion may explain Moving/Stopped; Crossing requires road relation and Waiting may depend on future trajectory, orientation, context or latent intent. Necessity, separability, supervision and improvement over a simpler shared representation are untested.

Neither direction is permanently primary (former Q18). E001 compares ROAD-Waymo → LOKI and LOKI → ROAD-Waymo plus both within-dataset evaluations. Better reverse transfer would be consistent with selection asymmetry, but would not establish its cause. Class-wise errors and within-dataset performance are needed. Zero-shot is important where semantics permit; excellent zero-shot performance is not required for useful label-efficiency results. Existing target inspection and later target-informed development must be disclosed under the [access rules](../experiments/README.md).

## Priorities and gates

1. Complete varied visual acceptance and reproducibility of the acquired ROAD-Waymo/Waymo index. Structural joins and partial 3D coverage are verified; training acceptance remains open. Failed robust linkage requires reconsidering supervision, without silently substituting another source.
2. Audit native semantics and the tentative four-state projection, and characterize LOKI population/identity/gaps/cohorts. [Dataset notes](datasets/README.md) own evidence and remaining checks.
3. Freeze minimal comparable kinematics/ego features, missingness, temporal grid, cohort/split and model-selection rules for [E001](../experiments/E001-kinematic-transfer/README.md). A visibility-comparable subset remains under discussion and cannot test the full 3D-only problem.
4. Once implementation is authorized and gates pass, run the small framewise MLP versus whole-track BiLSTM diagnostic in both directions. Inspect failures before selecting an extension.

The 2026-10-02 handoff authorized documentation, not model implementation. This migration changes organization only. Existing readers/galleries remain native inspection tools; no rigid schema, adapter hierarchy, viewer framework or substantial model implementation is warranted yet.

## Later hypotheses and unresolved direction

The primary formulation and contribution need advisor alignment and a [closest-work comparison](literature/README.md) (former Q9). E001 is diagnostic, not the final contribution. Neither shared heads, fusion nor a temporal encoder alone establishes novelty.

| Provisional hypothesis | Evidence needed before adoption |
|---|---|
| H1: heterogeneous real sources improve transfer | Single-source versus individually justified additions; nuScenes/ROAD order undecided, IDD-PeD later (Q15) |
| H2: native supervision heads support a shared representation | Audit incompatible taxonomies/missing labels; compare without a forced universal mapping |
| H3: 3D road/scene understanding adds value | Kinematics versus scene evidence for an identified road-relation failure |
| H4: paired multimodal supervision helps without target RGB | Matched RGB-visible/3D-only populations and validated pairing; RGB as input/privileged supervision/full modality is open (Q7) |
| H5: pretraining saves target annotations | Pretrained versus scratch at exactly equal target-label budgets and selection access |
| H6: source additions can hurt | Keep negative source/task ablations and report exposure/compute |

If justified, candidate flow is verified native observations → kinematic/visual/3D-scene representations → fusion if useful → offline temporal representation → native supervision/target heads → dense states → later uncertainty/review/export. Crops versus local context, derived pose, pedestrian-centered clouds/BEV, bbox/orientation, road-relative features, barriers/accessibility and ego/maps are options, not required modalities. Bidirectional recurrence, temporal convolutions, Transformers or ASFormer-like encoders/chunking remain alternatives after E001.

For justified multi-source work, start with ordinary joint training, dataset-balanced sampling, native-label loss masks and explicit missing modalities; balancing/loss weights remain open. Do not assume a sequential curriculum retains knowledge. Paired dropout, consistency or teacher/student distillation need a demonstrated 3D-only benefit; complementary embeddings need not be identical. Gradient-similarity diagnostics, PCGrad, GradNorm, CAGrad or task/source adapters require observed conflict and failure of simpler changes (Q16).

Sensor stress tests require a diagnosed shift: LiDAR ring/beam subsampling, angular/range sparsification, point dropout, FOV/noise/sweep/intensity changes; RGB resolution/blur/crop/lighting/weather/FOV perturbations for observed shortcuts. Canonical coordinates and verified road-relative quantities may reduce sensor dependence. Do not assume multimodal fusion is necessary.

Confidence, calibration, abstention, selective acceptance, independent audit and human effort remain conditional after error analysis (Q10). Unlabeled-target adaptation would be a separate future regime, excluded from strict zero-shot. [PedSynth++ investigation](archive/2026-10-01-synthetic-supervision.md) preserves the demoted rich-synthetic label-efficiency hypothesis; future synthetic controlled road/scene supervision, rare cases or sensor stress tests need a measured gap. No new CARLA FSM is planned. EMT is background (former Q11); ECP2.0 is a possible later application, not a traineeship dependency.

## Schedule

The traineeship ends **17 December 2026** (`2026-12-17`). Windows are conditional priorities, not guaranteed outcomes or frozen method choices.

| Window | Work / gate |
|---|---|
| 1–9 October | Visual/reproduction acceptance; four-state audit; common coordinates, ego/velocity, timing/cohorts; freeze E001 protocol |
| 12–23 October | After authorization, minimal representation and MLP/BiLSTM; both within-dataset and transfer directions; inspect failures |
| 26 October–13 November | If justified, scene/missing-visual methods, factorization or one complementary source; no requirement to add all |
| 16–27 November | Conditional missing-RGB, sensor/alignment, source and negative-transfer comparisons; IDD-PeD optional |
| 30 November–4 December | Supported zero-shot, equal-budget low-shot/scratch, source/cohort ablations, class-wise failures; uncertainty only if feasible |
| 7–17 December | Protected buffer: delayed runs, essential ablations, report, documentation, cleanup, handover and presentation |

Scale claims to evidence and preserve the buffer. Adding every source, synthetic generation and ECP2.0 deployment are not deliverables; do not plan required work beyond the traineeship.
