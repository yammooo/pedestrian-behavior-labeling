# Open questions

Status: Active; superseded questions retained for historical links

Last updated: 2026-10-01

## Purpose and ownership

- Contains: unresolved decisions, current evidence, and what would answer them.
- Links out: facts to [dataset notes](DATASETS/DATASET_MATRIX.md), semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), protocols to [EVALUATION_PLAN.md](EVALUATION_PLAN.md), and optional methods to [IDEAS_BACKLOG.md](IDEAS_BACKLOG.md).

## Immediate gates

### Q12. Can ROAD-Waymo behaviour annotations be linked robustly to Waymo 3D tracks?

Open. No release or linkage implementation has been acquired here. Establish segment/frame/object correspondences, association provenance and coverage, varied manual validation, and a predeclared acceptance rule. [Gate 1](DATASET_INSPECTION_PLAN.md#gate-1--road-waymo--waymo-linkage) owns the checks. Failure requires reconsidering source supervision.

### Q17. How much usable pedestrian sequence supervision does each source provide?

Open. [Population comparison](DATASETS/DATASET_MATRIX.md#pedestrian-sequence-scale) separates pedestrian tubes from all-agent/frame counts. Verify acquired-release counts by split, native identity, action, length and gaps. ROAD-Waymo's reported 11,759 pedestrian tubes include only 516 waiting action tubes; the retained 3D-matched subset is unknown. Reconcile IDD-PeD's 4,916 split total with the paper's >5,000, and recount nuScenes's externally reported 8,143 with explicit category/split scope. Scene diversity and non-overlapping identities matter alongside frame volume.

### Q13. What do the native source annotations actually mean?

Open. Extract exact ROAD-Waymo labels from the acquired version; compare definitions, applicability, timing, and examples with LOKI before zero-shot mapping. Audit nuScenes attributes/scene supervision and ROAD/IDD-PeD native annotations without forcing them into one taxonomy. The ontology table is preliminary.

### Q2. Does LOKI support a fair low-label benchmark?

Open. [Local population counts](DATASETS/LOKI.md#unique-pedestrian-population-2026-10-01) now establish 12,364 scenario-scoped 3D pedestrian tracks, including 4,139 without any 2D box. Independent physical identities, gaps/durations, episodes, transitions, final splits and evidence-based visibility cohorts remain incomplete. Resolve these before choosing budget units, sampling, and split manifests.

### Q14. What is the minimum usable target input and comparable representation?

Open. Verify coordinates, timing, ego compensation, point-cloud/map correspondence, and evidence quality. The [contract](DATASET_CONTRACT.md) supports heterogeneous availability; RGB is not universally required. Determine what 3D-only labeling can use reliably rather than accepting the old RGB/2D minimum.

## Experimental choices

### Q5. Which target states are observable from the chosen representation?

Open. Stopped/Waiting may be only partly identifiable; Crossing needs scene relation. Use controlled trajectory/scene diagnostics and class-wise errors. These target diagnostics must not select a strict zero-shot model.

### Q7. Which representations transfer, and which introduce domain shift?

Open. Separate sensor, geographic, scene, selection, and label-semantic effects. Compare verified kinematics, scene context, and available RGB; test whether paired supervision improves 3D-only performance. Do not assume multimodal fusion helps.

### Q8. What temporal and sampling protocol is defensible?

Open. Choose whole tracks versus chunks, useful context, gap handling, scene grouping, label-budget unit/denominator, validation-label accounting, and repeated draws/seeds. Decide treatment of the seven previously inspected LOKI scenarios. Metrics, temporal definitions, and numeric performance targets are not selected.

### Q15. Which additional source should be tested first, if any?

Open. ROAD-Waymo alone is the baseline candidate. nuScenes may provide scene/sensor diversity; ROAD may provide visual behavioural diversity. Inclusion and order are undecided. IDD-PeD remains a later option. Resolve using feasibility and a named gap in the source baseline, with controlled ablations.

### Q16. Do shared representations and native heads help without negative transfer?

Open. Dataset-specific heads, the motion/scene/crossing factorization, joint sampling/loss weights, fusion, temporal architecture, and RGB-to-3D objectives are hypotheses. Ordinary joint training is the starting candidate if multiple sources are used. Measure whether additions hurt before introducing gradient-conflict methods.

## Contribution and eventual labeling

### Q9. What contribution remains after the closest literature is compared?

Open. Compare heterogeneous partial supervision, differing label taxonomies, offline segmentation, cross-modal transfer, and annotation efficiency against primary sources. [GAP_ANALYSIS.md](LITERATURE/GAP_ANALYSIS.md) does not establish novelty.

### Q10. When is automated enrichment trustworthy enough for an unlabeled dataset?

Open; conditional after baseline analysis. Define confidence, calibration, independent audit, acceptable reliability/coverage, and human effort before selective export. No label/review quota or coverage threshold is accepted. ECP2.0 is a possible later application, not a deadline dependency.

## Superseded questions

These entries preserve IDs and historical anchors. Their former details remain in Git and [PedSynth++ research history](DATASETS/PED_SYNTH_PLUS_PLUS.md#research-history).

### Q1. Is the full PedSynth++ release accessible, and what does it actually label?

Superseded as an immediate gate. A local release was inspected; its modality limitations and generator gaps no longer support the original foundational plan. Remaining release provenance is recorded in the dataset note.

### Q3. Which PedSynth++ behaviors can supervise the LOKI states?

Superseded as active supervision. The historical mapping was unaccepted and depended on rich states unavailable in the inspected active generator.

### Q4. Which inputs can be produced comparably in all three datasets?

Superseded. The PedSynth++/LOKI/ECP three-way intersection no longer defines the contract; Q14 owns the heterogeneous-input question.

### Q6. Does rich synthetic supervision save real labels?

Deferred. This remains a possible future question if suitable synthetic supervision becomes available; current experiments begin with real sources.

### Q11. Can EMT serve as an external real-world test?

Deferred from the active plan. Preserve [EMT evidence](DATASETS/EMT.md) as background; no EMT experiment is currently selected.
