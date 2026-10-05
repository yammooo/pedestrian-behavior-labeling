# Open questions

Status: Active; superseded questions retained for historical links

Last updated: 2026-10-02

## Purpose and ownership

- Contains: unresolved decisions, current evidence, and what would answer them.
- Links out: facts to [dataset notes](DATASETS/DATASET_MATRIX.md), semantics to [LABEL_ONTOLOGY.md](LABEL_ONTOLOGY.md), protocols to [EVALUATION_PLAN.md](EVALUATION_PLAN.md), and optional methods to [IDEAS_BACKLOG.md](IDEAS_BACKLOG.md).

## Immediate gates

### Q12. Can ROAD-Waymo behaviour annotations be linked robustly to Waymo 3D tracks?

Partially resolved. The [acquired remote index](DATASETS/ROAD_WAYMO.md#acquired-index-and-access-2026-10-02) has verified official same-frame associations, explicit missingness and a full-export consistency audit. ROAD-authoritative population/class policy is accepted. Remaining work is varied visual validation, disagreement review, release/code provenance and a predeclared training acceptance rule. [Gate 1](DATASET_INSPECTION_PLAN.md#gate-1--road-waymo--waymo-linkage) owns the checks. Failure requires reconsidering source supervision.

The [RGB + BEV gallery](DATASETS/ROAD_WAYMO.md#rgb--lidar-bev-inspection-gallery-2026-10-02) now supports that audit. Two real tracks were inspected, including one Cyclist-class disagreement; this small sample does not resolve population-wide matching quality.

### Q17. How much usable pedestrian sequence supervision does each source provide?

Open. [Population comparison](DATASETS/DATASET_MATRIX.md#pedestrian-sequence-scale) separates pedestrian tubes from all-agent/frame counts. Verify acquired-release counts by split, native identity, action, length and gaps. ROAD-Waymo's acquired train/validation subset has 9,573 pedestrian tracks: 6,540 with any paired 3D frame and 4,606 fully paired. Characterize paired lengths/gaps and action episodes; the paper's 516 waiting action tubes are a different count from current labeled rows. Reconcile IDD-PeD's 4,916 split total with the paper's >5,000, and recount nuScenes's externally reported 8,143 with explicit category/split scope. Scene diversity and non-overlapping identities matter alongside frame volume.

### Q13. What do the native source annotations actually mean?

Open. ROAD-Waymo native vocabulary and observed pedestrian action strings are recorded in the dataset note. Manually validate the [tentative four-state projection](LABEL_ONTOLOGY.md#tentative-first-baseline-projection), including direction variants, unmapped actions/co-labels, multi-label conflicts and transition boundaries. Preserve native semantics; the first diagnostic mapping is not a universal ontology. Audit other datasets before later inclusion.

### Q2. Does LOKI support a fair low-label benchmark?

Open. [Local population counts](DATASETS/LOKI.md#unique-pedestrian-population-2026-10-01) now establish 12,364 scenario-scoped 3D pedestrian tracks, including 4,139 without any 2D box. Independent physical identities, gaps/durations, episodes, transitions, final splits and evidence-based visibility cohorts remain incomplete. Resolve these before choosing budget units, sampling, and split manifests.

### Q14. What is the minimum usable target input and comparable representation?

Open. The first baseline needs minimal common pedestrian and ego features: settle position/velocity semantics, units, axes, ego compensation, derivative support around gaps/endpoints and source-only normalization. Yaw, dimensions and acceleration remain later ablations. Define partial-feature and missing ego handling; a valid box may not supply a valid velocity. The proposed post-encoder missing pedestrian embedding does not settle whether to expose an extra availability feature. Point-cloud/map correspondence and the final minimum sensing contract remain later questions; RGB is not universally required.

## Experimental choices

### Q5. Which target states are observable from the chosen representation?

Open. Stopped/Waiting may be only partly identifiable; Crossing needs scene relation. Start with the common framewise/BiLSTM kinematic comparison and class-wise errors before adding scene evidence. Within-dataset diagnostics must not select a model presented as strict zero-shot on that dataset.

### Q7. Which representations transfer, and which introduce domain shift?

Open. Provisional RQ1 focuses on camera-visible supervision becoming usable without visual observations. ROAD-Waymo/LOKI simultaneously change sensor, geography, scene, selection and label semantics; the first bidirectional transfer experiment cannot cleanly separate those causes. Whether RGB later serves as optional inference input, privileged training information or a full modality, and whether alignment/distillation is needed, remain open. Do not assume multimodal fusion helps.

### Q8. What temporal and sampling protocol is defensible?

Partly specified for the first diagnostic: whole variable-length tracks, retained same-identity internal gaps, independent input/GT/padding masks, and full future context for the BiLSTM. Settle regular grid rate (5 Hz is a candidate), alignment/resampling tolerances, track start/end, minimum feature validity without aggressive short-track filtering, and empty-GT handling. Scene grouping, inspected-scenario treatment, primary metrics, seed/selection rules and later label-budget accounting remain open.

### Q15. Which additional source should be tested first, if any?

Open after the first ROAD-Waymo/LOKI kinematic diagnostic. nuScenes may provide scene/sensor diversity; ROAD may provide visual behavioural diversity. Inclusion and order are undecided. IDD-PeD remains a later option. Resolve using feasibility and a named baseline failure, with controlled ablations; no extra source enters the first comparison.

### Q16. Do shared representations and native heads help without negative transfer?

Open. Provisional RQ2 asks whether motion/scene/crossing factorization improves 3D-only transfer versus a simpler shared temporal representation. Factor definitions, supervision and necessity are untested. Dataset-specific heads, joint sampling/loss weights, fusion, final temporal architecture and RGB-to-3D objectives are later hypotheses. The proposed first MLP/BiLSTM comparison does not select the final model. Measure whether source additions hurt before introducing gradient-conflict methods.

### Q18. Which transfer direction should become primary?

Open. The first matrix includes both within-dataset evaluations and ROAD-Waymo → LOKI / LOKI → ROAD-Waymo. Decide whether one direction should lead or both remain equally important after baseline evidence. An asymmetry may be consistent with different supervision populations but cannot by itself isolate visibility selection from ontology/domain/class-frequency differences. Freeze source-only selection separately for each direction and disclose later target-informed development.

### Q19. Which population should the first diagnostic include?

Open. ROAD-Waymo begins from behaviour-labeled ROAD pedestrians with official 3D pairs; minimum paired coverage is unspecified. A visibility-comparable LOKI subset is reasonable but not selected. Define usable 2D/RGB evidence at observation and track levels, retain missing-input timesteps, and report exclusions. A comparison restricted to visible tracks cannot answer RQ1 for the broader 3D-only population; metadata must support the later split.

### Q20. How should the first model comparison be controlled?

Open settings within a specified small comparison: identical pedestrian/ego features, MLP encoders and classifier design, with only Model B adding a whole-track BiLSTM. Choose compatible representation widths, report added recurrent capacity, and settle training/seed controls without giving one model a more expressive head. Feature-derived temporal context must be identical and disclosed. No code implementation is currently authorized.

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
