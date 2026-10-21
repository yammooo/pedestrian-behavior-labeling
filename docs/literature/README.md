# Literature and contribution comparison

Updated **2026-10-15**. The [focused review and direction recommendation](offline-labeling-research-direction.md) owns the closest-work comparison, architecture tradeoffs, encoder/access checks and ranked contribution hypotheses. Eight close papers received deeper protocol reviews; additional shorter notes resolve specific mechanism, pretraining, supervision and reliability questions. The [PTv3 follow-up](papers/ptv3.md#concrete-outdoor-checkpoint-follow-up-2026-10-15) verifies concrete outdoor files/configs; [Utonia's training mixtures](papers/utonia.md#verified-pretraining-partitions-2026-10-15) include official held-out partitions. No novelty, state-of-the-art or reproduction claim. Dataset notes continue to own native facts and release findings.

**Recommended direction:** empirical offline-labeling study + established multimodal reference; one conditional complete-track-conditioned retrieval or complementary-supervision comparison. A Transformer replacement and joint raw RGB/4D architecture are not prerequisites. [Current research roadmap](../research.md#tentative-experiment-direction).

## Paper reviews

Individual paper notes live in `papers/`; cross-paper reviews remain in this directory.

| Paper | What the note resolves |
|---|---|
| [LOKI, ICCV 2021](papers/loki-method.md) | Four original current states versus shifted future intent; graph/goal supervision versus actual sensor encoding |
| [ROAD-Waymo, v3](papers/road-waymo-method.md) | Action detection/tube mAP versus given-track state F1; RGB 3D convolutions versus LiDAR input |
| [TITAN, CVPR 2020](papers/titan.md) | Recognition component, given instances and hierarchical actions, despite a forecasting title |
| [PePScenes, 2020](papers/pepscenes.md) | RGB/map/metric-track/ego precedent, pre-event population and no raw point encoding |
| [3D-keypoint crossing, ICRA 2023](papers/3d-keypoint-crossing.md) | Current LiDAR-derived crossing recognition; private behavior evaluation versus public Waymo forecasting |
| [Nighttime recognition, 2026](papers/nighttime-recognition.md) | Full supplied PDF: DINO/descriptor/Transformer precedent; sequence CLS output, three states, fixed camera |
| [CrossFeat, T-IV 2024](papers/crossfeat.md) | Pedestrian/history queries into spatial semantics; the closest retrieval mechanism and oracle-attribute caveat |
| [BiPed, ICCV 2021](papers/biped.md) | Target-specific semantic categories, independent/shared recurrent learning and future-ego-motion access |
| [MulCPred, Sensors 2024](papers/mulcpred.md) | Multimodal concepts, interpretability/accuracy tradeoffs and direction-dependent transfer |
| [Simple action recognition, STRC 2022](papers/simple-action-recognition.md) | Cheap supplied-track MLP/LSTM pose recognition and class-dependent temporal effects |
| [Stop&Go, ICRA 2022](papers/stop-and-go.md) | Future event forecasting, annotated attributes and sampled event AP versus our near-state scores |
| [ASFormer, **BMVC** 2021](papers/asformer.md) | Established offline dense labeling, hierarchy/refinement, segment metrics; corrected former ICCV citation |
| [LiMTR, 2024 workshop](papers/limtr.md) | Target-body LiDAR pooling versus surrounding scene; temporal ablation uncertainty; no released weights |
| [TubeDETR, CVPR 2022](papers/tubedetr.md) | Temporal queries revisiting spatial features; grounding is a mechanism precedent, not behavior labeling |
| [Scene Transformer, ICLR 2022](papers/scene-transformer.md) | Factorized agent/time attention and complete-trajectory conditioning already exist |
| [P4Transformer, CVPR 2021](papers/p4transformer.md) | Correspondence-free 4D point processing; outdoor timing/ego/missingness still require verification |
| [DINOv2](papers/dinov2.md) | Verified patch-feature interface, frozen RGB use and unknown recording-level web-pretraining overlap |
| [DINOv3](papers/dinov3.md) | Newer dense-feature alternative; natural-image versus satellite weights and access/provenance |
| [PTv3, CVPR 2024](papers/ptv3.md) | Point backbone versus pretrained checkpoint; serialization, output hierarchy and old-weight caveats |
| [Sonata, CVPR 2025](papers/sonata.md) | Default input color/normals; indoor/outdoor recipe distinction and frozen point features |
| [Concerto, NeurIPS 2025](papers/concerto.md) | Paired 2D/3D SSL; original model versus Utonia-derived default checkpoint names |
| [Utonia, ICML 2026](papers/utonia.md) | Missing-color/normal support; original/Small/Tiny downloads verified 2026-10-16; original Waymo/nuScenes exposure and unresolved small-model provenance |
| [OmniVGGT, 2025/CVPR 2026](papers/omnivggt.md) | RGB + optional aligned depth/cameras; no raw-LiDAR-only input; known Waymo pretraining |
| [MIT BEVFusion, ICRA 2023](papers/bevfusion.md) | Shared calibrated BEV, pretrained perception exposure and archived implementation; distinct NeurIPS namesake |
| [Zhou & Zeng, RAS 2024](papers/pedestrian-partial-supervision.md) | Full supplied PDF: four staged training sessions, not demonstrated joint multi-dataset behavior transfer |
| [Detecting 32 attributes, T-ITS](papers/detecting-32-attributes.md) | Existing partial-label pedestrian heads/gradient normalization and detection-versus-box classification |
| [MSeg, CVPR 2020](papers/mseg.md) | Semantic reconciliation and true held-out dataset controls; not a universal-ontology requirement |
| [MultiMAE, ECCV 2022](papers/multimae.md) | Aligned 2D multimodal masked learning and teacher/pseudo-label provenance |
| [ActionMAE, AAAI 2023](papers/actionmae.md) | Established missing-modality reconstruction; random dropout differs from natural missingness |
| [Calibration, ICML 2017](papers/calibration.md) | Cheap source-validation temperature scaling; no shifted-domain reliability guarantee |
| [SelectiveNet, ICML 2019](papers/selectivenet.md) | Risk/coverage, class retention and why acceptance needs independent error evidence |
| [ViCross, September 2026 preprint](papers/vicross.md) | Target-centric visual tokens/auxiliary spatial tasks; strong transfer failure despite in-domain accuracy |
| [Observe them over time, WACV 2024](papers/observe-over-time.md) | Multi-view attribute training but single-image inference; no changing-state track labeling |
| [Pedestrian Graph+, T-ITS 2022](papers/pedestrian-graph-plus.md) | Official lightweight prediction code/checkpoints; full-paper protocol remains deferred |
| [BEVContrast, 3DV 2024](papers/bevcontrast.md) | 2026-10-15 candidate screen: outdoor LiDAR self-supervision; separate SemanticKITTI/nuScenes weights and sparse-library requirements |
| [ScaLR, CVPR 2024 / ScaLR+](papers/scalr.md) | 2026-10-15 candidate screen: image-to-LiDAR distillation, published outdoor weights and inference-versus-pretraining modality distinction |
| [TARL, CVPR 2023](papers/tarl.md) | 2026-10-15 repository screen: temporal LiDAR pretraining, SemanticKITTI weights and intensity/compatibility gate |
| [DITR / D-DITR, 3DV 2026](papers/ditr.md) | Multi-level image-feature injection versus point-only inference after distillation; verified nuScenes PTv3 checkpoint/config candidates |

## Reference index

| Reference | Finding and project use |
|---|---|
| [Minimizing Human Labeling — Riaz, Wielgosz, López Peña, T-ITS 2025](papers/minimizing-human-labeling.md) | Substantive review: S2R-UDA-CP binary crossing self-labeling, smoothing, target-access accounting and independent PedGraph+ validation |
| [LOKI — Girase et al., ICCV 2021, pp. 9803–9812](papers/loki-method.md) | Recurrent trajectory/intention prediction with scene graphs and long-term goals; current actions differ from future targets. [Native evidence](../datasets/loki.md) |
| [ARCANE-PedSynth — Riaz, Wielgosz, López Peña, 2026](https://arxiv.org/abs/2605.24950) | Hybrid AI/manual CARLA controller and advertised rich FSM; claims differ from active generator/export findings. [Native evidence](../datasets/pedsynth-plusplus.md); [investigation](../archive/2026-10-01-synthetic-supervision.md) |
| [Stop and Go Forecasting — Guo, Mordan, Alahi, 2022](papers/stop-and-go.md) | Future stop/go on TRANS, hybrid fusion and sampled event AP; now reviewed in full. Motivates transition/context controls, not adoption of its oracle attribute inputs; our event protocol remains open |
| Rasouli & Kotseruba taxonomy (exact bibliographic record unverified) | Conceptual distinction among intention estimation, action prediction and risk assessment; no dataset/method adopted. Prevents future-derived labels/hindsight being called current observable behavior; primary terminology needs advisor alignment |
| [PIE — Rasouli, Kotseruba, Kunic and Tsotsos, ICCV 2019, pp. 6262–6271](https://openaccess.thecvf.com/content_ICCV_2019/html/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.html); [JAAD — Rasouli, Kotseruba and Tsotsos, ICCV Workshops 2017, pp. 206–213](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w3/html/Rasouli_Are_They_Going_ICCV_2017_paper.html) | Verified primary bibliographic records; crossing/intention and joint-attention/context references. [Native evidence and release unknowns](../datasets/pie-jaad.md); not selected sources |
| [EMT — Abdel Madjid et al., 2025](https://arxiv.org/abs/2502.19260) | Gulf-region visual tracking/forecasting/intention benchmark; [native evidence](../datasets/emt.md), no current experiment |
| [ECP2.0 — Krebs, Braun, Gavrila, TPAMI 2024, 46(12), 10929–10943](https://doi.org/10.1109/TPAMI.2024.3471170) | Offline track pseudo-GT construction, separate from behavior enrichment; [native evidence](../datasets/ecp2.md), possible later application |
| [ROAD-Waymo repository](https://github.com/salmank255/Road-waymo-dataset), [Waymo labeling specifications](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md) | [ROAD-Waymo note](../datasets/road-waymo.md): frontal population/partial linkage; E001 association accepted with qualifications, broader RGB projection remains unknown |
| [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md), [map tutorial](https://www.nuscenes.org/tutorials/map_expansion_tutorial.html) | [nuScenes note](../datasets/nuscenes.md): candidate limited motion/scene supervision |
| [ROAD paper](https://doi.org/10.1109/TPAMI.2022.3150906), [repository](https://github.com/gurkirt/road-dataset) | [ROAD note](../datasets/road.md): possible visual action/location source |
| [IDD-PeD project](https://cvit.iiit.ac.in/research/projects/cvit-projects/iddped), [repository](https://github.com/Ruthvik9/IDD-PeD) | [IDD-PeD note](../datasets/idd-ped.md): later optional visual/context source |

## Detailed motion-representation review

[PV-LSTM, OmniTraj, causal transfer and PerReg+](transferable-motion-representations.md): geometry supervision, timing-aware transfer, complementary causal supervision and parameter-efficient adaptation. The review distinguishes prediction from offline behavior labeling, source-data exposure and reported numerical inconsistencies.

## Closest neighboring work

The 2026-10-05/06 checks were preliminary. The [2026-10-13 comparison](offline-labeling-research-direction.md#closest-work-comparison) replaces their limited task/architecture assessment after full-text reading. In particular, nighttime recognition and Zhou & Zeng are no longer abstract-only; the user supplied both PDFs. The current review still does not establish exhaustive novelty or reproduce published scores.

| Work | Evidence and boundary | Implication for this study |
|---|---|---|
| [Nighttime recognition (2026)](papers/nighttime-recognition.md) | Full text: three-class sequence recognition with DINOv2 crops/descriptors/Transformer; one CLS output | An established multimodal reference; dense changing-state/missing-observation protocol is different, not automatically novel |
| [Rasouli et al. (2020), PePScenes](https://arxiv.org/pdf/2012.07773) | Full paper checked: RGB, semantic maps, 3D trajectories and ego state with modality ablations; binary crossing prediction from short pre-event observations | 3D-aware modality comparisons already exist. Our proposed study reconstructs multiple current states across complete observed tracks |
| [Yi et al. (BMVC 2021), ASFormer](papers/asformer.md) | Full method/ablations checked: offline framewise action segmentation | Offline framewise labeling and a Transformer alone do not establish novelty |
| [ViCross (September 2026 preprint)](papers/vicross.md) | Target-centric visual tokens/auxiliary supervision; JAAD_all→PIE binary F1 drops to 0.15 | Spatial supervision/transfer overlap is substantial; retain failure rather than interpreting accuracy as reliable transfer |
| [Gesnouin et al. (2022), Assessing Cross-dataset Generalization of Pedestrian Crossing Predictors](https://arxiv.org/abs/2201.12626) | Direct crossing-predictor transfer degrades despite strong ordinary train/test performance; task is future crossing prediction | Cross-dataset evaluation is necessary evidence, not novelty. Does not establish full-track offline frame-state recoverability under RGB/LiDAR/3D tracks |
| [Feng et al. (2024), MulCPred: Learning Multi-Modal Concepts for Explainable Pedestrian Action Prediction](https://www.mdpi.com/1424-8220/24/20/6742) ([author manuscript](https://arxiv.org/abs/2409.09446); [article mirror, Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510801/)) | Appearance, skeleton, trajectory, ego motion and local context; TITAN↔PIE crossing prediction drops substantially. Filtering unrecognizable concepts improves some transfer results; gains are metric/direction dependent | Supports testing modality contribution and structured representations. It does not test LiDAR-based offline reconstruction or prove motion/scene/crossing factors are necessary here |
| [Khan et al., ROAD-Waymo, v3 (2026 revision)](https://arxiv.org/abs/2411.01683v3) | Action/location/event labels and UK ROAD↔US ROAD-Waymo domain-adaptation motivation; [release evidence](../datasets/road-waymo.md) is distinct | Another-dataset evaluation alone is not a contribution. The current question concerns offline multimodal track labeling and partial, mismatched supervision |
| [MSeg (CVPR 2020)](papers/mseg.md) | Taxonomy/mask reconciliation improves held-out segmentation transfer | Semantic caution, not a requirement for one universal behavior ontology |
| [Zhou & Zeng (2024)](papers/pedestrian-partial-supervision.md) | Full text: nuScenes detection/tracking, Halpe pose, then joint JAAD attributes | Corrects the earlier implication of joint heterogeneous-dataset behavior learning; no held-out behavior transfer experiment |

## Contribution boundaries and review gaps

| Topic | Existing evidence / boundary | Current research implication | Review still required |
|---|---|---|---|
| Automatic crossing pseudo-labeling | [Riaz et al. 2025](papers/minimizing-human-labeling.md) already reports binary self-labeling and smoothing | Automatic labels/smoothing alone are not a contribution | Target access, label budgets, temporal semantics and independent audits |
| Native actions versus future intention | [LOKI](papers/loki-method.md) shifts actions; [current recognition and dense-label neighbors](offline-labeling-research-direction.md#closest-work-comparison) are now reviewed | RQ1 reconstructs current states; offline formulation alone is insufficient novelty | Native-definition/blinded evidence audit; preserve E001's accepted projection |
| Recoverability and modality contribution | [MulCPred](https://arxiv.org/abs/2409.09446) studies multimodal concepts; local LOKI examples show weak evidence | RQ1 compares evidence, classes and observation conditions; poor classification is not sensor-insufficiency proof | Comparable modality/temporal controls, human-evidence audit and limits of identifiability claims |
| Heterogeneous partial supervision | [Mordan](papers/detecting-32-attributes.md) trains partial JAAD tasks; [Zhou & Zeng](papers/pedestrian-partial-supervision.md) stages component datasets | RQ2 needs evidence beyond native heads and masked losses | One feasible source; matched generic/semantic supervision, data/compute/access controls |
| Non-identical taxonomies | [MSeg](https://openaccess.thecvf.com/content_CVPR_2020/html/Lambert_MSeg_A_Composite_Dataset_for_Multi-Domain_Semantic_Segmentation_CVPR_2020_paper.html) establishes risks of naive merging in segmentation | Preserve semantics; lightweight native heads are a candidate | Precise audit and shared/native-head comparisons without assumed equivalence |
| Scene grounding and factorization | Scene/road relation may distinguish Crossing/Waiting beyond motion | Methods under RQ1/RQ2, not separate RQs or settled factors | Road-relative/3D scene baselines, factor identifiability and simpler shared representations |
| Cross-dataset and visibility selection | [Gesnouin et al.](https://arxiv.org/abs/2201.12626), [MulCPred](papers/mulcpred.md) and [ViCross](papers/vicross.md) report prediction-transfer failures | ROAD-Waymo↔LOKI is useful but geography/sensors/semantics/population confound asymmetry | Population-aware controls and exact encoder/teacher recording exposure |
| Label efficiency, negative transfer and selective labeling | More supervision can conflict; reliable coverage needs independent evidence | Equal target-label access, retained negative ablations, uncertainty only after error analysis | Budget/compute accounting, learning-curve variability, calibration and independent audit |

The [ranked contribution hypotheses](offline-labeling-research-direction.md#strongest-contribution-candidates) are empirical/benchmark evidence first, one conditional complete-track-conditioned mechanism second, and semantically controlled heterogeneous supervision third. These are recommendations under the deadline, not a withdrawal of RQ2 or an established novelty claim. Established ingredients, exact differences, minimal controls and falsifiers are stated in the review. E001 is the preserved diagnostic foundation, not the final method or contribution.

[ARCANE-PedSynth](../datasets/pedsynth-plusplus.md) remains reported rich-synthetic motivation with limited inspected usability. Its former label-efficiency hypothesis is preserved in the [synthetic investigation](../archive/2026-10-01-synthetic-supervision.md); revisit only with new suitable supervision. Missing reviews are not evidence of novelty.
