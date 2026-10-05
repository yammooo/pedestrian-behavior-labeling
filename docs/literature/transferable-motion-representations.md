# Transferable motion representations and offline behavior labeling

Reviewed 2026-10-05 against the current [research questions](../research.md#research-questions). These works forecast motion; none directly establishes recoverability of LOKI-style offline behavior under our full sensor contract. Methods below are possible controlled comparisons, not adopted architecture.

## Sources and reading scope

- [PV-LSTM repository README](https://github.com/vita-epfl/bounding-box-prediction/blob/master/README.md): read the complete README, not a code/reproduction audit. It covers *Pedestrian Intention Prediction: A Multi-task Perspective* (2020) and *Pedestrian 3D Bounding Box Prediction* (2022).
- [OmniTraj publisher record](https://doi.org/10.1016/j.trc.2026.105971): local `S0968090X26004572.html` contains metadata, highlights and abstract, not full methods/results. Detailed assessment uses the [July 2025 arXiv v1 including supplement](https://arxiv.org/html/2507.23657v1); it is not a version-identical audit of the journal article. The saved record lists January 2027 issue placement and 2026 copyright/DOI.
- Rahimi et al., CVPR 2025, *Sim-to-Real Causal Transfer: A Metric Learning Approach to Causally-Aware Interaction Representations*, [official paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Rahimi_Sim-to-Real_Causal_Transfer_A_Metric_Learning_Approach_to_Causally-Aware_Interaction_CVPR_2025_paper.pdf), DOI `10.1109/CVPR52734.2025.01610`: read the supplied 11-page PDF and checked method/results figures visually. Referenced Appendices A–C are not included in this copy.
- Messaoud, Cord and Alahi, CVPR 2025, *Towards Generalizable Trajectory Prediction using Dual-Level Representation Learning and Adaptive Prompting*, [official paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Messaoud_Towards_Generalizable_Trajectory_Prediction_using_Dual-Level_Representation_Learning_and_Adaptive_CVPR_2025_paper.pdf), DOI `10.1109/CVPR52734.2025.02567`: read the supplied 11-page PDF and checked architecture/results visually. Its separate supplement is not included.

Local sources remain in `/home/yammo/Downloads/`; datasets, papers and generated review media are not committed.

## PV-LSTM: inexpensive geometry supervision

The README describes recurrent encoder/decoder forecasting of 2D boxes plus crossing intention or 3D boxes plus attributes, with JAAD, JTA and nuScenes support. Inputs are bounding-box annotations; RGB/video is only for visualization. Sensor presence in a dataset is not evidence that the model consumes those sensors. Supporting three datasets does not establish cross-dataset transfer, joint training or native-label compatibility.

Useful analogy: geometry forecasting can be an auxiliary representation task before behavior labels are used. It cannot supply Crossing/Waiting semantics by itself. Image-box changes also encode projection, range and camera/ego motion, so any comparison with metric 3D motion needs controlled coordinates and temporal support. The README targets Python 3.7/CUDA 10.1-era environments; compatibility with this project's Python 3.11 is untested. It is a reference, not a reason to restore the previous PV-LSTM committee.

## OmniTraj: temporal transfer is its own failure mode

ArXiv v1 separates unseen timing from unseen scenes: an NBA timing-only test trains at 5/2.5 FPS and tests at 1 FPS. An MLP embeds FPS and adds it to input tokens. Table 2 reports minADE20/minFDE20 of 1.87/2.49 without FPS encoding versus 1.18/1.22 with additive encoding; adding an extra token is weaker. Its heterogeneous inputs are trajectories, boxes and pose, not raw RGB/LiDAR. Multimodal pretraining can support trajectory-only inference. SDD/Trajnet++ are excluded from pretraining; the 859-hour framework includes those held-out datasets, so that total is not the actual pretraining exposure. Full training uses six H100s.

Implication: isolate timing/alignment effects before interpreting ROAD-Waymo↔LOKI domain failures. Nominal FPS is insufficient for irregular gaps; physical timestamps, elapsed time and feature derivative support need explicit handling. Borrowing a pretrained checkpoint also requires auditing its data exposure against each declared target access regime. The reported >70% improvement is a particular prediction-transfer result, not a behavior-labeling expectation.

## Causal transfer: complementary tasks rather than naive data mixing

The paper defines an agent's causal effect through paired ORCA simulations: remove one neighbor, rerun the dynamics and measure the focal agent's trajectory change. Its diagnosis distinguishes non-causal, directly causal and indirectly causal neighbors. Human camera-view judgments can miss influence transmitted through a mediator. Moreover, individually negligible removals need not remain negligible when combined. In this literature, "ego" can mean the focal pedestrian, not the recording vehicle.

A projection head maps factual/counterfactual scene representations to vectors. Their cosine distance should grow with simulated effect. Binary contrastive supervision distinguishes negligible from significant effects; a margin ranking loss preserves ordering of effect strengths. This is supervised metric geometry, not discovery of an identifiable causal graph or motion/scene/crossing factors.

Training combines real forecasting loss with simulated causal regularization. It needs real trajectory targets but no real causal annotations. Baselines include naive synthetic/real forecasting mixtures and non-causal removal augmentation. Figure 8 favors causal ranking, while ordinary mixing sometimes hurts. The 25%-data claim concerns real trajectory training fractions, not measured behavior annotation effort. NBA Table 2 gives baseline ADE/FDE 0.562/1.271 versus ranking 0.544/1.235 metres: useful but modest gains. Figure 6 averages five seeds; simulated OOD gains and remaining causal errors should temper broad causal claims.

For RQ2, this is evidence that an auxiliary task with complementary supervision can help more than extra examples alone. It also suggests a narrow possible synthetic role independent of rich PedSynth++ FSM labels; no simulator work is selected. Collision-avoidance simulation does not directly supervise traffic-light rules, curb/barrier semantics or Waiting intent. Context-removal tests need a justified intervention/label policy; nuisance invariance must not suppress informative road or traffic evidence.

## PerReg+: shared scene learning and small adaptation modules

Experiments use **vehicle trajectories only**, with agent states and vector road polylines. "Multimodal" primarily means multiple possible futures, not the project's RGB/LiDAR sensor mixture. nuScenes, AV2 and WOMD supply 32k, 180k and 1.8M samples; history is 2 seconds and forecast is 6 seconds. Multi-dataset training on all three is distinct from the WOMD→nuScenes transfer in Table 2.

Perceiver IO encodes many input tokens into fixed-size latents. An EMA teacher sees unmasked inputs including future trajectories; a masked student aligns a scene-level distribution with the teacher. Segment queries reconstruct agent trajectories and lanes. Forecasting, reconstruction and distillation losses are combined. Register queries provide latent decoder workspace rather than additional forecast proposals.

For adaptation, a learned scene-clustering head selects prompts; the main architecture is frozen while prompts **and the prediction head** are optimized. Clusters are learned groupings, not verified behavior concepts. Keeping a forecasting decoder can preserve pretrained function; a new framewise behavior head changes the task, so decoder-retention benefits cannot be assumed.

Table 2 reports WOMD→nuScenes B-FDE 3.12 without pretraining versus 2.75 with it, approximately 11.9% lower. Table 3's cumulative ablation gives 2.64→2.62 for prompt tuning (0.8%); the combined pipeline's gains should not be attributed to prompts alone. Reporting discrepancies merit code/supplement checks: Table 3 has masked-SD 2.64 while prose says 2.76; Table 1's nuScenes 3.06→2.62 is about 14.4%, whereas §4.2 says 11%. The training detail also mentions 64 GMM modes while the ablation describes six forecast queries plus registers. Do not copy settings without resolving these details.

## Consequences for this project

Keep E001 as the simple kinematic diagnostic. Later, if justified, compare motion/scene pretraining against scratch with equal behavior labels and access; use frozen-encoder behavior probes before adding prompts or full fine-tuning. A representation that predicts trajectory well may still fail to distinguish road-relative Crossing or Stopped/Waiting.

Report separately the benefit of auxiliary supervision, inference-time evidence, data quantity, timing normalization and target adaptation. Missing-observation reconstruction is useful training supervision, never recovered ground truth. Full-track future context is permitted offline, but future access and masking must be defined consistently per comparison. Human uncertainty, forecasting multi-future uncertainty, classification uncertainty and annotation disagreement are different objects.

No study here establishes a universal behavior ontology, intrinsic ambiguity from low accuracy, or automatic novelty for our proposed shared representation. Empirical RQ1 characterization and semantically controlled RQ2 behavior transfer remain to be tested.
