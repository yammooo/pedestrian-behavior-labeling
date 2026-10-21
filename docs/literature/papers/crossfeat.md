# CrossFeat: Semantic Cross-modal Attention for Pedestrian Behavior Forecasting

Marchetti et al., *IEEE Transactions on Intelligent Vehicles*, 2024, DOI [10.1109/TIV.2024.3449046](https://doi.org/10.1109/TIV.2024.3449046). Reviewed 2026-10-13: [full author manuscript](https://flore.unifi.it/retrieve/8e7f8954-3b4e-42b9-a400-9b9ca63c799b/CrossFeat_Semantic_Cross-modal_Attention_for_Pedestrian_Behavior_Forecasting.pdf), §§III–V, Figure 3 and Tables III–VI.

## Verified paper evidence

CrossFeat forecasts binary Crossing, Stop or Go from historical pedestrian observations. The current RGB image is converted to a semantic segmentation map, encoded into **spatial tokens**, and enriched through self-attention. Target ROI features, motion/behavior-history GRUs, scene attributes and ego velocity become queries into those tokens through cross-attention. The final head outputs one probability. Reported tests use JAAD/PIE crossing and JAAD/PIE/TITAN Stop&Go; the latter observes one second at 5 Hz. Table III reports crossing F1 0.81 on PIE. The semantic teacher uses Mapillary Vistas; body/scene attributes can require extra annotation or predictors. [Paper, §§III–V](https://doi.org/10.1109/TIV.2024.3449046).

## Closest methodological overlap

This already performs pedestrian-conditioned retrieval **before final compression**, with a temporally encoded query. A proposed target-motion query attending to scene patches must compare directly with this principle. Renaming its mechanism as track queries, contextual fusion or behavior-centric attention would not establish novelty.

An important difference is that CrossFeat retrieves from the current scene frame after compressing past motion. Our candidate could derive a per-frame query from the **complete recorded trajectory** and revisit retained scene tokens from other timesteps before dense labeling. That is a plausible gap to test, not an established first: [TubeDETR](tubedetr.md), [Scene Transformer](scene-transformer.md) and [ViCross](vicross.md) further constrain the claim.

## Adaptation requirements

Keep RGB patch or semantic tokens, box-relative geometry and target kinematics; change the prediction head to dense current states and make temporal support explicit. Remove ground-truth behavioral attributes that overlap our predicted states. Scene semantic labels and native behavior fields must not become oracle inputs. If a segmentation teacher is used, report its provenance and errors separately from the fusion mechanism.

No runnable author implementation or behavior checkpoint was verified from the inspected manuscript/project records. This makes a **clearly described adaptation of its attention pattern** more realistic than a claimed exact reproduction. It still deserves a baseline even if implemented with ordinary PyTorch attention and the same frozen features as our candidate.

## Decisive comparison

Use identical cached spatial features for pooled fusion, frame-conditioned queries and full-track-conditioned queries. All offline models receive complete observations. Match parameter/feature budgets, include a full-track summary added to pooled fusion, and inspect class/condition effects. Without those controls, an improvement can be due to more information, parameters or a stronger encoder rather than delayed scene retrieval.

[Direction review](../offline-labeling-research-direction.md).
