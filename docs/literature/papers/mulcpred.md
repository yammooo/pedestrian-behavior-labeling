# MulCPred: Learning Multi-Modal Concepts for Explainable Pedestrian Action Prediction

Feng et al., *Sensors* 24(20), 6742 (2024). Reviewed 2026-10-13: [author manuscript](https://arxiv.org/pdf/2409.09446), §§III–IV and Tables I–III; [published article](https://www.mdpi.com/1424-8220/24/20/6742). The verified [official implementation](https://github.com/Equinoxxxxx/MulCPred_code) supplies training and C3D/HRNet backbone-weight links; a trained MulCPred behavior checkpoint was not verified. The repository URL includes `_code`.

## Verified paper evidence

The model predicts crossing on TITAN/PIE and atomic actions on TITAN. Appearance, skeleton, trajectory, ego motion and local context branches produce features; recalibration yields concept activations aggregated by a linear classifier. Concept-localization and diversity losses encourage distinct patterns. Its inputs use short observation sequences, not complete dense-label tracks. Table III reports crossing transfer F1 0.37 for TITAN→PIE and 0.34 for PIE→TITAN; filtering uninterpretable concepts changes these to 0.39 and 0.45. Gains depend on metric and direction. Regularization can improve concept diversity while reducing in-domain predictive scores. [Paper, §§III–IV](https://arxiv.org/pdf/2409.09446).

## Interpretation for our project

Multimodal factors, concept explanations and crossing transfer already have substantive precedents. Our earlier `z_motion / z_scene / z_crossing` hypothesis cannot be described as novel merely because the factors have interpretable names. A learned concept is also not an identifiable physical or causal variable.

The cross-dataset results support retaining negative transfer and class-specific analysis. They do not demonstrate that an explanatory architecture reliably transfers, nor that human-recognizable concepts necessarily retain all predictive information. Some apparent interpretability improvements trade off task performance.

A full MulCPred reproduction would add pose extraction, concept losses and interpretability machinery before we know that our evidence is adequate. It is therefore a positioning reference rather than the first implementation choice. A simple appearance/context/geometry temporal model is a lower-cost baseline for establishing useful information.

## Possible decisive experiment

If heterogeneous supervision is pursued, compare ordinary shared embeddings/native heads with a proposed factorized representation while holding source observations, behavior labels and compute fixed. Report native auxiliary performance and behavior transfer, including negative effects. Concepts should earn their interpretation through independent probes or interventions with justified semantics; attention heatmaps alone do not establish causality.

Published binary F1 is not our equal-track four-class score. Camera-oriented population differences and exact future-label protocols must remain explicit in any adaptation.

[Direction review](../offline-labeling-research-direction.md).
