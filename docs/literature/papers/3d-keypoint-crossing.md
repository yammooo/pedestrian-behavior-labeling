# Pedestrian Crossing Action Recognition and Trajectory Prediction with 3D Human Keypoints

Li et al., ICRA 2023. Reviewed 2026-10-13: [full eight-page author manuscript](https://arxiv.org/pdf/2306.01075), §§III–V and Tables I–III. The paper's distinction between its private and public evaluations is essential.

## Verified paper evidence

The model recognizes **current binary crossing** from two seconds of history and forecasts four seconds of trajectory. A LiDAR pose estimator produces 3D keypoints. An ST-GCN encodes joint sequences, an RNN encodes the target track, and vectorized agent/road polylines interact through attention. Crossing, trajectory, keypoint forecasting, temporal-jigsaw and contrastive objectives share representations. Private-data crossing F1 rises from 0.935 for recognition alone to 0.966 with all objectives. **Public Waymo has no crossing labels in this study: the crossing branch is removed and only trajectory prediction is evaluated.** The private roadgraph is also absent in that public evaluation. [Paper, §§III–V](https://arxiv.org/pdf/2306.01075).

## Relevance and novelty boundary

This is the strongest reviewed counterexample to claims that LiDAR-based pedestrian behavior recognition, target-conditioned road context or auxiliary temporal supervision is new. It addresses current crossing, not only future intention. It nevertheless does not establish full-track, four-state, public cross-dataset labeling with incomplete RGB.

The reported private F1 is not a public Waymo behavior benchmark score. It cannot be used as a numerical baseline for E001, and the large private training population differs substantially from ours. No usable end-to-end behavior checkpoint or complete official reproduction package was established in this review; the manuscript alone is not an implementation.

## Feasible adaptation

A small target-centered point/geometry encoder is a more viable initial reference than recreating the entire pose pipeline. Use 3D keypoints only if an actual estimator, its training exposure and accuracy on sparse LOKI/ROAD clouds can be verified. Missing joints must be explicit; an inferred pose is not annotation GT. Creating pseudo-keypoints cannot establish recoverability without independent checks.

Auxiliary forecasting is a sensible **control** for RQ2, not an inherently novel objective. It can be compared with behavior-only training on exactly the same observations before adding a semantically different dataset.

## Falsification

A mechanism claiming fine-grained pose benefit needs comparison with kinematics, raw target point features and scene features, matched in temporal access. If gains disappear when observation completeness or model capacity is controlled, pose-specific reasoning is not established. Current label errors may also arise from road relations rather than body articulation.

[Direction review](../offline-labeling-research-direction.md).
