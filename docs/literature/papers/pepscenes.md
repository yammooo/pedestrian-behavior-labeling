# Pedestrian Behavior Prediction on a 3D Benchmark Dataset

Rasouli et al., 2020; PePScenes. Reviewed 2026-10-13: [six-page paper](https://arxiv.org/pdf/2012.07773), §§3–5 and Table 2. This extends nuScenes; it is not the original nuScenes motion-attribute taxonomy.

## Verified paper evidence

PePScenes adds crossing annotations to 719 selected pedestrians, with per-frame current crossing states and eventual crossing labels. The selection favors front-camera pedestrians relevant to a potential crossing. Its predictor combines full front-camera scene images, rasterized semantic maps, metric pedestrian trajectories and ego velocity. CNN scene/map features are globally pooled; separate LSTMs encode motion. It consumes no raw LiDAR point features. Experiments observe 0.5 seconds and sample 1–2 seconds before crossing. Table 2 reports binary F1 0.48 with all inputs versus 0.20 for the trajectory LSTM; accuracy is 0.87. The dense box augmentation includes interpolation and detection-based adjustment. [Paper, §§3–5](https://arxiv.org/pdf/2012.07773).

## Our interpretation

This is a close precedent for **RGB, metric 3D trajectories, ego motion and scene semantics**. Claiming that this combination or a modality-contribution study is new would be unjustified. Native LiDAR in nuScenes should not be confused with LiDAR consumption by the method.

The substantive experimental difference is current four-state reconstruction over complete observed tracks and a population retaining 3D-only pedestrians. Its pre-crossing binary F1 is not comparable to our four-class track-weighted F1. Its camera/intent-oriented population is also a confound, not an innocuous preprocessing choice.

The spatial-first representation is an appropriate low-cost reference: separate motion and visual/context branches followed by temporal integration and dense readout. Adapt it using our observed frame references, masks and source-only normalization. Do not copy its interpolation policy or substitute oracle semantic maps if equivalent information is unavailable in LOKI.

## Minimal decisive use

Compare kinematics with target appearance and local context while holding the temporal head fixed. Only then investigate spatial token retrieval. If pooled context solves the dominant errors, a joint spatiotemporal mechanism is not required. If it fails, demonstrate that relevant scene evidence is actually present before attributing failure to pooling.

PePScenes is not automatically an additional RQ2 source: release access, taxonomy, population and supervision coverage would need a separate gate. No new dataset preparation is authorized by this review.

[Direction review](../offline-labeling-research-direction.md).
