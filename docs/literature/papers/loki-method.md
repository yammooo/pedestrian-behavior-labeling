# LOKI: Long Term and Key Intentions for Trajectory Prediction

Girase et al., ICCV 2021, pp. 9803–9812. Reviewed 2026-10-13: [full paper and supplement](https://arxiv.org/pdf/2108.08236), particularly §§3–5 and Tables 2–3; [official release description](https://usa.honda-ri.com/loki). Native fields and our release limitations belong in [the dataset note](../../datasets/loki.md).

## Verified paper evidence

The annotations include the four pedestrian states used by E001, but the published task forecasts trajectories and intentions. It observes three seconds and predicts five; intention targets are constructed by shifting current action labels by 0.8 seconds. An observation RNN, CVAE goal proposals and a recurrent scene graph combine target motion, other agents and contextual information. The baseline is not an RGB–LiDAR feature encoder merely because the dataset contains those sensors. Intent conditioning and scene graphs improve some trajectory results: pedestrian single-sample ADE changes from 0.61 to 0.55 m in Table 2. The multiple-future pedestrian results are not uniformly improved by intent conditioning. These are forecasting errors, not dense behavior F1. [Paper, §§3–5](https://arxiv.org/pdf/2108.08236).

## Consequences for our comparison

LOKI is the closest annotation ontology, but its published target differs from ours. A meaningful adaptation would predict **unshifted current actions at every recorded timestep**, allow recorded future observations, and remove any behavior-label history supplied as input. Ground-truth goals or intended destinations would also be unavailable under our inference contract. Such an adaptation must be named and documented; the original trajectory scores cannot serve as our recognition baseline.

Scene graphs provide a precedent for target–agent–road relationships. We should first distinguish useful context from convenient native semantic annotations: a road label provided as supervision is not an automatically deployable scene feature. Our reader retains native annotations, but the current model contract does not permit behavior GT as evidence.

The paper weakens two possible novelty claims: four-state pedestrian understanding already exists, and joint temporal interaction reasoning already exists. A defensible difference would concern **empirical reconstruction over complete tracks, incomplete sensor observations and transfer between annotation populations**, with established baselines and a semantic audit.

## Baseline and falsification

Use a modest target/neighbor geometry representation as an optional relation baseline after identifying interaction errors. Do not reproduce the CVAE forecasting decoder just to obtain a behavior head. If a kinematic model plus simple road-relative quantities explains the same errors, a larger scene graph has no demonstrated necessity. Distinguishing Stopped from Waiting requires auditing native definitions and examples, not assuming these labels are identifiable from motion.

[Direction review](../offline-labeling-research-direction.md) · [E001/E001b evidence](../../../experiments/E001b-2d-bbox-contribution/results-review.md).
