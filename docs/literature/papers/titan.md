# TITAN: Future Forecast Using Action Priors

Malla, Dariush and Choi, CVPR 2020. Reviewed 2026-10-13: [full paper](https://arxiv.org/pdf/2003.13886), especially §§3–5; [official dataset page](https://usa.honda-ri.com/titan).

## Verified paper evidence

TITAN supplies hierarchical per-frame actions, including pedestrian atomic and contextual actions, over tracked RGB video with ego motion. Its method first recognizes actions from target image-patch histories with I3D/3D ResNet, then uses action, interaction and ego priors for forecasting. Detection is supplied externally. Recognition uses multiple heads and uncertainty-weighted task losses; forecasting uses recurrent target/neighbor interactions. The recognizers are pretrained on Kinetics-600. Action evaluation is reported per frame; future localization uses ADE/FDE/FIOU. The main forecasting variant reports 11.32-pixel ADE versus 17.79 without action priors. This comparison does not isolate a new behavior labeler. [Paper, §§4–5](https://arxiv.org/pdf/2003.13886).

## Why it is closer than a generic crossing predictor

The recognition component already labels several current pedestrian attributes on supplied instances. Consequently, neither “given pedestrian tracks,” “multiple states,” nor “temporal pedestrian crops” is individually new. Although forecasting dominates the title, excluding TITAN from the recognition comparison would exaggerate our novelty.

An adapted I3D or modest crop encoder with a per-frame head is a fair architectural reference once temporal support and access are matched. The original action-prior forecasting decoder is unnecessary for our output. An alternative inexpensive pose reference is [Xiong et al.](simple-action-recognition.md).

TITAN's attributes are not a ready-made replacement for our four-state semantics. Walking, crossing and waiting relationships must be checked before any heterogeneous supervision proposal. Action GT used as forecasting input must not become input to a labeler whose purpose is to generate those actions.

## Interpretation and limits

RGB-visible annotations do not establish performance on 3D-only pedestrians. Recognition metrics, action groups, supplied-box assumptions and training preconditions need matching before numerical comparison. Kinetics pretraining is external supervision and must be declared, even though it is not a Waymo/LOKI behavior checkpoint.

An empirical contribution could demonstrate which contextual states benefit from full-track scene evidence beyond these established crop-history recognizers. It would be falsified as a methodological claim if an ordinary bidirectional crop model performs equally well under the same evidence and capacity.

[Direction review](../offline-labeling-research-direction.md).
