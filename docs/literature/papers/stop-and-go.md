# Pedestrian Stop and Go Forecasting with Hybrid Feature Fusion

Guo, Mordan and Alahi, ICRA 2022. Reviewed 2026-10-13: [full paper](https://arxiv.org/pdf/2203.02489), TRANS construction, §IV and Tables II–IV.

TRANS derives Stop/Go candidates from JAAD/PIE/TITAN motion annotations, requiring pre/post states lasting more than 0.5 seconds. The model forecasts a transition using past video, motion and behavioral/scene attributes, with CNN/LSTM hybrid fusion. Evaluation reports AP on balanced negative sampling averaged over ten trials. Go AP improves by 9.5 points on JAAD versus its best video baseline; gains for Stop are weaker. **GT boxes and attributes are supplied in the reported experiments.** [Paper, §§III–IV](https://arxiv.org/pdf/2203.02489).

This is future **event forecasting**, not our near-boundary state F1 or an offline transition detector. Event AP cannot fill the missing transition-event metric in E001. If event evaluation is added later, onset definitions, GT-gap handling and matching tolerance need their own frozen protocol.

Its modality and temporal ablations already establish relevant precedent. For an adapted labeler, remove behavioral GT from inputs; relying on an external attribute recognizer introduces both cost and error. A supplied motion-state flag is especially problematic if the labeler predicts Moving/Stopped.

Use the paper to motivate a small context-duration control and honest transition analysis, not a new Stop&Go architecture. More future context may help retrospective labels but cannot prove that Waiting denotes observable rather than inferred intent.

[Direction review](../offline-labeling-research-direction.md).
