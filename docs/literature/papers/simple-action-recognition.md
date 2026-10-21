# Simple Yet Effective Action Recognition for Autonomous Driving

Xiong, Bertoni, Mordan and Alahi, STRC 2022. Reviewed 2026-10-13: [full conference report](https://www.strc.ch/2022/Xiong_EtAl.pdf), §§2–3 and Tables 2–3; [author implementation](https://github.com/vita-epfl/pose-action-recognition). This is a conference report, not a large-scale multimodal foundation model.

OpenPifPaf produces 2D joints; a small feedforward encoder predicts actions, optionally followed by an LSTM and parallel native action heads. Temporal sequences associate poses using supplied track IDs. Tests cover TCG gestures and TITAN actions. On TITAN's selected atomic subset, temporal pose improves Walking/Standing AP to 97.83/73.02%, but its mean AP is 73.78% versus 79.11% for single-frame pose. The repository exposes both GT-track and predicted-track options and links extracted poses/a trained TITAN model. [Paper, §3](https://www.strc.ch/2022/Xiong_EtAl.pdf); [implementation instructions](https://github.com/vita-epfl/pose-action-recognition).

This is a particularly inexpensive **adaptable given-track recognition baseline**. It also demonstrates why a headline about temporal benefit needs class-level qualification. Pose can recognize articulation while discarding road relationships needed for Crossing/Waiting.

For our benchmark, replace the taxonomy/head and explicitly allow bidirectional context. Missing/low-quality pose must remain masked and cannot exclude 3D-only tracks. Native 2D pose is not currently an input cache; estimator feasibility is an additional cost. Do not make this a required modality before inspecting whether body articulation explains our errors.

The paper/repository subsets and detection recall differ from our full eligible population; their AP is not four-class equal-track F1. An adapted pose encoder would establish a modality reference, not reproduce the published score.

[Direction review](../offline-labeling-research-direction.md).
