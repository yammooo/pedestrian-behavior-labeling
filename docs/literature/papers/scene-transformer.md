# Scene Transformer: A Unified Architecture for Predicting Multiple Agent Trajectories

Ngiam et al., ICLR 2022; reviewed 2026-10-13 using [arXiv paper/supplement](https://arxiv.org/pdf/2106.08417), §§3–4 and attention/masking ablations.

Agent states remain an agent × time × feature tensor. Alternating attention over agents/time combines road/traffic context through cross-attention. Masks specify observed and requested trajectories; conditional prediction can observe another agent's future or a complete AV trajectory. Evaluation concerns joint/marginal trajectory prediction on Argoverse/Waymo, not dense behavior labels. The architectural and masking experiments support efficient joint modeling. No ready-to-use pedestrian-state checkpoint was established in this review. [Paper, §§3–4](https://arxiv.org/pdf/2106.08417).

Complete-trajectory conditioning, masked future access and factorized spatiotemporal reasoning are therefore **established principles**. Our possible difference is their controlled use for reconstructing current pedestrian states and selecting scene evidence across a recorded track, rather than predicting hypothetical agent futures.

Adapting the state encoder/head is possible, but an all-agent forecasting model is unnecessary for the next increment. Missing observations need validity/timestamps distinct from requested-output masks. An entire track with intermittent input can still require labels at every context timestep.

Use a capacity-matched model receiving full trajectory summaries as a control before attributing gains to a new retrieval mechanism. Frozen scene features and a small temporal query layer offer a lower-cost experiment than joint all-agent/all-point training.

[Direction review](../offline-labeling-research-direction.md).
