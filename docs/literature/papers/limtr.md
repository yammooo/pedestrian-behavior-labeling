# LiMTR: Time Series Motion Prediction for Diverse Road Users through Multimodal Feature Integration

Oerlemans et al., NeurIPS 2024 Time Series workshop. Reviewed 2026-10-13: [full paper](https://arxiv.org/pdf/2410.15819), §§3–4 and Tables 1–2; [official implementation](https://github.com/Cing2/LiMTR).

Target 3D boxes, expanded by 15%, select up to 512 LiDAR points per historical frame. Points are centered/rotated around the target and include intensity/class information. PointNet-style layers pool spatially, then flatten/compress eleven history frames into one feature integrated into MTR. It forecasts trajectories rather than states. Waymo validation minADE changes from 0.6812 to 0.6389; mAP from 0.4029 to 0.4093. Its timestep ablation does **not** establish a clear winner: seed variance and metric disagreement remain. The official repo supplies code but explicitly withholds pretrained weights. [Paper, §§3–4](https://arxiv.org/pdf/2410.15819); [repo](https://github.com/Cing2/LiMTR).

Target-centered raw LiDAR encoding is established. However, **a tight body crop is not a local environmental crop**: it can discard curbs, road layout and nearby agents. That distinction is central to our Stopped/Waiting question. A small PointNet or BEV branch should distinguish target-body from surrounding-scene evidence.

The paper's gaze/pose explanation is a hypothesis inferred from forecasting gains, not direct pose-accuracy evidence. Its 12-A100 training setup is not an estimate for our lightweight adaptation. We should not reproduce MTR's forecasting decoder or scale merely to add point features.

Use supplied native associations, masks and verified transforms. Avoid treating padded zero points as measurements; dense behavior output requires retaining timestep resolution instead of flattening the complete track. Pretraining on Waymo would require an exact exposure audit for both target transfer and our source-held-out clips.

[Direction review](../offline-labeling-research-direction.md).
