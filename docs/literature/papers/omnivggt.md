# OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer

Peng et al., November 2025 preprint / CVPR 2026. Reviewed 2026-10-13: [paper and appendix](https://arxiv.org/html/2511.10560v1), §§3–5 and dataset details; [official implementation](https://github.com/Livioni/OmniVGGT-official); [model repository](https://huggingface.co/Livioni/OmniVGGT).

OmniVGGT processes **sets of RGB images**, optionally with image-aligned depth maps/validity masks and camera intrinsics/poses. GeoAdapters inject geometric cues into alternating within-frame/global attention. Outputs include camera parameters, depth, point maps and confidence. Stochastic auxiliary-modality availability is trained explicitly. Experiments cover geometric reconstruction and a robotic-policy integration, not pedestrian behavior. Training includes **Waymo among nineteen listed datasets**; Table 8's caption says twenty but enumerates nineteen. Official instructions load `OmniVGGT.safetensors`. [Paper, §§3–5/appendix](https://arxiv.org/html/2511.10560v1); [repo](https://github.com/Livioni/OmniVGGT-official).

Raw unorganized LiDAR is **not** a direct documented input. Turning it into camera-aligned depth requires intrinsics/extrinsics, synchronization and visibility handling. Optional camera input means the model can estimate geometry from RGB; it does not verify our missing calibration or create trustworthy measurements for 3D-only pedestrians. RGB remains essential.

Spatial tokens offer a potential feature interface, but behavior semantics, dynamic-agent identity, metric scale, arbitrary track gaps and usable full-track memory are unvalidated for us. Internal token extraction and any feature-only configuration must be checked against the exact code/weight pair; the inspected quick-start documents geometry outputs, not a guaranteed drop-in feature API. Listing a checkpoint is not a successful extraction test.

Known Waymo pretraining prevents treating a borrowed checkpoint as a clean dataset-excluded LOKI→ROAD reference. Exact E001 scene overlap remains unknown and may also affect ROAD within-domain evaluation. Removing a behavior head or freezing the encoder does not remove inherited exposure.

Do not make OmniVGGT or calibration reconstruction a December dependency. It is a useful joint-view/adapter precedent; independent RGB and 3D branches provide a cheaper experiment under our verified observations.

[Direction review](../offline-labeling-research-direction.md).
