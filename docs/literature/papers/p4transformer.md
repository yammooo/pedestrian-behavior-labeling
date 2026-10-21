# Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos

Fan, Yang and Kankanhalli, CVPR 2021. Reviewed 2026-10-13: [full paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf), §3 and experiments; [official implementation](https://github.com/hehefan/P4Transformer).

P4Transformer uses local 4D point convolution followed by attention over spatial/temporal features, without explicit point correspondence across frames. Heads perform clip action recognition or point-level semantic segmentation. Tests use MSR-Action3D, NTU RGB+D and Synthia; MSR accuracy is 90.94% with 24 frames. Action inputs do not require color. A supplied implementation includes model/data/training instructions; a general driving behavior checkpoint was not verified. [Paper, §§3–4](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf).

This establishes direct joint point-space/time encoding and weakens claims that using 4D tokens itself is novel. It avoids **point** tracking, not the supplied pedestrian-track contract. A correspondence-free architecture does not automatically handle asynchronous measurements, empty scans or mislabeled agent associations.

Moving-ego LiDAR would need verified pose compensation before static neighborhood geometry could be interpreted across time. Target-centered moving crops and ego/world-centered environmental context serve different purposes: aligning the pedestrian can erase absolute road relationships unless they are retained separately.

Treat it as an architectural boundary, not a required implementation. Indoor depth clips and synthetic segmentation do not establish an advantage over per-frame point/BEV encoding for sparse outdoor pedestrians. A decisive pilot would compare joint local processing with temporal modeling of equally sized per-frame features, on identical points/times and with physical geometry retained. Proceed only after a cheaper scene branch demonstrates useful evidence.

[Direction review](../offline-labeling-research-direction.md).
