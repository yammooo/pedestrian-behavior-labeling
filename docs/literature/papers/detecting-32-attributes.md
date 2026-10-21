# Detecting 32 Pedestrian Attributes for Autonomous Vehicles

Mordan, Cord, Pérez and Alahi, T-ITS, online 2021. Reviewed 2026-10-13: [full v2 manuscript](https://arxiv.org/pdf/2012.02647), §§III–IV and Tables III–IV; [official implementation](https://github.com/vita-epfl/detection-attributes-fields), DOI [10.1109/TITS.2021.3107587](https://doi.org/10.1109/TITS.2021.3107587).

A single-image composite-field network jointly detects pedestrians and predicts appearance, current behavior and crossing-forecast attributes. JAAD examples have partial attribute labels; losses/gradient normalization account for the tasks supervised on each example. Fork-normalization controls shared-backbone gradient scaling as task count changes. Results compare detection and supplied-box classification as distinct evaluations and show training stability/task-dependent gains. It excludes heavily occluded pedestrians. Code is available under AGPL; pretrained PifPaf initialization is used. [Paper, §§III–IV](https://arxiv.org/pdf/2012.02647); [repo](https://github.com/vita-epfl/detection-attributes-fields).

**Masked partial-label heads, shared pedestrian features and multitask normalization are not new.** This is closer to heterogeneous-annotation learning than a generic image-classification multitask reference, though it does not establish cross-dataset RGB/3D dense-label transfer.

Use ordinary observed-label losses/native heads as the RQ2 reference. Complex conflict handling is justified only after loss/sampling changes fail. A new contribution would need controlled evidence that particular complementary semantics improve behavior transfer, not simply that several heads can be trained.

Do not copy detection matching, balanced subsampling or occlusion exclusions into our supplied-track full population. Missing supervision is not a negative label, and tasks derived from behavior GT must not enter inference inputs.

[Direction review](../offline-labeling-research-direction.md).
