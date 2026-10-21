# Towards Comprehensive Understanding of Pedestrians for Autonomous Driving

Zhou and Zeng, *Robotics and Autonomous Systems* 171 (2024), 104580; [DOI](https://doi.org/10.1016/j.robot.2023.104580). Reviewed 2026-10-13 using the **full user-supplied 13-page PDF**, particularly §3.4, Figure 3.4 and Tables 4.1/4.5. Online publication was November 2023; issue year is 2024.

The pipeline combines camera BEV detection/tracking with body/face/pose/context attribute branches. Look/Cross heads optionally use history GRUs. Crucially, training comprises **four sessions**: detection on nuScenes, tracking on nuScenes, pose on Halpe-Fullbody, then joint training of thirteen attributes on JAAD. The paper reports JAAD mAP 66.6 versus 42.4 for its separately trained ResNet baseline, but inputs also differ. It describes possible transfer by freezing layers; it does not evaluate held-out cross-dataset offline behavior transfer or one joint missing-label training stream spanning the three datasets. [Paper, §§3.4–4](https://doi.org/10.1016/j.robot.2023.104580).

This corrects the earlier abstract-based implication in our index: **multi-dataset component training is established, but this paper does not settle RQ2's joint heterogeneous behavior-representation experiment**. Nor do its claims about safety-standard adherence substitute for demonstrated calibration or selective-labeling reliability.

It supplies no answer for missing RGB or 3D-only targets; its attribute system depends on images, pose and calibrated BEV features. No complete official implementation/weights were verified. Recreating its detector/tracker would expand our task unnecessarily.

A defensible RQ2 contribution must go beyond separate component pretraining: same inference representation, explicit native-label masks, complementary-source semantic supervision versus generic/self-supervised controls, and held-out transfer. [Mordan et al.](detecting-32-attributes.md) additionally establish partial-annotation multitask recognition within JAAD.

[Direction review](../offline-labeling-research-direction.md).
