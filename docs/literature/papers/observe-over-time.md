# Let's Observe Them Over Time: An Improved Pedestrian Attribute Recognition Approach

Thakare et al., WACV 2024. Focused boundary check, 2026-10-13: [full paper](https://openaccess.thecvf.com/content/WACV2024/papers/Thakare_Lets_Observe_Them_Over_Time_An_Improved_Pedestrian_Attribute_Recognition_WACV_2024_paper.pdf), §§3.2–3.6 and experiments.

Attribute-specific activation information is related across multiple images of the same person during training; predictions/GT are aggregated attribute-wise. **Inference is explicitly single-image.** Tests cover appearance-oriented pedestrian attribute datasets such as Market1501, PETA and PA-100k. The paper reports improved attribute scores and ablates the number of views. It does not label a changing road-behavior sequence. [Paper, §§3–4](https://openaccess.thecvf.com/content/WACV2024/papers/Thakare_Lets_Observe_Them_Over_Time_An_Improved_Pedestrian_Attribute_Recognition_WACV_2024_paper.pdf).

This is precedent for using other views to reveal obscured attributes, but its assumption of aggregatable person attributes is unsafe for state transitions. A pedestrian can both wait and cross at different times; unioning those labels destroys our dense target.

The title alone should not be counted as evidence of complete-track offline inference. It also does not establish a new need for our model: crop quality and observation masks should be checked before adding attribute-correlation machinery.

[Direction review](../offline-labeling-research-direction.md).
