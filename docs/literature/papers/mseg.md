# MSeg: A Composite Dataset for Multi-Domain Semantic Segmentation

Lambert et al., CVPR 2020. Reviewed 2026-10-13: [full paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lambert_MSeg_A_Composite_Dataset_for_Multi-Domain_Semantic_Segmentation_CVPR_2020_paper.pdf), taxonomy construction, held-out transfer and controlled ablations; [official implementation](https://github.com/mseg-dataset/mseg-semantic).

MSeg reconciles inconsistent segmentation taxonomies/annotation practices across seven training datasets into 194 classes, including substantial manual relabeling. It evaluates transfer to datasets absent from training and selects using training-domain validation. Controlled comparisons separate naive mixing, taxonomy reconciliation and mask correction; the combined alignment improves robustness. Public annotations/models/code are linked. This is supervised image segmentation, not dense pedestrian states or native-head partial supervision. [Paper, §§3–5](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lambert_MSeg_A_Composite_Dataset_for_Multi-Domain_Semantic_Segmentation_CVPR_2020_paper.pdf).

This supports **semantic auditing**, not a mandate to manually relabel our corpora into a universal ontology. Native-head learning and a restricted shared evaluation projection may be less costly, but neither guarantees that shared features are free of annotation shortcuts.

An RQ2 comparison should separate additional observations, additional labels and semantic compatibility. If naive extra-source training fails, inspect definitions and coverage before adding sophisticated optimization. If semantic labels add no benefit over generic objectives at matched exposure, complementary semantic supervision has not earned its claimed contribution.

MSeg checkpoints also have identifiable supervised source exposure; using a segmentation teacher requires auditing that exposure separately from the behavior network.

[Direction review](../offline-labeling-research-direction.md).
