# SelectiveNet: A Deep Neural Network with an Integrated Reject Option

Geifman and El-Yaniv, ICML 2019. Reviewed 2026-10-13: [full paper](https://arxiv.org/pdf/1901.09192), selective-risk/coverage objective and experiments.

A shared representation supports prediction, learned selection and an auxiliary full-coverage head. Training minimizes accepted-example loss while penalizing coverage below a requested level. Classification/regression experiments compare risk–coverage with confidence-based rejection and report improvements in their settings. The work distinguishes desired training coverage from actual test coverage. It does not establish reliable coverage under arbitrary domain shift or dense correlated action labels. [Paper, §§3–6](https://arxiv.org/pdf/1901.09192).

Learned rejection and risk–coverage tradeoffs are established. Our possible reliability contribution would be **measured annotation usefulness under naturally incomplete observations**, with audited accepted errors and class/cohort retention. A simpler source-calibrated confidence threshold should come first.

Global accepted accuracy can improve by rejecting most Waiting/Crossing examples. Report retained class support, track/group coverage and natural missingness, alongside overall risk–coverage. Segment/whole-track acceptance requires a separately defined unit; independent frame thresholds do not automatically produce consistent label sequences.

Do not claim an error guarantee from a source-validation threshold when the target distribution changes. Target-labeled threshold tuning is adaptation. Independent audit labels and review effort are additional evidence, not inferred from benchmark F1.

[Direction review](../offline-labeling-research-direction.md).
