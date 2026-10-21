# On Calibration of Modern Neural Networks

Guo, Pleiss, Sun and Weinberger, ICML 2017. Reviewed 2026-10-13: [paper](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), calibration definitions and temperature-scaling experiments.

The paper evaluates confidence calibration in image/text classifiers. A positive scalar temperature is fitted on validation logits by likelihood; it changes probabilities while preserving class order. Experiments find it effective across several architectures/datasets. This does not establish calibration under our cross-dataset observation shift or temporally correlated track sampling. [Paper, §§3–5](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).

This is an inexpensive **post-hoc reliability baseline**, not a new research method. Use source-validation fitting for target-excluded transfer, then report target calibration without refitting. Target-temperature fitting would be a separate labeled-access regime. Report NLL/Brier and class/condition reliability; low aggregate ECE can hide rare-class failure.

Equal-track scoring and frame correlation affect how calibration data/weights should be defined. Fitting temperatures on previously examined test predictions would violate the calibration evaluation. Existing saved logits make a future authorized diagnostic cheap, but this literature review does not compute or tune it.

[Selective prediction](selectivenet.md) · [Direction review](../offline-labeling-research-direction.md).
