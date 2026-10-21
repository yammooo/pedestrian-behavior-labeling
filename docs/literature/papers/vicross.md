# ViCross: Unified Vision-Centric Pedestrian Crossing Action Prediction

Tian, Yang and Wang, September 2026 preprint. Reviewed 2026-10-13: [full v1 paper](https://arxiv.org/html/2609.07420v1), §§3–5 and Table 6; [official implementation](https://github.com/2tianyao1/ViCross). This is a recent preprint, not an independently reproduced result.

A first-frame target box initializes fine/coarse visual tokens; a Qwen2-VL model predicts future crossing. Auxiliary past localization, future-position and decision objectives regularize shared features. LoRA and small extra modules are trained. Table 6 reports JAAD_all→PIE F1 **0.15**, versus 0.71 within JAAD_all; JAAD_beh is a related subset, not another independent dataset. Code and a task-checkpoint download are linked. Reported hardware uses modified 4090s with about 48 GiB, not our laptop's memory budget. [Paper, §§3–5](https://arxiv.org/html/2609.07420v1); [repo](https://github.com/2tianyao1/ViCross).

Target-centric token allocation and future-location auxiliary supervision already overlap with possible method claims. Strong within-domain accuracy does not demonstrate reliable transfer. Its forecast output/first-box-only contract differ from our dense supplied-track labeling and 3D-only population.

An MLLM is not justified as the next baseline. A small frozen-feature model can test scene relevance more directly and preserve numerical outputs. This paper is useful for positioning and retained negative transfer; it does not establish that language reasoning or generation is required.

[Direction review](../offline-labeling-research-direction.md).
