# Towards Good Practices for Missing Modality Robust Action Recognition

Woo et al., AAAI 2023; ActionMAE. Reviewed 2026-10-13: [full paper](https://arxiv.org/pdf/2211.13916), §2–3 and availability/fusion comparisons; [official implementation](https://github.com/sangminwoo/ActionMAE).

Per-frame modality encoders feed temporal Transformers; a second module drops and reconstructs modality features. Supervised clip-action classification is combined with feature reconstruction. RGB/depth/infrared tests use NTU and related action datasets. Transformer fusion is more robust than summation/concatenation in the reported setup, but still degrades under some missing modalities; reconstruction improves these comparisons. Code is available, while a generally reusable behavior checkpoint was not verified. [Paper, §§2–4](https://arxiv.org/pdf/2211.13916); [repo](https://github.com/sangminwoo/ActionMAE).

Missing-modality attention, dropout and reconstruction are **established baselines**, not automatic novelty. Natural missingness in LOKI is different from random sensor removal: it can correlate with camera geometry and annotation population. Compare both, and retain an availability-only control where appropriate.

First use explicit masks and a functioning kinematic/3D path, then ordinary modality dropout. Add feature reconstruction only if a diagnosed failure remains. Whole-modality clip removal does not directly address intermittent missing target observations; per-slot masks and physical timing remain necessary.

A model trained on available modality combinations cannot guarantee reliable labels when its informative evidence is absent. Report coverage/class effects rather than treating reconstructed features as measured evidence.

[Direction review](../offline-labeling-research-direction.md).
