# TubeDETR: Spatio-Temporal Video Grounding with Transformers

Yang et al., CVPR 2022. Reviewed 2026-10-13: [full paper/supplement](https://arxiv.org/pdf/2203.16434), §3 and component ablations; [official implementation and checkpoint links](https://github.com/antoyang/TubeDETR).

The task localizes a space-time object tube specified by text in an untrimmed RGB video. A sparse video/text encoder retains spatial feature maps, with a fast branch for local detail. Decoder time queries alternate temporal self-attention with time-aligned spatial cross-attention. It predicts boxes and temporal boundaries, evaluated on VidSTG/HC-STVG with grounding metrics. Ablations support its temporal decoder and fast branch. Code, training/evaluation scripts and trained models are publicly linked; their supervision includes visual-language grounding, not pedestrian state labels. [Paper, §§3–4](https://arxiv.org/pdf/2203.16434); [repo](https://github.com/antoyang/TubeDETR).

This is a strong **mechanism precedent** for preserving spatial tokens and revisiting them with temporally contextualized queries. Replacing text with a track representation is an adaptation; temporal queries or factorized space-time attention alone cannot be claimed as inventions.

Our target is already localized, so box/tube detection losses and a text encoder are unnecessary. A modest decoder operating on frozen RGB/3D tokens is more plausible than transferring the whole grounding network. No LiDAR or missing-modality benchmark is supplied by this work.

Use it to define the strongest control for a complete-track-conditioned retrieval hypothesis: per-timestep queries contextualized over the full track, then cross-attention into the same retained spatial tokens. If this ordinary adaptation matches a proposed special mechanism, methodological novelty is weak. Interpretation should not equate an attention weight with a verified causal explanation.

[Direction review](../offline-labeling-research-direction.md).
