# Utonia: Toward One Encoder for All Point Clouds

Zhang et al., ICML 2026. Reviewed 2026-10-13: [v2 paper/appendix](https://arxiv.org/pdf/2603.03283v2), Tables 6/7/12 and implementation details; [official API](https://github.com/Pointcept/Utonia); [actual model card/files](https://huggingface.co/Pointcept/Utonia).

Utonia jointly self-distills an RoPE-modified PTv3 across indoor, outdoor, object and video-derived clouds. Modality blinding and scale handling address domain/channel shifts. The default model takes coordinates/color/normals and **explicitly permits absent color/normals as zeros**. Point features and hierarchical inverse mapping are exposed. Table 6 reports ScanNet linear-probe mIoU 77.0 without color versus Concerto's 36.8. Table 12's pretraining includes **Waymo, nuScenes and SemanticKITTI**, among other sources. `utonia.pth` and an original training checkpoint are published; model weights are CC-BY-NC. [Paper](https://arxiv.org/pdf/2603.03283v2); [model card](https://huggingface.co/Pointcept/Utonia).

This is the most directly relevant reviewed **pretrained point-feature interface for missing-RGB/normal observations**. It is still a static spatial encoder, not a verified full-track behavior labeler. Its geometry features require correct units, grid scale and independent validity; metric track/road relationships should be retained alongside any rescaled point inputs.

It is technically worth a bounded feature-extraction check if external pretrained representations are allowed. It is not a clean strict LOKI→ROAD backbone under our target-excluded protocol. Waymo pretraining may intersect both ROAD training and held-out clips, because E001's split is experiment-specific. Exact recording IDs, pretraining splits, inherited teachers and weight revision need audit; overlap was not measured here.

Do not infer behavior performance from segmentation or synthetic object similarity. A few source-validation feature probes and local-scene visualizations would resolve compatibility more cheaply than fine-tuning 137M encoder parameters. Do not train a replacement foundation model to remove dataset exposure within the December schedule.

## Verified pretraining partitions (2026-10-15)

The actual Pointcept training configs at revision `1342eda30e96cbb5fadf5374ff8eb18f6de15c71` resolve an important previously open question:

- [Stage 1](https://github.com/Pointcept/Pointcept/blob/1342eda30e96cbb5fadf5374ff8eb18f6de15c71/configs/utonia/pretrain-utonia-v1m1-0-base_stagev1.py) includes `WaymoImagePointDataset` with `split=["training", "validation"]` inside the training mixture.
- [Stage 2](https://github.com/Pointcept/Pointcept/blob/1342eda30e96cbb5fadf5374ff8eb18f6de15c71/configs/utonia/pretrain-utonia-v1m1-0-base_stagev2.py) again includes both Waymo partitions; nuScenes and SemanticKITTI entries use `split=["train", "val", "test"]`.

These are observation-access settings for self-supervised pretraining, not evidence of behavior-label training or a recording-level manifest of the released weight. They show that **train-only pretraining cannot be assumed**, and adopting an official validation split alone does not remove exposure under this recipe. All annotated ROAD clips in our acquired release come from Waymo training; see the [PTv3 follow-up](ptv3.md#why-simply-adopting-the-original-waymo-split-is-insufficient-today). Exact checkpoint-to-run/recording correspondence remains unverified. Config snapshots and public weight metadata are retained under ignored `outputs/inspection/literature/20261015-ptv3-checkpoints/`.

This narrows the earlier uncertainty without ruling out a practical, exposure-disclosed Utonia labeler. No weights were executed, existing splits changed or runs launched.

### Independent split recheck (2026-10-16)

**Precise conclusion: published pretraining includes Waymo training and validation; it does not establish use of all Waymo frames or the Waymo test split.** The [paper, Appendix Table 12](https://arxiv.org/html/2603.03283v2), reports 23,691 Waymo training samples, 5,976 validation samples, zero test samples, total 29,667. Appendix A.1 identifies the table as the pretraining mixture. These are the paper's sample counts, not independent recording-level coverage measurements.

The code independently supports that interpretation:

- [Stage 1, lines 502–521](https://github.com/Pointcept/Pointcept/blob/1342eda30e96cbb5fadf5374ff8eb18f6de15c71/configs/utonia/pretrain-utonia-v1m1-0-base_stagev1.py#L502-L521) puts `split=["training", "validation"]` under `data.train.datasets` with `test_mode=False`.
- [Stage 2, lines 909–924](https://github.com/Pointcept/Pointcept/blob/1342eda30e96cbb5fadf5374ff8eb18f6de15c71/configs/utonia/pretrain-utonia-v1m1-0-base_stagev2.py#L909-L924) repeats the same choice. This is not a separate validation-only evaluation entry.
- [WaymoImagePointDataset](https://github.com/Pointcept/Pointcept/blob/1342eda30e96cbb5fadf5374ff8eb18f6de15c71/pointcept/datasets/waymo.py#L239-L258) inherits the [image-point loader](https://github.com/Pointcept/Pointcept/blob/1342eda30e96cbb5fadf5374ff8eb18f6de15c71/pointcept/datasets/defaults.py#L308-L325), which opens each named split JSON and merges their entries into one training dataset.
- The [preprocessing launcher](https://github.com/Pointcept/Pointcept/blob/1342eda30e96cbb5fadf5374ff8eb18f6de15c71/pointcept/datasets/preprocessing/concerto/waymo/preprocess_waymo.sh#L1-L6) defaults to training and validation. Its [frame filter](https://github.com/Pointcept/Pointcept/blob/1342eda30e96cbb5fadf5374ff8eb18f6de15c71/pointcept/datasets/preprocessing/concerto/waymo/preprocess_waymo.py#L481-L488) skips train/validation frames without segmentation labels; the existence of those labels is used for frame selection, which alone does not demonstrate supervised segmentation loss during Utonia pretraining.

Paper and recipe agree on split exposure. Released-weight execution logs, actual split manifests and exact ROAD recording/frame overlap remain unverified; Tiny/Small provenance remains unresolved as below. Consequently, adopting Waymo validation for our test set would **not establish unseen-observation evaluation** with the original Utonia recipe. Fresh pinned source copies/hashes: `outputs/inspection/literature/20261016-utonia-waymo-recheck/`.

## Downloadable checkpoints (2026-10-16)

The official release now includes **three encoder sizes**. Tiny and Small are announced as October 2026 additions in the [inference README](https://github.com/Pointcept/Utonia/blob/de1f9deef4ee65e508d74347f8d0eb848d878abb/README.md). Live Hugging Face metadata was checked at revision `9457f025e36c7001ee962a542f0d27a5b600e1e7`; sizes below are decimal download sizes, not GPU memory requirements. [Pinned model card](https://huggingface.co/Pointcept/Utonia/blob/9457f025e36c7001ee962a542f0d27a5b600e1e7/README.md) · [Files](https://huggingface.co/Pointcept/Utonia/tree/9457f025e36c7001ee962a542f0d27a5b600e1e7).

| File | Download size | Role |
| --- | --- | --- |
| [utonia.pth](https://huggingface.co/Pointcept/Utonia/resolve/9457f025e36c7001ee962a542f0d27a5b600e1e7/utonia.pth) | 549.2 MB | Original encoder, simplified inference checkpoint |
| [utonia_small.pth](https://huggingface.co/Pointcept/Utonia/resolve/9457f025e36c7001ee962a542f0d27a5b600e1e7/utonia_small.pth) | 149.1 MB | Smaller pretrained encoder |
| [utonia_tiny.pth](https://huggingface.co/Pointcept/Utonia/resolve/9457f025e36c7001ee962a542f0d27a5b600e1e7/utonia_tiny.pth) | 37.4 MB | Smallest pretrained encoder |
| [pretrain-utonia-v1m1-0-base_stagev2.pth](https://huggingface.co/Pointcept/Utonia/resolve/9457f025e36c7001ee962a542f0d27a5b600e1e7/pretrain-utonia-v1m1-0-base_stagev2.pth) | 7.01 GB | Original pretraining checkpoint for Pointcept; unnecessary for ordinary frozen feature extraction |
| [utonia_linear_prob_head_sc.pth](https://huggingface.co/Pointcept/Utonia/resolve/9457f025e36c7001ee962a542f0d27a5b600e1e7/utonia_linear_prob_head_sc.pth) | 112.7 kB | ScanNet linear segmentation head, compatible with original encoder only; not a behavior head |

The model card gives stage channels/depths: original `(54,108,216,432,576)` / `(3,3,3,12,3)`; Small `(36,72,144,288,432)` / `(2,2,2,6,2)`; Tiny `(18,36,72,144,216)` / `(2,2,2,6,2)`. Current code loads checkpoint-specific architecture through `utonia.model.load("utonia_small")` or `"utonia_tiny"`. All use the point-feature interface; missing color/normals are zero-filled. They are not separately published RGB-only models. The existing ScanNet head cannot be reused for Small/Tiny. [Official inference instructions](https://github.com/Pointcept/Utonia/blob/de1f9deef4ee65e508d74347f8d0eb848d878abb/README.md#quick-start).

**Provenance remains a gate:** the card recommends domain-specific distillation from the original model for small encoders, but does not identify the released Tiny/Small training recordings, exact recipe or teacher lineage. Their availability does not establish a Waymo-free variant; keep their exposure **unknown**, rather than assume either identical pretraining or clean independence. The original model's published recipe includes the official held-out partitions described above. All released weights carry CC-BY-NC-4.0.

**Project interpretation:** Small is a plausible first bounded compatibility/runtime probe on the RTX 4080; the original model remains a candidate for frozen extraction if measured cost is acceptable. This is a proposal, not a selected backbone or a performance ranking. No weight was downloaded or executed. Saved metadata, model card, inference README, revisions and file SHA-256 identifiers: `outputs/inspection/literature/20261016-utonia-checkpoints/`.

[Sonata comparison](sonata.md) · [Direction review](../offline-labeling-research-direction.md).
