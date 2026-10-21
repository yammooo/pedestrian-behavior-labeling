# ROAD-Waymo: A Large-Scale Action Awareness Dataset for Autonomous Driving

Khan et al.; reviewed 2026-10-13 against [arXiv v3](https://arxiv.org/pdf/2411.01683v3), §§III–V and Tables III–VI, and the [official repository](https://github.com/salmank255/Road-waymo-dataset). Version-specific native facts remain in [the dataset note](../../datasets/road-waymo.md).

## Verified paper evidence

The benchmark jointly localizes agents and recognizes actions, locations and composite events. Published action baselines use RGB video with 3D-RetinaNet and I3D/SlowFast backbones; “3D” here does not mean a raw LiDAR behavior encoder. Evaluation uses frame/tube detection mAP with spatial or tube IoU matching, rather than classification on supplied tracks. The paper compares ROAD, ROAD-Waymo and their combination, and separately performs adaptation with target observations. Table III reports action frame-mAP of 14.0% on test for SlowFast-08. The temporal AV-action experiment concerns the recording vehicle, not dense pedestrian states. [Paper, §§IV–V](https://arxiv.org/pdf/2411.01683v3).

## Consequences for our comparison

This is substantive neighboring action-labeling work, not merely an intention-prediction dataset. Given-track labeling removes detection and association errors from the scientific comparison. Therefore our F1 cannot be compared numerically with detection mAP, and a high classification score cannot establish superiority over ROAD's detectors.

A useful adaptation is a published video backbone followed by pooling at supplied pedestrian boxes and a dense behavior head. Freeze the population, box provenance and temporal support; report it as an adaptation, not a reproduction of 3D-RetinaNet. We do not need to reintroduce pedestrian detection or tube linking.

The v3 text claims official Waymo-aligned splits and server evaluation. Our [official-download verification](../../datasets/road-waymo.md#official-download-verification-2026-10-16) confirms that the acquired annotation file matches the current official bucket checksum, yet ROAD train/validation map to official Waymo training and ROAD test to official Waymo validation. E001 adds its own frozen 70/15/15 split. The paper/release contradiction is verified; the authors' rationale is unknown. The explicit alignment statement was not found in checked v1/v2 texts. The paper also contains separate descriptions of input sequence lengths. Do not silently substitute its protocol for E001 or infer that its reported test equals our local test.

## Novelty boundary

Road-agent action recognition, compositional annotation and cross-country/dataset evaluation are established. The plausible contribution is a controlled offline multimodal classification setting with explicit incomplete observations, semantic projection and equal-track evaluation. It requires explaining why these controls reveal something that the detection benchmark does not, rather than simply removing a harder task component.

Target-input adaptation results are useful context for RQ2 but cannot be relabeled as target-excluded zero-shot transfer. [Shared access rules](../../../experiments/README.md#strict-zero-shot-access).

[Direction review](../offline-labeling-research-direction.md).
