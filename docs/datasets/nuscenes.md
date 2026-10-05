# nuScenes

Status: Official documentation checked; local release and supervision feasibility unverified

Last updated: 2026-10-01

## Candidate role

Possible next source for 3D scene grounding and sensor/domain diversity, with limited motion supervision. It is not a rich behaviour dataset equivalent to LOKI. Inclusion and whether it precedes ROAD remain open.

## Documented observations and annotations

The [official site](https://nuscenes.org/) reports LiDAR, radar, six cameras, 3D boxes, and Boston/Singapore scenes. The [schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md) provides instance-linked 3D annotations, calibrated sensors and ego poses. Native annotated keyframes are at **2 Hz**; this is not dense per-sensor-frame behaviour GT.

The [official detection constants](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/constants.py) include `pedestrian.moving`, `pedestrian.standing`, and `pedestrian.sitting_lying_down`. These native attributes must not be equated automatically with ROAD or LOKI behaviour labels.

The [map expansion tutorial](https://www.nuscenes.org/tutorials/map_expansion_tutorial.html) documents drivable areas, road segments, lanes, pedestrian crossings, walkways and dividers. The schema separately documents lidarseg. Their availability and version/alignment must be verified for the selected package.

## Supervision hypothesis

Local pedestrian-centered scene evidence could support road-relative representations, while attributes could supervise physical motion. Map location is contextual evidence, not a fabricated crossing or waiting label. Barriers/accessibility are candidate questions, not guaranteed complete map targets.

Inspect attribute coverage, time gaps, map/point labels, visibility, coordinates and calibration before specifying losses. Compare adding this source against ROAD-Waymo alone; do not assume that extra sensor/domain diversity improves transfer.

## Remaining checks

Local version/access and package coverage are unverified. Determine usable scene targets, native keyframe-to-sensor timing, map transforms, pedestrian-track continuity, and computation/storage costs. Preserve missing annotations instead of interpolating them into behaviour GT. The [inspection plan](README.md) and [ontology audit](README.md#native-semantics-and-evidence) own the gates.

## Pedestrian population and sequence scale (2026-10-01)

The [official dataset paper](https://arxiv.org/pdf/1903.11027) describes **1,000 scenes of 20 s**, split into **700 train / 150 validation / 150 test scenes**. Public train/validation annotations cover 850 scenes; test annotations are withheld. Scene counts are not pedestrian counts.

An independent primary study, [Kalatian and Farooq, §2.2.1](https://eprints.whiterose.ac.uk/id/eprint/188503/1/2104.08123.pdf), reports extracting **8,143 unique pedestrians** using annotation IDs, before its crossing filters. This is an author-reported extraction count; exact release, split and pedestrian-category inclusion are insufficiently specified to treat it as our verified full-release count.

The [official tutorial](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html) says instances are tracked within scenes, not across scenes. Verify counts directly from `instance.json` joined with `category.json`, separately for each split and pedestrian subclass. Then count attribute availability and track lengths at 2 Hz. Native motion attributes provide limited supervision; these tracks do not all have crossing/waiting behaviour GT.

A separate [PePScenes paper](https://ml4ad.github.io/files/papers2020/PePScenes%3A%20A%20Novel%20Dataset%20and%20Baseline%20for%20Pedestrian%20Action%20Prediction%20in%203D.pdf) reports **719 behaviour-labeled pedestrian tracks** selected from nuScenes for front-of-ego crossing relevance. This is a selected extension, not nuScenes population size or an adopted source.
