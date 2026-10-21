# Pedestrian Graph+: A Fast Pedestrian Crossing Prediction Model Based on Graph Convolutional Networks

Gantier Cadena, Qian, Wang and Yang, T-ITS 2022, DOI [10.1109/TITS.2022.3173537](https://doi.org/10.1109/TITS.2022.3173537). Checked 2026-10-13 against the [official implementation](https://github.com/RodrigoGantier/Pedestrian_graph_plus). **Full-paper methods/results were not verified in this pass**; this is an implementation/baseline boundary note.

The repository provides JAAD/PIE training/testing, pose/context/ego inputs, graph-based crossing forecasting and checkpoint paths in its `weigths` directory. The claimed target is whether the pedestrian will cross; it is not our four-state dense output. The original Pedestrian Graph reference is ITSC 2019, not ICCV 2023. [Author repo](https://github.com/RodrigoGantier/Pedestrian_graph_plus).

It is an adaptable lightweight pose-fusion reference and an independent evaluator in the existing [self-labeling review](minimizing-human-labeling.md). Choosing it would require checking the full observation/target protocol, preprocessing, estimator exposure and checkpoint compatibility first. No numerical ranking or inference test is claimed here.

Because [Xiong et al.](simple-action-recognition.md) already supplies an inexpensive verified current-action recognition pattern, obtaining another prediction model's full text does not presently block our architecture decision. Revisit this paper if a pose-based final comparison is chosen.

[Direction review](../offline-labeling-research-direction.md).
