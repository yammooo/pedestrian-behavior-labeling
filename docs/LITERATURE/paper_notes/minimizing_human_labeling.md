# Minimizing Human Labeling in Training Deep Models for Pedestrian Intention Prediction

Citation: Riaz, Wielgosz, López Peña, IEEE T-ITS, 2025. [Published article record](https://ddd.uab.cat/record/313066); [PDF](https://ddd.uab.cat/pub/artpub/2025/313066/Minimizing_Human_Labeling_in_Training_Deep_Models_for_Pedestrian_Intention_Prediction.pdf).

## Problem

Reduce real human labels for crossing-intention prediction using S2R-UDA-CP: synthetic labeled training data, real unlabeled tracks, iterative pseudo-labeling, and a small labeled real validation set.

## Inputs

PedSynth labeled videos, real pedestrian tracks, and limited labeled real validation data.

## Outputs / labels

Crossing/non-crossing pseudo-labels and training data; PedGraph+ assesses generated-label quality separately.

## Online or offline?

Prediction-oriented training/self-labeling. It also must find relevant behavioral segments rather than merely classify pre-cut clips.

## Evaluation / relevance

Shows that automatic crossing pseudo-labeling and temporal smoothing already exist; neither alone is a sufficient novelty claim.

## Dataset(s)

Synthetic PedSynth plus real PIE training data without using its C/NC labels, and a labeled PIE validation set for model selection. The paper reports on PIE test data separately.

## Method

Iterative synthetic-to-real unsupervised domain adaptation: train on synthetic labels, pseudo-label real tracks, retrain on synthetic plus pseudo-labels, and retain iterations that improve validation performance. The paper uses a smoothing/post-processing window.

## Most relevant results

Its contribution is evidence that synthetic training and real-track self-labeling can produce useful binary C/NC labels. It evaluates those generated labels with a separate PedGraph+ model, not just the generating model. These results do not establish multi-state label efficiency for LOKI-style actions.

## What we can reuse

Failure-driven treatment of segments, temporal smoothing as a baseline candidate, and its caution around human-label hindsight.

## What it already solves / does not solve

It addresses synthetic-to-real binary C/NC pseudo-labeling, including temporal smoothing (10 frames in its reported protocol), but not the effect of native fine-grained PedSynth++ pretraining on a four-state real annotation task.

## Open questions raised

How to avoid causal confusion when human labels use future knowledge, and whether its segment assumptions transfer to LOKI/EMT.
