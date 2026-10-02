# ARCANE-PedSynth: Synthetic Multi-Pedestrian Datasets with Behavioural Crossing Annotations

Citation: Riaz, Wielgosz, López Peña, 2026.

Link: <https://arxiv.org/abs/2605.24950>

Project relevance updated: 2026-10-01. Paper claims below are distinct from [inspected release/generator findings](../../DATASETS/PED_SYNTH_PLUS_PLUS.md).

## Problem

Generate reproducible synthetic multi-pedestrian autonomous-driving data with dense behavior annotations beyond binary crossing/non-crossing labels.

## Inputs

CARLA simulation with configurable pedestrian/traffic scenarios. The paper reports synchronized RGB, LiDAR, DVS, and estimated 2D pose keypoints for the resulting data.

## Outputs / labels

The paper describes per-frame crossing labels and a 12-state behavioral FSM. [Table 4](https://arxiv.org/pdf/2605.24950) lists `RETREAT`; the inspected public generator enum instead contains `NORMAL_CROSSING`. The active-path and local-release limitations are recorded in the dataset note; declared states do not establish exported rich supervision.

## Online or offline?

The framework supports data generation for pedestrian crossing prediction. Its advertised dense state labels motivated the former offline-supervision hypothesis, now demoted because the required rich implementation is unavailable in the inspected path.

## Dataset(s)

ARCANE-PedSynth framework; PedSynth++ example dataset (533 multi-pedestrian clips across 12 weather conditions, as reported).

## Method

The paper reports CARLA-based generation with a hybrid AI/manual pedestrian controller and a 12-state behavior FSM; pose is estimated from rendered RGB rather than injected as perfect simulator skeleton ground truth. This describes the paper, not a reproduction of all declared states.

## Evaluation

The paper demonstrates the generated dataset/framework. It does not establish PedSynth++→LOKI label compatibility or real-world multi-state zero-shot transfer.

## Most relevant results

The paper motivates rich synthetic supervision and estimated pose. It does not establish those rich labels or sensor synchronization in the inspected release/generator.

## What we can reuse

The distinction between sensor observations and simulator-private state, plus optional future controlled synthetic generation if real-source experiments justify it.

## What this paper already solves

It presents a synthetic generation framework and reports behaviour-rich labels. Local reproduction of the reported rich behaviour implementation has not been established.

## What it does not solve

Canonical ontology mapping, target observability, zero/few-shot transfer to LOKI, or automatic annotation of real unlabeled datasets.

## Relationship to our possible contribution

This is historical motivation for the synthetic route. The current study begins with real-source transfer; completed CARLA geometry exports remain useful side work rather than a commitment to synthetic training.

## Open questions raised

Whether a future, versioned release could support an independently validated auxiliary role. The current plan must not depend on promised unavailable code. The paper's author-request distribution and the Zenodo demo are distinct from the inspected local copy; see the dataset note for provenance gaps.
