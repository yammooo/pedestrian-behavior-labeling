# E003a — Structured observation masking

**Draft specification, 2026-11-02.** The user agreed the masking structure and bounded resampling. The probability weights, exact severity limits and model reference below are **proposed defaults, not frozen settings**. No implementation or training has started. This draft replaces the unused E003 scene-retrieval placeholder as the immediate proposed direction; retrieval remains a deferred hypothesis in the [literature review](../../docs/literature/offline-labeling-research-direction.md).

## Question and fixed contract

Does training with varied branch availability and temporal persistence improve missing-input robustness and ROAD-Waymo↔LOKI generalization? Keep the complete given-track timeline, source-only normalization, frozen feature caches, original accepted behavior GT and equal-track CE. Predict every accepted GT slot, including slots where augmentation removes all inputs. This is supervised state prediction under input corruption; no reconstruction decoder, teacher, new labels or auxiliary loss is introduced. [ModDrop/ActionMAE precedents](../../docs/literature/README.md#observation-masking-precedents).

The [training-only feasibility audit](../E002-multimodal-reference/README.md#training-gt-runs-and-temporal-mask-feasibility-2026-11-02) supplies the descriptive evidence. Observed constant-label runs have median 15 ROAD /20 LOKI slots. GT gaps break runs. The common grid is 5 Hz; LOKI physical timing remains nominal. The policy uses no per-track GT to choose, place or accept masks.

## First comparison

**Fixed-N only:** every source track is presented exactly once per epoch in both conditions.

| Condition | Presentations per source epoch | Status |
|---|---:|---|
| No observation masking | N original tracks | Existing matching control, subject to configuration verification |
| E003a masking | N tracks, each presented once with a fresh clean-or-masked view | First proposed increment: one new seed 0 run per source |

Proposed reference: E002d `g-rgb-dditr / temporal-small`, branch LayerNorm, geometry/RGB/LiDAR widths 64/128/128, temporal width 128, two blocks, temporal/spatial dropout 0.1, RGB descriptor dropout 0.5 and LiDAR descriptor dropout 0.0. This has completed controls on both sources. Keep that model, E002 optimizer/warmup, effective batch 64, maximum 100 epochs/patience 15 and seed 0 fixed. Confirm the reference before implementation; changing descriptor dropout requires a matching control. Inputs, targets, populations and splits remain those of E002.

Fresh views are generated online, not stored as duplicate tracks. Every training track retains one presentation per E003a epoch; reshuffle assignments and redraw masks across epochs. Rejected candidates do not add optimizer updates or replace a track with another track.

## Sampler

| Choice | Value | Status |
|---|---|---|
| Available branches | Geometry G, RGB R, LiDAR L | Agreed |
| Candidate persistence lengths | 1, 5, 10, 20 slots, or the full track T | Agreed working set from discussion |
| Probability of original view | 0.5 per track presentation | Proposed |
| Probability of full-track persistence | 0.25, conditional on drawing an augmented view | Proposed |
| Finite persistence distribution | Remaining 0.75 uniformly over distinct candidates in {1, 5, 10, 20} strictly less than T | Proposed |
| Retained branch count k | Draw independently per block from {1, 2, 3}, uniformly | Count choices agreed; uniform probabilities proposed |
| Selected branches | Uniform subset of size k from {G,R,L}, constant within that block | Proposed explicit distribution |
| Candidate-mask draws | At most 3 total for an augmented presentation | Agreed working limit |
| After three failures | Use the original view of the same track | Agreed |

An original view keeps all naturally available inputs. For an augmented view, sample one persistence length for that track, partition its timeline into consecutive blocks from the first slot, and truncate the last block at T. Choose k and its subset independently in each block. At length1 the subset can change each frame; at length T it stays constant for the whole track. Fixed first-slot alignment is the simple proposed implementation; random phase is not included in this draft.

Never clamp an oversized candidate to T or count T twice. If no finite candidate exists (T=1), the full-track case is the only realizable length. Full-track and finite probabilities are conditional on an augmented draw; retries redraw length, counts and subsets, but not the original/augmented coin flip. Even k=3 can produce an unchanged view; accept it and record its actual unchanged status.

Let A[t,m] be natural availability and S[b,m] the selected branches for block b:

`E[t,m] = A[t,m] AND S[block(t),m]`

k is the number of allowed branches, not guaranteed observed branches. A selected RGB-only block may therefore contain empty timesteps. Keep those timesteps and their labels. Do not switch branches automatically when the selected branch is absent. No deliberate k=0 draw is included in the proposed first policy; such outages remain a separate unresolved extension needed to explicitly enforce sparse all-branch snapshots.

## Severity checks and resampling

The numerical limits below make the previous discussion concrete and require agreement before freezing:

1. Let `O[t] = any_m A[t,m]` and `V[t] = any_m E[t,m]`. Require `sum(V) >= ceil(0.25 * sum(O))`: retain some input on at least 25% of originally observable timesteps. No GT enters this check.
2. Define `c(T) = min(10, floor(T/4))` slots. For **each contiguous zero-input interval of V**, count how many slots in that interval originally had O=true. Require that added-hidden count to be at most c(T). Natural empty slots consume no budget but can connect removed observations into one interval; do not check only separate runs of newly hidden slots. T<4 permits no newly fully hidden timestep, although branch removal is still possible.
3. If either check fails, draw a fresh candidate mask, up to 3 candidates total. After three failures use E=A. Do not relax limits or change track identity. The original view always passes by construction; frozen eligible tracks have some natural input.

The cap is at most 10 newly hidden slots (2 s of grid coverage), with a quarter-track cap for shorter tracks. It does **not** guarantee that every actual empty interval is shorter than 2 s: existing natural gaps can extend it. Record both actual empty-interval length and added-hidden count. No observed-input count can guarantee behavioral identifiability.

The checks use full unpadded context, not accepted-GT positions. A view with no input at any GT slot may still have useful unlabeled context and is not automatically rejected. Filtering would change the intended sampling distribution, so actual accepted/fallback frequencies must be reported rather than assuming the nominal probabilities survived.

## Data flow and observation meaning

Sample the accepted mask once from metadata before feature loading and packing; use a separate seeded mask RNG so cache hits, thread scheduling and model dropout do not affect mask draws. Reuse that mask through forward/backward recomputation. Keep track length, timestamps/RoPE positions, padding and GT masks unchanged. Skip masked LiDAR feature reads/readout where possible; this is an efficiency hypothesis, not a measured speedup.

Masked branches have zero descriptors and effective validity 0; retained inputs are not rescaled by inverse keep probability. Whole-G masking also removes its six input-validity groups; partial natural geometry validity is preserved when G is retained. Keep persistent caches untouched. Store natural and effective masks separately in local evidence; the model receives effective validity, not a flag identifying synthetic masking.

This masks **branch representations**, not independent physical sensors. G includes 2D bbox features; target-centered RGB/LiDAR still use the supplied track localization. Cached velocities can contain information derived from adjacent observations. No raw-sensor outage, motion recomputation or RGB/LiDAR appearance augmentation is claimed.

## Evaluation, evidence and acceptance

- Select checkpoints on the full source validation split at natural availability, using the existing four-class track-weighted macro-F1/CE/earlier-epoch rule. Do not select on opposite validation or a newly optimized masked-validation score.
- Compare natural source and opposite validation F1/CE; then replay identical fixed diagnostic masks for control and augmented models. Exact stress cases/mask seeds are TBD before execution. This is target-informed exploratory development following E002 inspection, not an untouched zero-shot comparison. Held-out tests remain closed.
- Report actual clean/changed/fallback shares, draws per accepted view, rejection reasons, accepted lengths/counts/subsets by source, per-branch coverage, retained observable-time coverage and empty-gap distributions. Log masked versus retained GT support and class/transition diagnostics post hoc, without changing GT or mask selection.
- Existing online train CE/F1 remain dropout-active, changing-weight metrics, now also measured on augmented observations. They are not fixed-checkpoint train scores or directly matched to natural validation difficulty. No extra train inference pass is added.
- Readable acceptance checks: natural availability is never invented; kept inputs and source statistics are unchanged; masking does not delete slots/GT; full-track k=1 stays on one branch; invalid finite lengths never create duplicate full-track cases; seeded draws are repeatable; severity checks handle natural gaps and mask-bridged gaps correctly; exactly 3 failed candidates restore the original without extra track presentations; masked inputs remain zero after branch biases/LayerNorm; cached inputs remain unchanged; control and augmented models use identical diagnostic masks.

**Remaining decisions:** approve or revise the proposed probability weights, exact severity limits and model reference; decide whether deliberate k=0 intervals belong in the first comparison; freeze diagnostic evaluation masks and repeat the metadata-only sampler audit with the accepted constraints. Then implementation, a bounded GPU check and the two-run command can be prepared.
