# E002b — Branch normalization and model capacity

**Design agreed 2026-10-25; shared LiDAR execution is implemented and passes CPU/CUDA equivalence and bounded timing checks. Its actual three-run W&B smoke awaits specific approval; independent geometry/RGB training is user-started.** Follow-up to the [eight E002a source-validation runs](../E002a-feature-feasibility/README.md#completed-source-validation-runs--2026-10-25). This record owns the new comparison; E002a retains its original models, checkpoints and results.

## Question and evidence

Do final branch LayerNorm and reduced shared or branch capacity improve source-validation track-weighted macro-F1 and cross-entropy? A smaller training/validation loss gap alone is not success: an underfitting model can have a small gap and worse predictions.

E002a's geometry + D-DITR arm achieved validation F1 0.6257 on ROAD and 0.7587 on LOKI in seed0, above geometry, geometry + RGB and the three-modality arm. RGB arms had stronger loss overfitting, especially on ROAD; geometry + D-DITR also deteriorated late. These observations motivate the comparison but do not establish a scale imbalance, RGB shortcut or sufficient embedding width. The design follows inspection of both sources' validation results and the earlier test-informed geometry choice; it is development, not untouched-test or strict zero-shot evidence.

## Agreed matrix

Train separately on ROAD-Waymo and LOKI. Arms are `g` (geometry), `g-rgb` (geometry + RGB), and `g-dditr` (geometry + LiDAR). Only enabled descriptors and their validity flags enter fusion. The combined RGB + LiDAR arm is outside this increment.

| Variant | Final branch LayerNorm | Geometry / RGB / LiDAR widths | Temporal width / FFN |
|---|---|---|---|
| `norm` — normalized reference | Yes |64 /128 /128|256 /1024|
| `temporal-small` — reduced shared capacity | Yes |64 /128 /128|128 /512|
| `branches-small` — compact branches | Yes |32 /64 /64|256 /1024|

The initial development matrix inherits seed0: three variants × three arms × two sources = **18 separately trained models**. Further matched confirmation seeds and their budget are **TBD** before execution of that extension. Shared preparation changes execution cost, not the number of trained models. Temporal64 and branch widths16/32/32 are not part of this matrix.

`temporal-small` changes fusion output, both temporal blocks, final temporal normalization and classifier input together. It tests a smaller shared representation and temporal capacity, not temporal reasoning alone. `branches-small` changes enabled branches together; it cannot isolate the geometry contribution from the scene branch's compression.

## Exact changes from E002a

All variants use a separate affine LayerNorm on each enabled branch's final descriptor, epsilon1e-5. Normalize across descriptor channels independently at each timestep; apply the branch-validity mask **after** normalization. Missing descriptors and padding remain zero before fusion. Append unchanged binary validity flags outside normalization. Do not normalize the concatenation as a substitute, fit target statistics or change geometry's existing source-training normalization.

| Component | `norm` / `temporal-small` | `branches-small` |
|---|---|---|
| Geometry |20→64→64, GELU after each linear|20→32→32, same activations|
| RGB |Frozen cached CLS768→128|Same CLS768→64|
| LiDAR spatial projection / seed / attention width |68→128 /128 /128|68→64 /64 /64|
| LiDAR spatial FFN |512|256|

LiDAR retains four spatial heads, one learned seed and one cross-attention/FFN block. Reducing its width also reduces spatial readout capacity; this is not solely a final-output bottleneck. The encoder's64-channel point features, feature LayerNorm, relative XYZ/20, inside-box indicator and saved point selections are unchanged. No additional output-only projection is introduced.

Both temporal variants retain two full-track bidirectional pre-norm blocks, four heads, RoPE base10000, the complete5 Hz grid, and a four-class head. Width128 means32 channels/head; width256 means64. Compact spatial width64 means16 channels/head. No attention window, track splitting or truncation is introduced.

Fusion inputs include only enabled descriptors and one flag per enabled branch:

| Variant | `g` | `g-rgb` | `g-dditr` |
|---|---|---|---|
| `norm` |65→256|194→256|194→256|
| `temporal-small` |65→128|194→128|194→128|
| `branches-small` |33→256|98→256|98→256|

## Retained data, fitting and evaluation

Inherit the [E002a data contract](../E002a-feature-feasibility/README.md#retained-data-and-evaluation-contract), exact E001b geometry inputs, four-state targets, masks, frozen populations/splits and source-only normalization. Retain3D-only tracks and partial observations. ROAD training/validation has4,380/1,110 tracks; LOKI8,574/1,861. No new collection, eligibility rule or target projection.

Reuse frozen DINOv3 3× crop CLS and indexed D-DITR features, with their existing checkpoint/exposure provenance and20 m target-specific point selections. Only enabled modalities are loaded. Cache files and schemas remain unchanged; no encoder inference or cache rebuild is required. Learned branch outputs are model-specific and cannot be shared between models.

Inherit [E002a fitting](../E002a-feature-feasibility/README.md#deferred-training-pilot): float32, deterministic execution, effective batch64 complete tracks with the final partial batch retained; equal-track masked CE; AdamW LR3e-4, weight decay1e-4, betas(0.9,0.999), epsilon1e-8; three-epoch update-based warmup; gradient clipping1.0; maximum100 epochs and patience15. Dropout stays0.1 at existing spatial/temporal sites. No descriptor/modality dropout, loss change, class weighting, smoothing, new augmentation or encoder fine-tuning.

Full source validation after each epoch selects the highest four-class track-weighted macro-F1, then lower CE and earlier epoch on exact ties. Patience resets only on strict F1 improvement. Retain class/condition diagnostics, denominators, track CE, epoch/update loss history, checkpoints, saved predictions and per-model local/W&B evidence. Training CE remains an online changing-weight/dropout quantity, not a matched evaluation pass. No held-out or cross-dataset inference in this increment.

New attempt evidence belongs under ignored `outputs/experiments/E002b/`, with references to the existing E001/E001b and E002a caches. W&B remains `yammo-unipd/pedestrian-behaviour-labeling`, one run per trained model; checkpoints and full predictions remain local.

Compare `temporal-small` and `branches-small` separately with `norm` within each source/arm/seed. Existing E002a no-final-LayerNorm seed0 runs are historical controls for `norm`, conditional on the execution-equivalence checks below. If equivalence cannot be established, the need/budget for fresh matched controls is **TBD**; do not silently add runs or attribute a runner change to LayerNorm. Do not replace E002a checkpoints with normalized models at inference.

## Shared prepared batches for LiDAR variants

Share training and validation preparation only among `g-dditr` variants with the same source, seed and shuffle schedule. Geometry/RGB-only arms retain ordinary independent execution. Shared loading does not constitute independent repetitions. The implementation and bounded acceptance are recorded below; they do not establish a full-epoch runtime.

One bounded CPU producer prepares a physical microbatch from the existing indexed cache once. Each active model independently performs forward/backward on that immutable input, releasing its activation graph before the next model. Accumulate each model's track-loss sum divided by the actual logical batch count; clip and update each optimizer once after the logical batch completes. Retain bounded prefetch, complete tracks and a common packing schedule safe for all models; do not retain a whole logical point batch in VRAM.

Each model owns its parameters, optimizer, learning-rate/update counters, training RNG state, early stopping, best/last checkpoints, output directory and W&B run. Model initialization/dropout randomness must not depend on the number or order of other active models. A stopped model leaves the group without changing surviving models' data order or RNG. No shared learned readout or retained cross-model activation graph. Failures preserve available local evidence and propagate; no automatic retry or resume.

### Implementation sequence

1. Prepare the three model definitions and their shape, masking, backward and checkpoint checks, preserving the E002a constructor and state dictionaries. This is a separate increment from the grouped trainer.
2. Add a shared-input accumulation loop using the existing logical64-track loader and bounded prefetch. For each prepared physical microbatch, run each active model's forward/backward sequentially under its own RNG state; then release the inputs. At the logical boundary, clip and step each model's optimizer independently.
3. Add the E002b group lifecycle with explicit per-model logging handles, local evidence, validation/selection and stopping. Keep experiment policy outside shared training operations; reuse existing artifact/metric helpers. An error ends the group and preserves each unfinished attempt's failure evidence.
4. Prove equivalence to independent execution before timing two models, then three. Include the final partial update, dropout, recomputation, model-order/stopping invariance and immutable inputs. Benchmark preparation once versus repeated loading, total group time, memory and GPU activity.
5. Recalibrate on the4080, run bounded both-source artifact/W&B checks, then supply the manual launcher. Full training remains a separate manual action.

The launcher will expose one command per dataset for all nine models: geometry/RGB variants execute independently, while the three LiDAR variants share both training and validation preparation. Validation evaluates each model sequentially per prepared microbatch, retaining independent complete-split metrics and selection. Both choices are agreed; training remains a separate manual action.

## Model preparation — 2026-10-25

[`experiments/e002b.py`](../../src/pedestrian_behavior/experiments/e002b.py) exposes the frozen `model_spec(arm, variant)` and `build_model(arm, variant, profile)` for all nine model configurations. Shared [`multimodal.py`](../../src/pedestrian_behavior/models/multimodal.py) accepts their dimensions and final branch-normalization setting; its default constructor retains E002a behavior. No runner, cache or fitting change is included.

| Variant | `g` parameters | `g-rgb` parameters | `g-dditr` parameters |
|---|---:|---:|---:|
| `norm` |1,603,588|1,735,300|1,843,972|
| `temporal-small` |411,396|526,596|635,268|
| `branches-small` |1,591,556|1,657,540|1,662,788|

All counted parameters are trainable; frozen encoders are absent from this model. Compact geometry changes total capacity little because the temporal blocks dominate. [`Model acceptance`](../../tests/test_e002b_models.py) covers all nine shapes, enabled flags, post-LayerNorm missing/padding masks with nonzero affine bias, backward propagation and exact checkpoint roundtrips. A direct comparison against the pre-change E002a implementation confirms identical seeded parameters/state keys and evaluation/dropout logits for all four original arms. The full locked CPU suite passes206 tests with12 CUDA-only skips;362 local owner links and `git diff --check` pass. These CPU checks alone do not revalidate execution profiles; the later independent geometry/RGB acceptance below supplies its own CUDA/native checks.

## Before implementation and full execution

1. Preserve original E002a configuration/checkpoint loading. Check all new shapes, enabled-only flags, missing/padding masking after LayerNorm and checkpoint/prediction reconstruction; retain masked equal-track loss/gradient semantics.
2. First implement a bounded two-LiDAR-model benchmark against independent execution using identical complete-track inputs, common packing, initial parameters and isolated RNG. Verify losses, gradients and optimizer updates, including dropout; test that model ordering/stopping does not alter the survivor. Check inputs remain unchanged. Extend to the intended three-model group before full execution.
3. Measure preparation/read counts, group wall time versus independent total time, GPU activity, peak VRAM and host RAM. Report group time and amortized per-model time separately. Existing single-model warm timings do not establish shared-run speedup or full-epoch runtime. Accept/reject shared execution on measured benefit and correctness.
4. Recalibrate the changed model/execution code on the4080 and verify bounded both-source integration and per-model local/W&B lifecycle. Feature caches remain read-only. Full-epoch runtime and the final execution profile are **TBD**; no new runnable command or population run is authorized by this documentation alone.

## Independent geometry/RGB execution — 2026-10-25

The user requested manual geometry/RGB training while the shared LiDAR runner is prepared. [`e002b_run.py`](../../src/pedestrian_behavior/experiments/e002b_run.py) reuses E002a source assembly and the existing attempt lifecycle with explicit E002b model, evidence-root and calibration policy. E002a remains the default caller; its extraction-policy file stays unchanged so existing RGB cache identities remain valid. Only `g` and `g-rgb` are accepted here.

[`run-e002b-independent.py`](../../scripts/run-e002b-independent.py) runs six sequential seed0 models for one source, with optional arm selection. This launches no LiDAR model and does not change the18-model matrix. Each model retains full source validation, per-update/epoch loss logging, independent best/last checkpoints and final validation predictions/plots. Use the [manual commands](../../README.md#e002b-geometry-and-geometry--rgb) with fresh output directories;100/15 fitting and all scientific settings above remain fixed.

[`calibrate-e002b.py`](../../scripts/calibrate-e002b.py) takes only packing limits from the earlier completed profile, then measures fresh CUDA CE/backward/AdamW for all six geometry/RGB configurations on both sources' development fixtures plus64×102 temporal shape bounds. Training checks completed status, E002b identity, exact model-spec coverage, execution hashes, backend and GPU. No old code hashes are rewritten and no feature cache is regenerated. The CUDA calibration completed at code revision `dbb02cf` in `outputs/experiments/E002b/calibration-independent/profile.json` (SHA-256 `6868cb53e6d84b99b5defa47d7f0ed61f8ac371d3c6c623370a34045ac4c99d7`). It covers12 real source-fixture workloads plus six64×102 shape bounds; maximum allocated VRAM is477,665,792 bytes, below the9,711,719,546-byte ceiling. These are resource checks with discarded weights, not behavior results. Both source populations fit the tested100/102-slot bounds.

The full locked CPU suite passes209 tests (12 CUDA skips); the managed CUDA11.8 suite passes all209. In the separate extraction/training environment, all E002b checks pass; the full209-test suite reproduces one existing E001 CUDA padding-invariance tolerance failure (maximum difference3.99e-6 versus1e-6). No E001 code or tolerance changed. Four actual W&B smokes pass at the same code revision, recorded in `outputs/experiments/E002b/independent-integration/integration.json`. Each uses one optimizer update and one previously inspected source-validation track; smoke scores are not behavior results. All four reconstruct selected-checkpoint CE/F1 from saved predictions, export20–21 PNG/SVG pairs, preserve every protected input and make zero encoder loads/calls. No held-out output is produced. Per-case runtime is19.0–27.6 s, peak CUDA allocation79–103 MB. This validates the independent commands only, not shared LiDAR execution or population-scale runtime.

| Source / arm / variant | W&B smoke |
|---|---|
| ROAD / `g` / `norm` |[ka4k13p4](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/ka4k13p4)|
| ROAD / `g-rgb` / `temporal-small` |[5plawwp7](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/5plawwp7)|
| LOKI / `g` / `branches-small` |[lk4rtxzd](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/lk4rtxzd)|
| LOKI / `g-rgb` / `norm` |[8g5lgze6](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/8g5lgze6)|

## Shared LiDAR implementation — 2026-10-25

[`shared_accumulation.py`](../../src/pedestrian_behavior/shared_accumulation.py) implements sequential independent updates over one prepared physical batch. [`e002b_group.py`](../../src/pedestrian_behavior/experiments/e002b_group.py) assembles the source once and owns per-model selection, checkpoints, local predictions/plots and explicit W&B handles (`reinit="create_new"`). Final selected checkpoints share another complete source-validation pass. The common packing profile and persistent loader remain unchanged when a model stops. No shared learned features, encoder inference, cache writes, resume or retry.

Initialize each model with seed0 separately and capture its Torch CPU/CUDA state immediately afterward. Restore/capture that state around its forward/backward, including checkpoint recomputation. The current deterministic CPU producer uses no Python/NumPy random operations. Group wall time and peak memory describe the shared process; epoch histories label the training duration `shared_group_seconds`, not independent model compute.

[`Core acceptance`](../../tests/test_shared_accumulation.py) compares independent and grouped logits/loss, pre-step gradients, AdamW state, final weights and RNG. Checks include dropout, a65-track split with a final one-track update, real LiDAR spatial recomputation, reversed model order, stopping one model after the first epoch, and unchanged inputs. CPU comparisons match exactly; preparation reads each track once per group epoch. Forward and final-logging failures propagate, finish all unfinished logging handles and preserve available local evidence. At implementation revision `1eacc41`, the full locked CPU suite passes214 tests with12 CUDA-only skips; the managed CUDA11.8 suite passes all214 without skips. Native lifecycle and timing evidence follow separately.

The [manual shared commands](../../README.md#e002b-shared-lidar-training) run only the three LiDAR variants per dataset. `scripts/run-e002b-all.py` sequences the six independent geometry/RGB models and the LiDAR group for a future complete nine-model run. Existing independent commands, their hashed dependencies and feature formats are preserved so active training can continue in its original checkout.

The first bounded calibration used managed PyTorch2.7.1/CUDA11.8 and passed at `outputs/experiments/E002b/calibration-lidar-group/profile.json` (SHA-256 `1aa217aa0892fbd4801bc04e44057fa73ff5b6d05a0a0790440ade2f7bc2a39b`). Training updates for all three models took20.492→17.780 s on ROAD and11.095→9.658 s on LOKI; validation took11.844→11.221 s and3.304→2.892 s. Both phases reduced track reads192→64, with exact weights/loss/RNG/packing and validation-logit parity. This profile is retained as execution evidence, **not used by the manual training commands**: existing E002b training uses the extractor environment's CUDA12.6 build. Its separate backend-matched profile is required below.

### Backend-matched benchmark and calibration

The authoritative profile is `outputs/experiments/E002b/calibration-lidar-group-extractor/profile.json` (SHA-256 `717a6a058e13de1148eef2c6f17d6ab553662bc1a71d88dbb374afdfde2b0672`), produced at `55680b3` on the4080 using the same PyTorch2.7.1+cu126 /CUDA12.6 extractor environment as existing E002b training. The dedicated checkout is `/home/user20/projects/pedestrian-behavior-labeling-e002b-shared`; the active independent checkout and its runtime hashes are unchanged.

The compact [local calibration audit](../../outputs/experiments/E002b/shared-group-calibration-audit.json) retains profile/system hashes, tests, timings, parity and protected-input checks; full numerical telemetry stays on `aalto` beside each profile.

Each source uses one warm and one measured complete64-track update per path and64 ordered validation tracks. All three variants match independent execution exactly for parameters, CE, training RNG, packing and validation logits. Training is timed independent-first/shared-second; validation shared-first/independent-second. Inputs are warm, no OS/CUDA caches are cleared, and weights are discarded. No other GPU process appeared in the initial snapshot. Model construction, source assembly and artifact exports are outside these timing columns.

| Source / phase | Three independent models | Shared three-model group | Speedup | Track preparations |
|---|---:|---:|---:|---:|
| ROAD training update |18.108 s|15.644 s|1.16×|192→64|
| ROAD validation batch |12.100 s|10.139 s|1.19×|192→64|
| LOKI training update |10.008 s|9.072 s|1.10×|192→64|
| LOKI validation batch |3.492 s|2.771 s|1.26×|192→64|

Amortized shared training time is5.21 s/model on ROAD and3.02 s/model on LOKI; these are group time divided by three, not independent model latencies. The benefit is modest despite eliminating two-thirds of preparation: device transfers, spatial/temporal computation and optimizer work still occur per model, and the original loader already overlaps preparation with computation.

Peak process CUDA allocation is4,870,881,280 bytes (4.54 GiB), reserved10,521,411,584 bytes (9.80 GiB); host RSS peaks at6,182,043,648 bytes (5.76 GiB). The187 whole-GPU telemetry samples average39.3% utilization, maximum100%; they mix warmup, preparation and compute and do not establish per-path utilization. Protected-input checks pass; no encoder inference or feature-cache writes occur. These bounded measurements **do not establish full-epoch or worst-case runtime**.

An actual ROAD smoke with three simultaneous W&B handles is prepared using `independent-integration/selection.json`, at most two selected training updates and one epoch, followed by selected-checkpoint prediction/plot checks. Automatic permission review rejected the launch because earlier approval named four independent smokes rather than these three new shared runs. Specific approval is pending; no new W&B run or full LiDAR training was started by that rejected action.

## Results

Independent geometry/RGB training has been started by the user; completed E002b scientific comparisons have not been audited here. Modality/capacity effects remain unknown. This record freezes the agreed comparison, not a claim that normalization or compression will improve generalization.
