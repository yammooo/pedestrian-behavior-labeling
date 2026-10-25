# E002b — Branch normalization and model capacity

**Design agreed 2026-10-25; model definitions and independent geometry/RGB launcher implemented; GPU readiness, grouped execution and training pending.** Follow-up to the [eight E002a source-validation runs](../E002a-feature-feasibility/README.md#completed-source-validation-runs--2026-10-25). This record owns the new comparison; E002a retains its original models, checkpoints and results.

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

**Agreed execution direction, not yet benchmarked:** share training and validation preparation only among `g-dditr` variants with the same source, seed and shuffle schedule. Geometry/RGB-only arms retain ordinary independent execution. Shared loading does not constitute independent repetitions.

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

All counted parameters are trainable; frozen encoders are absent from this model. Compact geometry changes total capacity little because the temporal blocks dominate. [`Model acceptance`](../../tests/test_e002b_models.py) covers all nine shapes, enabled flags, post-LayerNorm missing/padding masks with nonzero affine bias, backward propagation and exact checkpoint roundtrips. A direct comparison against the pre-change E002a implementation confirms identical seeded parameters/state keys and evaluation/dropout logits for all four original arms. The full locked CPU suite passes206 tests with12 CUDA-only skips;362 local owner links and `git diff --check` pass. GPU memory calibration and native integration remain pending; old execution profiles are not revalidated by these CPU checks.

## Before implementation and full execution

1. Preserve original E002a configuration/checkpoint loading. Check all new shapes, enabled-only flags, missing/padding masking after LayerNorm and checkpoint/prediction reconstruction; retain masked equal-track loss/gradient semantics.
2. First implement a bounded two-LiDAR-model benchmark against independent execution using identical complete-track inputs, common packing, initial parameters and isolated RNG. Verify losses, gradients and optimizer updates, including dropout; test that model ordering/stopping does not alter the survivor. Check inputs remain unchanged. Extend to the intended three-model group before full execution.
3. Measure preparation/read counts, group wall time versus independent total time, GPU activity, peak VRAM and host RAM. Report group time and amortized per-model time separately. Existing single-model warm timings do not establish shared-run speedup or full-epoch runtime. Accept/reject shared execution on measured benefit and correctness.
4. Recalibrate the changed model/execution code on the4080 and verify bounded both-source integration and per-model local/W&B lifecycle. Feature caches remain read-only. Full-epoch runtime and the final execution profile are **TBD**; no new runnable command or population run is authorized by this documentation alone.

## Independent geometry/RGB execution — 2026-10-25

The user requested manual geometry/RGB training while the shared LiDAR runner is prepared. [`e002b_run.py`](../../src/pedestrian_behavior/experiments/e002b_run.py) reuses E002a source assembly and the existing attempt lifecycle with explicit E002b model, evidence-root and calibration policy. E002a remains the default caller; its extraction-policy file stays unchanged so existing RGB cache identities remain valid. Only `g` and `g-rgb` are accepted here.

[`run-e002b-independent.py`](../../scripts/run-e002b-independent.py) runs six sequential seed0 models for one source, with optional arm selection. This launches no LiDAR model and does not change the18-model matrix. Each model retains full source validation, per-update/epoch loss logging, independent best/last checkpoints and final validation predictions/plots. Use the [manual commands](../../README.md#e002b-geometry-and-geometry--rgb) with fresh output directories;100/15 fitting and all scientific settings above remain fixed.

[`calibrate-e002b.py`](../../scripts/calibrate-e002b.py) takes only packing limits from the earlier completed profile, then measures fresh CUDA CE/backward/AdamW for all six geometry/RGB configurations on both sources' development fixtures plus64×102 temporal shape bounds. Training checks completed status, E002b identity, exact model-spec coverage, execution hashes, backend and GPU. No old code hashes are rewritten and no feature cache is regenerated. GPU calibration/integration evidence is pending.

## Results

Not run. Shared-preparation performance and modality/capacity effects are unknown. This record freezes the agreed comparison, not a claim that normalization or compression will improve generalization.
