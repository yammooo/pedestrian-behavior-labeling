# E002a — First multimodal behavior pilot

**Agreed pilot specification, updated 2026-10-22.** First behavior pilot within [E002](../E002-multimodal-reference/README.md). Scope, branch/tensor interfaces, dropout, fitting, batching policy and450 GB point-cache budget are settled below. The [model, trainer and manual commands](#model-and-training-acceptance--2026-10-21) are implemented, with bounded GPU operational checks; full RGB preparation and all-entry readback are complete. Modality benefit and final research architecture remain hypotheses. At the2026-10-22 snapshot, build6050233's first full ROAD geometry attempt was active; the [population-scale throughput check](#full-population-throughput-check--2026-10-22) reveals an input-pipeline problem. No full attempt had completed at that snapshot. Completed extraction/resource/semantic probes are [preparatory evidence](#preparatory-work), not separate behavior experiments. Later E002 comparisons may use different architectures.

The existing directory name is retained for stable commands, artifact references and links. This record owns the pilot specification, preparation, fitting settings and eventual behavior results.

## Question and interpretation

Do appearance and local 3D scene features improve offline pedestrian behavior labeling beyond geometry, including missing RGB and cross-dataset transfer? Use an established multimodal reference before considering a new retrieval mechanism. No modality benefit or methodological novelty is assumed; [literature positioning](../../docs/literature/offline-labeling-research-direction.md).

E001b geometry is a pragmatic reference, not a universal winner. Its BiLSTM track-weighted macro-F1, mean ± sample SD in percentage points: ROAD→ROAD **52.18 ±1.55**, ROAD→LOKI **45.57 ±1.10**, LOKI→LOKI **62.03 ±1.61**, LOKI→ROAD **47.53 ±0.91**. Other inputs win or tie elsewhere; [five-seed review](../E001b-2d-bbox-contribution/results-review.md). This representation choice followed inspection of both datasets' tests: disclose **test-informed development**.

## Next increment: behavior utility, not another segmentation gate

The [frozen-feature semantic probe](#frozen-feature-linear-semantic-probe--2026-10-20) reached pedestrian/sidewalk IoU54.1/35.2% on two Waymo development validation clips. This shows accessible semantics, not adequate segmentation or behavior usefulness. Our labeler receives pedestrian tracks/boxes; its point branch supplies additional target/scene evidence. A temporal model cannot restore information discarded by a frozen encoder, while spatial readout/fusion can also be bottlenecks. A negative behavior result cannot identify the encoder alone as the cause.

Prepare one source-training/validation pilot on each source:

| Arm | Fusion receives | Fusion projection |
|---|---|---|
| `g` | Geometry64 + `geometry_valid` |65→256|
| `g-rgb` | Geometry64 + RGB128 + their two validity flags |194→256|
| `g-rgb-dditr` | Geometry64 + RGB128 + D-DITR128 + their three validity flags |323→256|

**Enabled-branch protocol, agreed2026-10-22 (`enabled-branches-v2`):** instantiate and supply only enabled branches. No dummy descriptors or disabled-branch availability enter fusion. Keep tracks, targets, enabled-branch preprocessing, temporal width and fitting protocol matched; branch/fusion parameter counts vary. This tests adding each modality **including its availability**, rather than content conditional on shared availability. **Utonia is deferred**, replacing the earlier four-arm proposal: first establish RGB/D-DITR behavior utility and a practical memory/storage strategy. Its feature specification and exposure/resource evidence below are retained for a separately agreed later comparison, not a fourth current arm. Frozen encoders are the initial pilot policy, not a final freezing decision. The Waymo-adapted semantic head is diagnostic only; neither its logits nor semantic GT enter the behavior inputs.

This explicitly supersedes the shared-three-flag/zero-descriptor protocol used by build6050233 and the21 October acceptance. Preserve those outputs as historical `shared-availability-v1`; do not combine protocols in a paired comparison. The active checkout/process is unchanged by this revision. Frozen feature identities/storage are unchanged, but model/execution hashes change: profile05 is historical and a new resource calibration is required before new attempts, alongside the unmet [throughput gate](#throughput-gate--agreed-2026-10-22).

The earlier LR/query sweep is not the next increment. No further standalone segmentation optimization is planned before the behavior pilot.

## Retained data and evaluation contract

- Reuse E001/E001b's frozen eligible complete tracks, 5 Hz grids, four-state targets, native associations and group splits. Preserve 3D-only LOKI tracks, internal gaps and unlabeled context.
- Generate four logits at every non-padding timestep. Only accepted GT contributes to equal-track CE and metrics; retain E001 selection/aggregation semantics.
- Geometry normalization uses source-training statistics only. Frozen image transforms are fixed; no pooled cross-dataset feature fitting or target PCA/compression.
- Source training/validation only in the pilot, using all frozen eligible tracks in those splits: **LOKI8,574 training /1,861 validation; ROAD-Waymo4,380 /1,110**. Keep the same complete tracks/GT across arms; no additional group/track subset. Train separate models per source; each source's validation selects its own checkpoints. No held-out inference.
- Experimental ROAD partitions are not official Waymo partitions: all acquired annotated ROAD clips used by E001 originate in official Waymo training. Keep existing splits; [native verification](../../docs/datasets/road-waymo.md#official-download-verification-2026-10-16).

## Tensor conventions

`B`: tracks/batch; `T`: padded track length; `N`: retained local point sites/frame; `N_max`: spatial padding length; `d_e`: encoder feature width; `D`: shared spatial attention width; `M`: learned spatial seeds. Logical shapes do not require keeping all `B × T` point sets resident together; spatial processing must be chunked without changing complete-track semantics or gradients.

## Geometry branch: exact E001b input

**Input `[B,T,20]`: 14 continuous values + six existing validity flags.** Retain [E001b's definitions](../E001b-2d-bbox-contribution/README.md#additional-observation-contract):

| Group | Continuous values | Validity |
|---|---|---|
| K | Pedestrian XY velocity | One vector flag |
| T | XY displacement from first valid position | One vector flag |
| R position | Pedestrian-minus-ego XY position | One vector flag |
| R velocity | `v_ped − v_ego`, XY | One vector flag |
| 2D box | Image-normalized horizontal center, **bottom** vertical coordinate, width, height | Bbox flag |
| 2D velocity | Horizontal-center and **bottom** vertical velocity | Bbox-velocity flag |

Use source timestamps/immediate-neighbor derivatives, no interpolation; zero missing numeric values after source normalization. ROAD uses native Waymo FRONT boxes without ROAD fallback; LOKI uses native boxes. Do not substitute E001c centers/dimensions.

Agreed MLP: `Linear(20,64) → GELU → Linear(64,64) → GELU`, output `[B,T,64]`. Internal flags preserve partial missingness. Derive `geometry_valid = any(six flags)` and zero the final descriptor if all groups are missing. Append this derived flag at fusion in every arm; retain the six internal flags for partial missingness.

## RGB branch

Frozen/eval **DINOv3 ViT-B/16 LVD-1689M**; [verified checkpoint/interface](#rgb-checkpoint-completion--2026-10-19), [official implementation](https://github.com/facebookresearch/dinov3).

```text
Native RGB + native target bbox
→ centered square: side3 × max(box width, box height)
→ bicubic256 ×256, fixed pretrained RGB normalization
→ frozen DINOv3 normalized CLS [768]
→ trainable Linear(768,128)
→ RGB descriptor [128]
```

**Agreed 2026-10-21:** cache the frozen RGB CLS at **3×** crop scale for the full pilot training/validation populations, separately from earlier1.5× probe outputs. Store float32 `[T,768]`, explicit validity, track/native-time identities, crop/transform settings and checkpoint provenance; zero missing rows without filtering tracks. Never reuse a1.5× CLS cache as3×. RGB is a complete persistent cache without LRU eviction, separate from the450 GB point budget; dense descriptors total2.313 GB before masks/identities ([disk audit](#bounded-cache-simulation--2026-10-21)). Do not cache the full population's RGB images/crop pixels.

Keep the intended crop center; fill out-of-image pixels with pretrained RGB mean. No native-box/image intersection means missing RGB; retain partial boxes and record intersection/padding fractions. Numeric geometry remains unclipped. No full-image branch, patch retrieval, RGB augmentation or encoder fine-tuning initially. CLS utility is untested; DINO web-pretraining overlap is unknown.

## Point-cloud branch

### Encoder features

Encode the **complete native sweep** once per scene/timestamp before pedestrian selection. For the current pilot retain pinned D-DITR preprocessing, voxel inverse mappings and feature-coordinate alignment; [D-DITR contracts](#encoder-contracts), [Utonia contracts](#frozen-point-feature-pca--2026-10-20).

| Encoder / scope | Frozen representation | `d_e` | Token input with geometry |
|---|---|---:|---:|
| NuScenes D-DITR, distilled + segmentation-fine-tuned | Decoded point features **before** its16-class head |64|68|
| Original Utonia — deferred | All five encoder levels, inverse-mapped and concatenated: `54+108+216+432+576` |1386|1390|

D-DITR uses XYZ+strength. Utonia uses its configured XYZ with absent color/normals zero; no LiDAR RGB coloring. Both effective input grids are5 cm under their pinned recipes. The original Utonia [ScanNet linear-probe demo](https://github.com/Pointcept/Utonia/blob/de1f9deef4ee65e508d74347f8d0eb848d878abb/demo/2_sem_seg.py) concatenates all levels; we reuse that feature assembly, not its class-specific weights. The earlier1224-channel three-level PCA representation remains visual evidence, not the behavior input.

Utonia's raw feature width is~21.7 times D-DITR's. This affects storage and input-projection cost, not downstream attention width after projection. Report the different feature stages/supervision; do not interpret this as a controlled architectural difference.

### Pedestrian crop and point tokens

Use a **20 m horizontal-radius cylinder**, no height cutoff, centered on the observed native target position. Require a usable sweep, target center and coordinate transform; no center interpolation across missing observations. The crop restricts readout locations, not necessarily the frozen encoder's receptive context. Earlier preparatory15 m measurements remain unchanged and require resource re-estimation for20 m.

**Agreed 2026-10-21:** retain every D-DITR encoder-input voxel representative inside the20 m crop, using its aligned decoded feature and native/inverse references. No mapping back to duplicate features for every native return, additional subsampling, token cap, target-support filtering or GT/model-error-based selection. Preserve empty/missing crops through the existing missing-input policy. Shared-site alignment with Utonia is deferred with that arm.

Agreed per-point input and projection:

```text
relative_xyz_i: target-relative coordinates in fixed initial ego-heading axes
inside_box_i:   geometric membership in the supplied native rotated cuboid
x_i = concat(LayerNorm(feature_i), relative_xyz_i /20, inside_box_i)
H_i = Linear(d_e +4, D)(x_i)
```

Relative XYZ has3 channels; membership has1. Box dimensions/heading define membership but are not extra geometry-MLP inputs. Containment is not verified point ownership. No point-index positional embedding; explicit target-relative coordinates supply spatial information. LayerNorm on encoder features is agreed for the first implementation.

**Agreed 2026-10-21: `D = 128`**; a future Utonia comparison would retain this width. D-DITR projects 68→128 (8,832 parameters with bias); Utonia projects 1390→128 (178,048). Expansion before nonlinear attention provides processing capacity, not new observations. Equal RGB/LiDAR descriptor widths are a practical starting budget, not evidence that they retain equal information.

### Spatial cross-attention and output

One cross-attention block, four heads, **one unconditioned learned seed (`M = 1`)**, agreed 2026-10-21. Attention width is128, with32 channels/head. Spatial Q/K/V and attention output projections are bias-free, matching the W-only equations below; other branch/fusion linears retain their biases. The seed is a global trainable parameter repeated for every valid frame, not conditioned on geometry/RGB embeddings.

```text
Point tokens H: [N,D]       Seeds S: [M,D]
Q = LN(S) W_Q              K = LN(H) W_K       V = LN(H) W_V
Split into4 heads: Q [4,M,D/4], K,V [4,N,D/4]
A = Drop(softmax(Q K^T / sqrt(D/4) + padding_mask))  # [4,M,N]
U = S + Drop(ConcatHeads(A V) W_O)
FFN(x) = Linear2(Drop(GELU(Linear1(x))))  # 128→512→128
Z = U + Drop(FFN(LN(U)))
LiDAR descriptor = Z[0]     # [128]; no extra output projection
```

This is cross-attention, not prepended-token unrestricted self-attention: no `N × N` attention in this readout. Every `Drop` uses0.1 in training and is disabled in evaluation; [exact shared policy](#dropout-normalization-and-initialization). Geometry enters through point coordinates/membership; no learned semantic class predictions are required. The single seed output forms one128-channel descriptor/frame; it is not an extra temporal timestep. The spatial summary is made before temporal reasoning; future frames do not change its point selection. [PMA/Set Transformer precedent](https://proceedings.mlr.press/v97/lee19d.html).

Multiple seeds remain a possible later control, not a first-pilot setting. With the same128-channel width, four seeds would produce `[4,128]` instead of `[1,128]`: more summary slots, not the same total descriptor size. Different seeds could preserve different spatial evidence, but specialization is not guaranteed and seeds may become redundant. Cross-attention score/value aggregation scales with seed count; point encoding/projection is shared. Flattening would require a larger fusion input; compressing back to128 reinstates an output bottleneck, though aggregation is more expressive. Passing spatial tokens separately through time would change the temporal architecture. No extra seeds can recover information absent from frozen encoder features or discarded point sampling.

### Spatial readout objection and revised proposal (2026-10-16)

The initial pooling discussion motivated retaining coordinates until learned aggregation. Shared unconditioned seeds were chosen on19 October; fixed mean/max remains a possible control, not the default. DINO CLS is also learned compression. Full-track-conditioned retrieval remains a conditional later hypothesis; none of these summaries is assumed lossless or novel.

## Missing inputs and padding

| Branch | Validity rule | Invalid output |
|---|---|---|
| Geometry | Any existing feature-group flag true; partial groups retain their masks | Zero final64-channel descriptor if all groups missing |
| RGB | `rgb_valid`: usable native image/box crop | Zero final128-channel descriptor |
| LiDAR | `lidar_valid`: usable20 m crop with at least one retained spatial token | Skip attention; zero final128-channel descriptor |

Append enabled flags in branch order: `g` gets `[geometry_valid]`; `g-rgb` gets `[geometry_valid,rgb_valid]`; `g-rgb-dditr` gets `[geometry_valid,rgb_valid,lidar_valid]`. The geometry flag is the deterministic OR of the six existing geometry flags, not an additional observation. An enabled flag has the same meaning across arms. No extra target/context availability bits enter the model; retain point counts as diagnostics in the LiDAR arm. A crop with context but no inside-box points can still be valid LiDAR.

`bbox_valid` describes numerical box geometry; `rgb_valid` additionally requires a usable image/crop and nonzero native box/image intersection. They often coincide but are not interchangeable. Geometry retains its own bbox masks even when RGB is disabled. No RGB or point cache is opened by `g`; `g-rgb` opens only RGB and never queries point-cost metadata or extracts points. A full geometry-only attempt requires no RGB preparation or encoder probe. RGB-enabled attempts retain the pinned shared probe verification and full RGB-preparation gate.

Gate **after** all branch projections: learned seeds, residuals and biases otherwise produce nonzero outputs for missing input. Never invoke attention with every spatial key masked. Genuine missing observations are retained; corrupt/nonfinite inputs and missing required decoding metadata fail visibly.

Spatial padding masks cover nonexistent point tokens; temporal padding masks cover nonexistent track timesteps. Missing modalities and unlabeled slots do **not** remove time positions. GT masks affect loss/metrics only.

## Fusion and temporal head

```text
enabled descriptors + their validity flags, in geometry/RGB/LiDAR order
→ concat [65 / 194 / 323] → trainable Linear(arm_input_width,256)
→ fused tokens [B,T,256]; no additive positional encoding
→ two full-track bidirectional pre-norm Transformer layers, with temporal RoPE on Q/K
→ LayerNorm(256) → Linear(256,4) → logits [B,T,4]
```

Agreed widths: geometry64, RGB128, LiDAR128, enabled fusion inputs65/194/323 and temporal256. Temporal settings: two layers, four heads, FFN256→1024→256 with GELU, dropout0.1 and final LayerNorm; full noncausal attention within each track, never across tracks. Temporal Q/K/V and output projections retain biases. Apply the same residual/FFN/dropout structure as the spatial block, with self-attention over frame tokens and RoPE on Q/K. Mask padding and retain missing/unlabeled context. Only missing **enabled** descriptors are zeroed. `g` is geometry with a new Transformer; it is not the historical E001b BiLSTM score.

**Temporal positional encoding, agreed 2026-10-21:** use [RoPE](https://arxiv.org/abs/2104.09864) on Q/K in each temporal self-attention layer, not on V; no added sinusoidal/learned position vectors. Keep full bidirectional attention. Rotate adjacent channel pairs over all64 channels/head with frequency base10,000 and position `(nominal_grid_time − track_start_time) /0.2`, equivalent to slot index at5 Hz. Preserve positions through missing modalities/GT; mask temporal padding. No track-length normalization or absolute recording timestamps. This replaces the earlier additive sinusoidal proposal; no behavior advantage is yet measured.

Initially frozen: enabled DINOv3/point encoders, in evaluation mode. Trainable: geometry MLP, enabled RGB projection and/or point normalization/projection/spatial seed/attention/FFN, fusion, temporal layers and classifier. Trainable operations use deterministic FP32; frozen extraction precision is separately recorded. Only behavior supervision trains these modules; source semantic-head probe weights are not reused.

### Dropout, normalization and initialization

Use independent dropout0.1 applications at these four locations in the spatial block and each temporal layer:

| Location | Application |
|---|---|
| Attention probabilities | After masked softmax, before multiplying by V |
| Attention output | After output projection, before residual addition |
| FFN hidden activations | After GELU, before the second linear |
| FFN output | After the second linear, before residual addition |

This matches the dropout locations in the [pinned PyTorch Transformer implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/modules/transformer.py). No additional dropout in the geometry MLP, RGB/point projections, fusion or classifier; no modality dropout, token removal or stochastic depth. Validation/inference disables all dropout; frozen encoders remain in evaluation mode during training too. Spatial recomputation must reuse the original forward's RNG state/masks.

All trainable LayerNorms use affine parameters and epsilon1e-5. Normalize point features before adding geometry; normalize projected point tokens once for shared spatial K/V, and the seed separately for Q. Use PyTorch default linear/LayerNorm initialization and spatial seed `Normal(0,0.02)`. Reset seed0 before each attempt and construct only enabled branches. Different branch/fusion shapes advance RNG differently; common temporal weights are not asserted identical across arms. Disabled branches have no parameters or gradients.

## Access and comparison safeguards

The deferred Utonia alternative is a **Waymo-exposed reference, not an upper bound**. Its [pretraining includes Waymo training/validation](../../docs/literature/papers/utonia.md#verified-pretraining-partitions-2026-10-15), potentially overlapping E001 source-test recordings; exact intersections are unmeasured. Source-only fitting does not establish dataset-excluded LOKI→Waymo transfer or unseen Waymo recordings. A shared encoder choice informed by both sources' development scores must be disclosed. LOKI behavior utility remains untested for both encoders.

D-DITR is LiDAR-only at inference but inherits DINOv2 image distillation and nuScenes semantic fine-tuning; [lineage](../../docs/literature/papers/ditr.md). Different DINO teachers do not imply aligned RGB/LiDAR features or complementarity. Waymo semantic-probe decisions are additional label-informed development. Freezing does not erase pretraining access; retain [shared evaluation safeguards](../README.md#strict-zero-shot-access).

Later modality ablations remain deferred: geometry, +RGB, +LiDAR without RGB, +both, each with its enabled validity. With five seeds/two sources/four variants they would require40 attempts/80 test cells, separately authorized after pilot settings freeze. No held-out tests in the current pilot.

## Deferred training pilot

One seed0 attempt/arm/source gives **six planned exploratory attempts** across three arms/two sources, run **one at a time** with the [manual sequential launcher](../../README.md#e002a-source-validation-pilot). Use readable W&B names `e002a-<source>-<arm>-s0`, with source `loki` or `road-waymo`; fresh local attempt directories distinguish manual restarts. These diagnose effects, not reliable winners. Repeat central comparisons before final claims. Do not multiply the pilot into a backbone/query/LR search.

Agreed fitting settings:

| Setting | Value |
|---|---|
| Loss | Equal-track CE over accepted GT; no unlabeled/padding contribution |
| Optimizer | AdamW, peak LR0.0003 for every trainable module, betas(.9,.999), epsilon1e-8, weight decay1e-4 on all trainable parameters, including biases/LayerNorm/spatial seed |
| Updates | Deterministic FP32; effective batch64 complete tracks via equal-track accumulation; retain final partial update; global gradient clip1 once after accumulation, before stepping |
| Schedule/budget | Three epochs of update-based linear warmup, then constant LR0.0003; maximum100 epochs including warmup, patience15 |
| Execution gate | [Conservative worst-case complete epoch under30 minutes](#throughput-gate--agreed-2026-10-22), including training and full source validation; not met by build6050233 |
| Selection | Highest source-validation track-weighted macro-F1; exact ties lower CE, then earlier epoch; patience resets only for strict F1 improvement |
| Augmentation | Natural missingness and ordinary network dropout; no modality/crop augmentation initially |

For1-based optimizer update `k`, let `W = 3 × ceil(training_tracks /64)` and set `LR(k) = 0.0003 × min(k/W,1)` before that update. Warmup has402 updates for LOKI /207 for ROAD. Microbatch count never advances the schedule. Evaluate full source validation in evaluation mode after every epoch, including warmup; selection/patience operate from epoch1. Epoch training CE aggregates the changing-weight training losses over actual tracks, without an extra training-set pass.

Reset Python/NumPy/PyTorch and the persistent loader generator to0 before each attempt; retain E001 deterministic algorithms, cuBLAS setting, disabled cuDNN benchmarking and TF32 policy. Frozen extraction keeps its separately documented precision/repeatability qualifications. Encoder initialization/extraction must preserve training RNG streams and restore deterministic/backend settings, so cache hits/misses do not alter later dropout. Do not silently relax determinism for trainable operations.

Earlier LR candidates1e-4/1e-3 and one/four spatial seeds are historical development options, not a required sweep; the first pilot fixes LR3e-4 and one seed. Reuse E001 loss/checkpoint/evaluation operations where applicable; preserve existing data/results and default behavior for prior experiments. Atomic `best.pt`/`last.pt` store trainable model state, configuration/normalization references and selection evidence, not copies of frozen encoder weights. Fresh attempt directories, no training resume, automatic restart or retry; persistent frozen-feature cache reuse is independent of training resume.

### Local evidence and W&B

Retain the existing W&B entity/project and tidy namespaces. Main native curves are `train/loss_step` versus hidden `optimizer_step`, `train/loss_epoch`, `val/loss` and `val/macro_f1` versus hidden `epoch`; add `train/learning_rate` versus `optimizer_step`. A step record represents the actual accumulated optimizer update, including the final partial update, not each physical microbatch. Keep detailed support/provenance in config/tables and resource/cache diagnostics in a separate collapsible section; no parameter/gradient histograms or automatic provenance charts.

After reloading the selected checkpoint, save full source-validation predictions with identities/masks/condition membership, primary/per-class metrics, raw/track-weighted confusions and the applicable existing observation strata. Native/E001b strata remain available in every arm. Add `rgb-observed` only for RGB-enabled arms and `lidar-observed`/point counts only for the LiDAR arm; disabled availability is uncomputed, not a false missing observation. Paired summaries check aligned targets/populations and matching measured flags wherever both arms enable that modality; reject mixed fusion protocols. Save PNG/SVG exports and numerical evidence locally; log agreed metrics/tables/plots to W&B, never checkpoint/full-prediction artifacts. No held-out tests or event metrics are added. Record effective configuration, command/revision/diff, hashes/encoder provenance, seed, versions/hardware, optimizer steps, denominators, runtime, GPU/system memory and cache state/traffic/timing. Local evidence precedes external logging; W&B errors propagate without fallback/retry.

One seed diagnoses feasibility and possible modality effects, not reliable superiority. Inspect source-validation class/missing-input changes and compute costs before choosing central comparisons to repeat; negative results do not identify the backbone alone as the bottleneck.

## Complete-track memory and batching policy — agreed 2026-10-21

Keep every selected complete track and every encoder voxel token in its crop. Do not drop long/dense tracks, split them into independent training windows, truncate temporal attention or change inference context to meet a memory limit. Numerical VRAM ceilings and frame concurrency must follow measurement, not the analytical estimates below.

- **Frozen feature supply:** encode scenes without encoder gradients, share scene outputs across represented pedestrians and discard frozen internal activations. Cache frozen outputs before trainable LayerNorm/projection. **Final budget, agreed 2026-10-21:450 GB decimal point payload**, shared across dataset namespaces, training/validation and sequential attempts; byte-based LRU eviction and native decoding/frozen encoding on misses. Use immutable scene/native-frame union entries for the represented tracks'20 m crops, with coordinate/native references and pinned source/encoder/preprocessing identity. Retain every agreed token; capacity controls retention, never population or token selection. Extraction semantics and feature/identity pairing must be equivalent on hits/misses. Cache misses add encoder residency/workspaces to the VRAM budget; fully cached supply may unload encoders. RGB CLS is fully cached separately. The [production cache](#production-feature-caches--2026-10-21) is implemented; integrated training/cache throughput remains unmeasured.
- **Configurable cache budget:** expose `point_cache_gb`, default450 (CLI `--point-cache-gb 450`), and `--point-cache-root`, default `outputs/experiments/E002a/point-cache/` on the Linux output filesystem; convert GB once to integer bytes and record the resolved path/limit. Changing capacity requires no code edit, feature regeneration or cache-identity change; a lower limit trims entries by the same LRU policy. Exclude RGB, indexes, temporary writes and other outputs from the payload limit. Reserve3 GB RGB,15 GB metadata/temporary-write allowance and10 GB pilot outputs, leaving about61 GB operating margin in the [disk audit](#bounded-cache-simulation--2026-10-21). A configurable 20 GB low-free-space guard is a minimum, not a target; check temporary-write headroom and fail visibly rather than silently shrink capacity. The cache CLI and lazy supply API are implemented; behavior-training integration is next.
- **Two batching levels:** globally shuffle complete tracks with one persistent seed0 generator, form64-track logical optimizer updates, then pack physical microbatches **within each update** using point cost `sum(N_bt)` and temporal padding/attention cost. Do not bucket across update boundaries. Match logical update membership across arms; physical microbatches may differ. Spatial frame chunks control point sets resident at once; they never split the temporal sequence. Use lengths/token counts only, never behavior labels or errors. Every track appears once/epoch, including expensive tracks; grouping/ceilings are fixed before a run and recorded.
- **Gradients, conditional memory paths:** ordinary execution is the default for workloads that fit. Use spatial activation recomputation and/or CPU-input staging/offloading only for workloads that require them, according to preflight-calibrated, recorded shape/memory thresholds; no OOM-triggered retry. Retain the small descriptors for full-track temporal attention. Recompute only the trainable readout from frozen features during backward, not the frozen encoder. Preserve RNG behavior and delay updates until all contributions are accumulated; do not detach the readout from behavior supervision. Ordinary GPU-input checkpointing alone is insufficient. Exact implementation details must pass gradient equivalence before execution; [checkpointing](https://docs.pytorch.org/docs/2.7/checkpoint.html) and [saved-tensor offloading](https://docs.pytorch.org/docs/2.7/autograd.html#torch.autograd.graph.save_on_cpu).
- **Effective batch:** accumulate summed individual track losses divided by the logical update's actual track count (64 except the final partial update), then clip once and step. Do not average microbatch means equally when sizes differ. E001 equal-track/partial-GT semantics remain unchanged. Different microbatch shapes need not draw identical dropout masks; reproduce a run with its recorded packing/profile. Fixed-shape recomputation must preserve masks.
- **Preflight:** measure the agreed20 m block on short, typical and long/dense development tracks in both datasets, including naturally missing branches. Time ordinary execution, recomputation and offloading on fitting fixtures, choosing the least overhead path that meets the budget; verify outputs, parameter gradients and one optimizer update within declared FP32 tolerances, including unequal accumulation and temporal padding. Record peak allocated/reserved VRAM, host RAM, transfer/I/O, tracks and point tokens per second, and timings after warm-up. Derive safe microbatch and frame-chunk ceilings from these measurements. If the largest selected example cannot fit, fail visibly and revisit the agreed execution strategy; no OOM-driven retry, silent cap or dropped track.
- **Throughput/concurrency:** run one behavior attempt at a time initially. Maximize measured tracks/second under safe VRAM/host-RAM limits, not utilization percentage or batch size alone. Include actual cold/warm cache I/O, native misses and encoder residency/transients in integrated calibration. Save the resource profile before training; two-process scheduling/shared-cache contention is deferred.

This settles the feature-supply direction and disk payload ceiling, not its numeric RAM/VRAM limits or measured throughput. Recomputation saves activations, not the stored frozen-feature payload. Validate the bounded cache before full preparation: miss timings must include native reads/decoding, transfer, scene reuse and frozen encoding; a small forward-only encoder timing is not end-to-end epoch throughput. No additional sampling or precision reduction is authorized by this policy.

## Analytical VRAM budget — 2026-10-21

For the agreed FP32 spatial block, a small CPU autograd saved-storage inspection (N=257, shared normalized H for K/V, trainable feature LayerNorm) gives **2,592 bytes/token for D-DITR** and **13,168 for Utonia**. Point-dependent retained storage is approximately `4 × sum(N_bt) × (2*d_e + 4*128 + 8)` bytes: raw features and normalized projection inputs, H/normalized H/K/V, and LayerNorm statistics. This is a tensor accounting model, **not a CUDA peak measurement or guaranteed bound**; backend, dropout and tensor-lifetime choices can change allocations.

Illustrative102-slot tracks, constant N at each slot, cached encoder outputs, frozen encoders not GPU-resident. GiB of retained point buffers only:

| Encoder / tokens per frame | Microbatch1 track |2 tracks|4 tracks|8 tracks|
|---|---:|---:|---:|---:|
| D-DITR /16,000 |3.94|7.88|15.76|31.52|
| D-DITR /32,000 |7.88|15.76|31.52|63.03|
| D-DITR /70,056 |17.25|34.50|69.00|138.00|
| Utonia /16,000 |20.01|40.03|80.06|160.11|
| Utonia /32,000 |40.03|80.06|160.11|320.23|
| Utonia /70,056 |87.63|175.27|350.53|701.06|

These N values reference the older15 m sample; actual20 m counts remain unmeasured. Add temporal activations, model/optimizer state, backward temporaries, CUDA context/workspaces and allocator reserve. For varying tracks use actual summed tokens, not necessarily batch×maximum lengths/counts; rectangular padding may incur the latter. Existing OOM evidence reports11.60 GiB total GPU capacity. The current normalized block differs from that earlier stress graph.

Spatial recomputation with **CPU-resident frozen inputs** can instead bound point activations by concurrently processed frames C. Plain checkpointing with every input still on GPU retains a potentially large input payload. A deliberately approximate planning rule `1 GiB + 2 × C × N × bytes_per_token` (GiB conversion implicit) allows one extra local-buffer set plus1 GiB for other costs; it is not a measured peak or safety guarantee. At70,056 tokens and C=1/4/8 this gives D-DITR approximately1.34/2.35/3.71 GiB and Utonia2.72/7.87/14.75 GiB. Host memory, transfer/runtime and temporal microbatch costs still need validation. The calculation itself does not choose numeric microbatch/frame limits; the agreed policy above preserves all tokens and reserves recomputation/offloading for workloads that need them. These cached-feature estimates exclude any co-resident encoder in an on-demand path. Utonia figures remain deferred-alternative evidence.

[Machine-readable assumptions/calculations](../../outputs/experiments/E002a/analytical-vram-budget/budget.json). No behavior implementation or GPU run; original probes/artifacts unchanged.

## Execution and feature-supply probe — 2026-10-21

Authorized bounded check, not behavior fitting. Select minimum/median/maximum slot-length tracks separately within each dataset's previously sampled E002a development groups, from eligible E001 train/validation only; locator breaks ties, no labels/errors or new population filtering. Keep all slots and20 m encoder voxels. Cache3× RGB CLS only for these six complete tracks; population-wide RGB preparation is not implied.

Reuse native readers and pinned encoders. Compare initial native decoding/encoding/crop/cache writing with a separate native traversal and exact feature/native-index checks; measure shared uncompressed scene/union caches, OS-warm cache reads and lossless compression on one frame/group. LiDAR-only native reads omit camera decoding, since RGB CLS is cached. Preserve native, frozen collection/setup and earlier evidence hashes. Do not evict the machine's OS page cache; report warmed-file limitations. No full-population cache extrapolation from these deliberately length-selected tracks.

Temporary resource graph: agreed64/128/128 descriptors,323→256 fusion, two temporal256/four-head/GELU-FFN1024 layers with RoPE. Geometry is zero, cached RGB is real; squared logits provide synthetic gradients, not behavior CE or a behavior result. Probe-only spatial dropout0, temporal dropout.1 and final LayerNorm are recorded, not a hidden fitting search. Compare ordinary/recompute-from-CPU/spatial-saved-tensor-offload paths with microbatch1/2/4; larger microbatches repeat a real complete track as a labeled shape test. First execution warms each condition, then two timed executions; OOM ends that condition without retry, truncation or a smaller substitute. Offload conditions estimated to need more than60% of available host RAM are recorded as unexecuted; this omits a resource stress shape, never a population track. Existing fully extracted/verified fixtures can be supplied read-only via `--fixtures` for a separate resource-only attempt in a fresh output.

Check output, every parameter gradient and one AdamW update across memory paths on a small fitting fixture (FP32 rtol1e-4/atol1e-5). Record peak allocated/reserved VRAM, host RSS, system GPU samples, warm timings, frame/token counts and explicit failures. Cached-readout peaks exclude unloaded encoder weights/workspaces; native supply and readout timings are separate, not a measured overlapped epoch. Selection includes long tracks but does not guarantee the densest20 m crop in either population.

Command from the isolated extractor environment: `python scripts/probe-e002a-execution.py --probe outputs/experiments/E002a/native-probe-07 --output outputs/experiments/E002a/execution-probe --device cuda`. Fresh output required. [Acceptance](../../tests/test_execution_probe.py) checks crop support, selection, relative RoPE, RGB-free native reading and memory-path gradient/update equivalence. No behavior training, held-out access or W&B.

### Execution results — 2026-10-21

**Completed resource attempt `execution-probe-04`**, code `c8b399204f3d87353794065667e022dcfb1818e2`,123.4 s for resource-only checks. Native fixtures were extracted in `execution-probe-02`: six complete tracks/311 slots,193 valid3× RGB CLS descriptors and304 unique scene sweeps verified on a separate native traversal. Shared referenced point payload is **1.253 GiB** across216 referenced scene-cache files; six RGB caches total **963,715 bytes**. Empty/unusable point slots remain temporal positions. This deliberately selected sample is not a population storage estimate.

Peak **allocated GiB / warm forward+backward seconds** for the temporary graph; microbatch4 repeats each complete track as a shape test:

| Track | Slots | Summed point tokens | Ordinary B=1 | Ordinary B=4 | Recompute B=4 |
|---|---:|---:|---:|---:|---:|
| loki-short | 1 | 62,379 | 0.32 / 0.037 | 0.79 / 0.128 | 0.33 / 0.169 |
| loki-typical | 22 | 1,368,425 | 3.54 / 0.585 | OOM | 0.37 / 3.216 |
| loki-long | 102 | 988,192 | 2.59 / 0.417 | 9.88 / 1.665 | 0.28 / 2.439 |
| road-waymo-short | 7 | 4,383 | 0.11 / 0.006 | 0.14 / 0.017 | 0.10 / 0.024 |
| road-waymo-typical | 79 | 2,021,044 | 5.18 / 0.866 | OOM | 0.48 / 4.946 |
| road-waymo-long | 100 | 83,704 | 0.30 / 0.084 | 0.93 / 0.389 | 0.11 / 0.772 |

Dense LOKI22-slot B=2 also fits at6.88 GiB allocated/7.66 reserved,1.168 s; dense ROAD79-slot B=2 reaches10.13/10.84 GiB,1.743 s, with little headroom. Recomputed B=4 peaks at0.37/1.34 GiB allocated/reserved for LOKI22 and0.48/2.93 for ROAD79. Recomputation adds about38–43% to B=1 compute on the substantial LOKI22/102 and ROAD79 cases; relative overhead is larger on tiny/sparse workloads. These are cached-feature resource graphs, not measured behavior epochs; full encoder residency, real feature assembly/CE/optimizer-step peaks still require final integration checks.

Of54 declared conditions, **46 complete, two ordinary B=4 OOM, six offload conditions unexecuted under the host-RAM guard**. Host RSS reaches12.14 GiB for the probe process; host headroom depends on other applications and earlier pinned allocations. Ordinary execution remains preferable where it fits. For example LOKI22 offload B=1 takes1.398 s versus ordinary0.585 s and recompute0.805 s; it is not a default speed optimization. No selected track or point token was removed.

The baseline native traversal/encoding totals100.64 s across304 sweeps, excluding cache-equality checks. All six OS-warm point/RGB cache reads total1.01 s. They are separately measured supply paths, not an overlapped end-to-end epoch; the baseline encodes even timestamps with no usable target crop. A selective on-demand audit skips unavailable target-crop timestamps while retaining the labeler's full temporal sequence: **216 sweeps take74.24 s**, plus0.44 s reader setup and separate encoder initialization. Every resulting cached feature matches exactly. This is still much slower than1.01 s warmed cache supply on this sample, but is not a full-epoch estimate. Lossless `.npz` on five median nonempty group frames retains **88.5–88.7%** of the uncompressed size but takes **2.9–4.6×** as long to read/verify under warmed-file conditions. This does not resolve the earlier terabyte-scale full-cache concern.

**Recommendation:** use all-token-cost-aware microbatches, ordinary execution for fitting shapes and recomputation for expensive ones; avoid routine CPU offloading. Prefer shared uncompressed point-feature caching for the bounded pilot if its declared disk budget fits, rather than repeated encoding each epoch. Full-population feature supply and disk capacity remain open; no full cache or behavior training was launched.3× RGB CLS caching is agreed; only these six-track fixtures are prepared so far.

**Acceptance/evidence:** outputs, every parameter gradient and one synthetic AdamW update match across memory paths at rtol1e-4/atol1e-5 (maximum reported difference1.94e-7). All **890 protected** sources/fixtures remain unchanged. Independent artifact checks reconstruct point counts, unique rows,20 m membership, finite pickle-free float32 caches and zero missing RGB rows. CPU116 (six CUDA skips), CUDA116 and isolated-runtime5 checks pass.

This preparatory graph used spatial dropout0 and temporal0.1. Its measurements/equivalence evidence remain unchanged; the final spatial0.1 policy requires the newly declared fixed-shape dropout/RNG and integrated resource checks before behavior training.

Manual attempts remain: `execution-probe` failed on integer timestamp filename concatenation; `execution-probe-02` completed native extraction but was manually interrupted before unguarded offload stress, with interruption also preventing its final evidence write; `execution-probe-03` failed the GPU optimizer-update equivalence check with default biased spatial attention. The final graph uses the specified bias-free spatial projections; equivalence passes without relaxed tolerances. Attempt04 independently hashes/verifies the reused fixtures and original sources. Original1.5×/15 m probes and all E001-family data/results are unchanged.

[Local report](../../outputs/experiments/E002a/execution-probe-04/report.md), [numerical evidence](../../outputs/experiments/E002a/execution-probe-04/execution.json), [artifact audit](../../outputs/experiments/E002a/execution-probe-04/artifact-audit.json), [nonempty compression audit](../../outputs/experiments/E002a/execution-probe-04/compression-audit.json), [selective on-demand audit](../../outputs/experiments/E002a/execution-probe-04/selective-on-demand-audit.json), [GPU system samples](../../outputs/experiments/E002a/execution-probe-04/system.csv) and [validation/failure logs](../../outputs/experiments/E002a/execution-validation/) are collected locally.3× RGB fixtures are also local in `outputs/experiments/E002a/execution-probe-02/rgb/`. Full point fixtures remain on aalto at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/execution-probe-02/`; final resource evidence is in sibling `execution-probe-04/`.

## Bounded-cache simulation — 2026-10-21

Authorized metadata-only replay before implementing a400 GB output-feature cache. Simulate each source independently, cold initial cache, six epochs of the existing seed0 persistent whole-track shuffle and ordered source validation after each epoch. One byte-bounded LRU is shared between training/validation. Entries are immutable native scene/frame union features; training population, point tokens and temporal context are unchanged. Use frozen eligible train/validation metadata with native3D, position/pose and anchor-valid input masks; never open held-out archives or use GT to rank/filter cache requests.

True20 m entry sizes are unknown. Calibrate nominal bytes separately to the earlier15 m source/split union estimates, with uniform per-frame sizes or group-varying sizes based on observed random-group means and deterministic imputation for other groups. Normalize both models to the same split total, and vary footprint by.75×/1×/1.5×/2× at400 GB. Controls: no-cache/unbounded at1×;8/200/600 GB at1× for both models. The8 GB case is a small-capacity locality reference, not a chosen RAM allocation. GB is decimal (400 GB=372.5 GiB). These are sensitivity scenarios, not measured20 m footprints or confidence intervals.

Report cold and epochs2–6 training/validation hit rates, prior-epoch versus intra-epoch reuse, avoided encoder requests, byte reads/writes/evictions and resident size. Preserve reusable pickle-free request/order traces, estimated-size provenance, input hashes and failure evidence. This does not measure actual cache I/O, atomic-write overhead, asynchronous overlap or epoch speedup; future memory-bucketing/frame sharing can change request order and avoid calls even without a disk cache. Fixed track-sequential replay is a reference, not a finalized E002a loader policy.

Command: `uv run --locked --extra cpu python scripts/simulate-e002a-cache.py --output outputs/experiments/E002a/cache-simulation`. Fresh output required. [Hand-calculated acceptance](../../tests/test_cache_simulation.py): byte eviction/recency, oversize bypass, cyclic thrashing, cold/unbounded controls, input-only selection and exact reference-loader shuffle. No encoder inference, native-data decoding, full feature cache or training.

**Completed `cache-simulation-02`** at revision `67291101a588222dbf9a34d7b7bccd96cced4b5a` in38.8 s locally:32 scenarios, six epochs each,384 training/validation phase records. All **15,930 protected files** retain their hashes. Initial replay `cache-simulation/` is retained; the final replay includes the8 GB controls in the reproducible command.

| Source | Eligible tracks train/val | Feature requests train/val | Unique frames train/val | Nominal footprint train/val GB |
|---|---:|---:|---:|---:|
| LOKI |8,574 /1,861|262,071 /60,231|26,017 /5,273|389.6 /39.2|
| ROAD-Waymo |4,380 /1,110|302,264 /78,730|42,917 /9,348|496.3 /341.5|

400 GB cache, mean epochs2–6; ranges span uniform/group-varying size models. Each source starts separately; footprints are estimated, not measured20 m payloads.

| Footprint multiplier | LOKI hit % | ROAD-Waymo hit % | LOKI writes GB/epoch | ROAD-Waymo writes GB/epoch |
|---|---:|---:|---:|---:|
|0.75×|100|88.5|0|553–554|
|1×|97.0|85.9|106–107|852–854|
|1.5×|87.1–87.2|80.9–81.0|866–868|1,594–1,611|
|2×|77.9–78.2|74.4|2,024–2,053|2,703–2,721|

At nominal size, training-only hits are98.3% LOKI /85.3% ROAD with400 GB versus only4.2–4.5% with8 GB. The small cache's20–22% combined hit rate mainly reflects ordered validation reuse. Much of ROAD's400 GB benefit is also **within an epoch**, sharing frames between pedestrians: only about900–1,023 hits/epoch reuse entries last accessed in an earlier epoch, versus about21,495 for LOKI. Validation eviction is included. Existing in-memory frame sharing could capture some of these hits without disk caching; avoided requests are relative to literal track-sequential extraction.

**Recommendation:** a400 GB bounded shared cache merits a live prototype; the trace does not support dismissing it as overhead alone. Do not claim wall-time speedup yet. Nominal logical full-entry reads reach4.3 TB LOKI /5.5 TB ROAD per epoch, alongside the writes above. These are modeled accesses, not physical SSD traffic: OS caching, selective crop reads and overlapping I/O can change the cost. Measure actual20 m entry sizes, disk throughput, native miss latency and one representative epoch before choosing production feature supply. Reserve write/index headroom outside the payload ceiling. No production cache, behavior model or training was added by this simulation; all tracks/context/tokens remain intact.

[Report and full tables](../../outputs/experiments/E002a/cache-simulation-02/report.md) · [Numerical results/provenance](../../outputs/experiments/E002a/cache-simulation-02/simulation.json) · [Independent audit](../../outputs/experiments/E002a/cache-simulation-02/artifact-audit.json) · [LOKI trace](../../outputs/experiments/E002a/cache-simulation-02/loki-trace.npz) / [ROAD trace](../../outputs/experiments/E002a/cache-simulation-02/road-waymo-trace.npz) · [CPU acceptance](../../outputs/experiments/E002a/cache-simulation-02/validation/cpu-tests.log). CPU121 checks pass with six CUDA-only skips; byte/request conservation, capacity/scale controls, source hashes and saved sampler permutations pass independently. CUDA execution is unnecessary for metadata replay.

**Disk check, 2026-10-21 12:29 UTC:** aalto's resolved E002a root is `/home/user20/projects/pedestrian-behavior-labeling/outputs/experiments/E002a/`; the review checkout links there. Its Linux filesystem has **539.12 GB available** (`df -B1`, user-available bytes), already accounting for the existing18.89 GB E002a directory, including extractor environment9.04 GB and weights1.45 GB. The native-data NTFS partition has only36.80 GB available and is not the proposed cache location. No files were deleted or moved.

The frozen eligible train/validation populations contain753,019 temporal slots: dense float32 RGB CLS storage is **2.313 GB** at768 values/slot, even retaining all missing rows. Reserve3 GB for RGB/masks/identities,15 GB for point indexes/metadata and temporary writes, and10 GB for new pilot outputs/checkpoints/logs. These are planning allowances, not measured new artifacts; full RGB image/crop storage is not planned. A **450 GB point payload** leaves about61 GB beyond these allowances, versus only11 GB at500 GB (475 GB leaves36 GB). **User accepted450 GB as the final configurable default**, superseding the earlier500 GB agreement. Preserve the400 GB simulation as measured evidence; no450 GB hit rate or live speedup has been measured. The production implementation/acceptance and subsequently completed full RGB cache are recorded below; point payloads remain lazy.

## Production feature caches — 2026-10-21

Implemented separately from behavior fitting. Use the pinned cu126 extractor environment for lazy native/encoder supply; the E001 cu118 environment is unchanged. [Commands](../../README.md#e002a-feature-caches), [storage/payload acceptance](../../tests/test_feature_cache.py). The cache API can supply any frozen eligible source training/validation track; a bounded selection affects preparation requests, **not** the population defining a frame's voxel union. Test archives are never opened by the catalog.

- **RGB:** one persistent uncompressed, pickle-free entry per complete track: float32 `[T,768]` CLS, validity, native keys/times and compact UTF-8 JSON crop bounds/padding/intersection evidence. Missing rows stay zero. Original 256², 3× square, native-only boxes and fixed 16-crop batches; no pixels retained, no LRU. Initialization and extraction are lazy.
- **Points:** one immutable union per dataset/scene/native frame, shared across all eligible represented pedestrians. Store float32 features64/XYZ, int64 native representative indices, native cuboids, frame transform and initial ego anchors. Recover every track's 20 m cylinder, relative XYZ and geometric membership on read; no duplicate per-track feature arrays or extra sampling. Missing slots return empty arrays; no usable sweep gives an empty entry. Missing required transforms/cuboids, malformed data and nonfinite features fail.
- **Storage:** one 450,000,000,000-byte LRU across both point namespaces, source training/validation and attempts; count actual NPZ file bytes, including embedded identity. SQLite index/temporary files are outside the cap. Measured per-track token/inside-box counts live in a checksummed persistent index outside the payload ceiling, survive eviction and return `unknown` until measured (`FeatureSupply.point_cost`). Capacity/root are configurable; shrinking trims oldest entries without changing identity. Oversize entries are served without retention and counted explicitly, including a zero-capacity diagnostic; inputs/tokens remain identical. RGB and point roots cannot overlap data or each other.
- **Integrity:** checkpoint/config/author revisions, relevant extraction-code hashes, backend versions, frozen setup/collection/all source-track hashes, scene/raw-source snapshot and participants identify entries. Raw images/LiDAR use resolved path/size/mtime snapshots once per scene/attempt, rather than rehashing terabytes each epoch; native sources must remain immutable during an attempt. Native annotation/association SHA checks run before decoding. Payload SHA and exact dtype/shape/finite checks run on every hit; corruption never becomes an automatic miss/retry.
- **Writes/ownership:** exclusive process lock and zero workers; refuse nonempty unindexed directories or foreign indexes before cleanup; fsync payload, atomic rename, then committed index. Interrupted unindexed/temporary writes are disposable and cleaned on reopen; indexed missing/corrupt payloads fail. Check the configurable 20 GB free-space floor **plus the new temporary payload** before writing; failures propagate without shrinking capacity. No concurrent cache owners or training resume machinery.
- **Supply/runtime:** `FeatureSupply.rgb(dataset, locator)` and `iter_points(dataset, locator)` preserve complete slot order. The latter shares one native traversal across ordered misses; `points(..., slot)` supports isolated requests. Hits do not load encoders/native readers. Only one frozen encoder is resident; explicit `release()` frees it. Seeded extraction restores Python/NumPy/CPU/CUDA RNG and deterministic/cuDNN/TF32/workspace settings even on failure; cache capacity/hits cannot change later dropout.

`cache-e002a.py` prepares all source RGB by default or explicitly warms point entries. Each invocation writes a fresh local configuration/provenance/status/counters/hash record; reuse of existing valid feature entries is independent of training resume. `check-e002a-cache.py` requires an explicit small development selection and empty dedicated caches, checks real cold/warm/reopened supply, native re-encoding, shared pedestrians and capacity reduction, and keeps failures. No W&B or behavior training in cache acceptance.

**Native acceptance at `70e88de`; final storage ownership guard at `7f0f51d`.** RTX4080, dedicated test caches, same six development tracks. Full CPU **144 tests / seven CUDA skips**, CUDA **144**, isolated extractor **23** pass; `uv lock --check`, `uv pip check`, documentation links and `git diff --check` pass. Tests cover byte-LRU recency, persistence/capacity reduction/oversize bypass, atomic-write/low-space failures and orphan recovery, locking/refusal of foreign cache roots, corrupt/nonfinite/unsafe arrays and cost-index checksums, missing RGB, full union geometry, source-only population selection, RNG/error restoration and propagated failures with local evidence.

| Check | Observed result |
|---|---|
| Native cold/warm/reopen/eviction |97.7 s; six complete RGB tracks /311 slots,193 valid CLS;11 bounded point requests /10 shared union entries |
| Independent native extraction |Full-sweep re-encoding on each dataset exactly matches cached feature/native-index/coordinate/cuboid pairing; another pedestrian reuses the same entry |
| Capacity/RNG |Ten entries evicted at zero capacity; all11 measured cost references survive, re-encoded crop is identical; Python/NumPy/CPU/CUDA RNG and deterministic settings preserved |
| Complete-track streaming |311 slots /218 point requests /216 unique entries; every per-track token count exactly matches the earlier verified fixture |
| Fresh-process warm replay |218 hits, zero misses/encoder loads/calls/writes; source catalog/protection checks included in31.6 s total, versus171.7 s for the partly cold streaming pass |
| Actual storage/integrity |Point payload4,148,833,720 bytes (4.15 GB),216 entries,220 persistent cost references; physical NPZ bytes equal the index; RGB1,080,829 bytes |
| Protected sources |15,944 unchanged in native acceptance;15,934 unchanged in each full-track replay |

The whole-track partly cold pass starts with one retained entry from eviction acceptance:215 new encodes /3 hits, not an empty-cache epoch. Point read/checksum/validation time is12.47 s for the fresh warm replay; cold misses take119.31 s and payload writes13.72 s. End-to-end totals also include full source-catalog validation, native stat/index work, crop reconstruction, provenance and protected-file checks. OS files are warm; no page-cache flush, training overlap or full-population hit-rate claim. The4.15 GB payload contains unions for **all eligible represented pedestrians** in those frames, whereas the older1.253 GiB resource fixture only represented its six selected tracks; this is the expected union scope, not extra point sampling or wider crops. The bounded native check peaks at1.01 GB CUDA allocated /5.34 GB host RSS, excluding any behavior model.

[Native acceptance/provenance](../../outputs/experiments/E002a/cache-acceptance-02/acceptance.json) · [independent artifact/count/index audit](../../outputs/experiments/E002a/cache-acceptance-02/artifact-audit.json) · [final storage guard/all-entry audit](../../outputs/experiments/E002a/cache-acceptance-02/storage-ownership-audit.json) · [complete-track record](../../outputs/experiments/E002a/cache-stream-check-02/cache.json) · [fresh-process warm record](../../outputs/experiments/E002a/cache-stream-warm-02/cache.json) · [CPU](../../outputs/experiments/E002a/cache-cpu-tests.log) / [CUDA](../../outputs/experiments/E002a/cache-cuda-tests-final-guard.log) / [extractor tests](../../outputs/experiments/E002a/cache-extractor-tests-final-guard.log). Records/numerical reference fixtures are local and in `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/`; test payloads stay on aalto in `cache-acceptance-02-data/`. Earlier `cache-acceptance` / `cache-stream-check` records at5271698 are retained; final acceptance adds the persistent cost-index checks. No behavior training or W&B runs.

At that storage acceptance, full-population RGB preparation was unexecuted; it is now complete in the model/training acceptance below. Representative-epoch physical I/O/throughput remains unmeasured; bounded OS-warm reads cannot establish the earlier modeled hit rates or training speedup. The final storage guard rereads/checksums all216 real entries and220 cost references without inference; extraction/selection semantics are unchanged from70e88de. The production system is implemented/tested; population-wide cache filling is separate from this acceptance.

## Model and training acceptance — 2026-10-21

**Historical shared-availability-v1 acceptance.** The22 October [enabled-branch revision](#enabled-branch-fusion-revision--2026-10-22) supersedes its all-arm fusion/availability policy. Original numerical evidence and operational runs below are retained unchanged; they do not calibrate the revised code.

Implemented `models/multimodal.py`, complete-track packing/accumulation, the E002a feature adapter and [single-run/sequential-pilot commands](../../README.md#e002a-source-validation-pilot). Exact E001b source-normalized geometry and all frozen populations remain unchanged. Shared E001 loss, selection, atomic checkpoints, inference, prediction archives, strata, plots and W&B are reused through optional hooks; old callers/defaults pass regression checks. Trainable attention uses PyTorch SDPA **math/FP32**: fused CUDA attention failed mask-stride alignment on irregular point counts in the first calibration. That failed attempt is retained; determinism is not relaxed.

Packing starts with persistent shuffled64-track logical membership, then stable first-fit by length/cost within each update. Unknown costs stay single until measured; no across-update bucketing. `packing.jsonl` records pass, logical/physical indices, locators, lengths, point counts and unknown-cost membership, since different physical shapes may draw different dropout masks. Ordinary execution is selected by **padded** spatial-token workload; expensive shapes checkpoint only the trainable readout with CPU inputs. Offload remains verified but is not the automatic default. Full temporal attention always sees every slot. One frozen encoder may stay resident across misses; preflight reserves2 GiB for miss transients and rejects unsafe measured shapes without retry.

The completed-profile gate binds source code, PyTorch/CUDA version, device and GPU model. Full attempts require the matching completed full-population RGB preparation record; an RGB miss fails instead of silently encoding. Point capacity remains450 GB by default. Status/config/diff/provenance, per-update loss/LR, epochs, exact packing, cache traffic/timing, host memory and GPU telemetry stay local; provenance/support are config/table content in W&B. Checkpoints contain readout state and frozen references, not encoder weights. Only source validation is predicted after selection, with observed-RGB/LiDAR strata. The six-attempt launcher summarizes paired track changes for RGB versus geometry and LiDAR versus RGB as well as the combined change, using saved predictions.

Full RGB preparation uses scene-order traversal, a preparation-only2 GB decoded-image LRU and four CPU cropping threads. Per-track ordered16-crop encoder batches and every pixel/CLS/mask/reference remain unchanged. Native acceptance compares this route against independent serial per-track extraction exactly. Earlier slow preparation attempts are retained as interrupted evidence; code changes use a new immutable cache namespace. No established collection, bbox cache or result is rewritten.

**Operational acceptance complete.** Final feature extraction and calibration ran at `bea99d8`; subsequent protection, audit/configuration and explicit preparation-gate changes are accepted at `e50d79e`. Six configuration assemblies pass without inference, optimizer updates or W&B. Total model size is1,975,044 parameters; trainable counts are1,669,508 (`g`),1,767,940 (`g-rgb`) and1,975,044 (`g-rgb-dditr`). Frozen encoder parameters are separate.

| Check | Expected → observed |
|---|---|
| Tensor/missingness/temporal contract | Shared flags and initialization; absent descriptors zero; GT ignored; future context/RoPE through gaps and padding exclusion pass |
| Accumulation/recomputation | Unequal/final partial equal-track aggregation passes; ordinary/checkpoint/offload outputs, all gradients and AdamW updates agree at the prescribed tolerances, including dropout0.1 and frame concurrency1/2/4 |
| Calibrated execution | Actual model CE/backward/clip/AdamW on six complete fixtures plus labeled shape stresses; final profile completes without OOM, peak CUDA allocation7.54 GB; one-frame spatial chunks selected from timed1/2/4 checks |
| Native feature parity | Four-thread scene-image reuse matches independent serial RGB extraction exactly on six tracks; native point re-encoding/shared unions/eviction/RNG checks pass;15,944 protected files unchanged |
| Cold/warm integration | One epoch/update on three training tracks plus one previously inspected source-validation track per dataset, cold then warm; warm attempts load neither encoder and make zero encoder calls |
| Final-profile integration | Both datasets pass selected-checkpoint reload, saved-prediction metric reconstruction and22 PNG/SVG pairs each; no held-out inference |
| W&B/local failures | Six actual smoke runs finished; cloud audit confirms only agreed scalar/table/image content, no checkpoint or full-prediction uploads; injected logging failures propagate with local evidence preserved |
| Regression/dependencies | Full CPU157 tests (eight CUDA-only skips), CUDA157 with no skips; actual cu126 extractor13 model tests with no skips. Lock/package checks and documentation links pass |

The selected [resource profile](../../outputs/experiments/E002a/model-calibration-05/profile.json) permits64 tracks,665,856 temporal cells,2,736,850 padded spatial tokens per physical microbatch/ordinary path, and524,288 tokens per one-frame spatial chunk. Larger measured workloads use CPU-input spatial recomputation; full temporal context remains intact. Calibration reserves2 GiB for frozen-encoder miss transients. These are profile-specific measured/estimated safety gates, not universal memory guarantees. Earlier calibration directories, including the failed fused-attention attempt, remain intact.

| Final-profile smoke | End-to-end seconds | Peak CUDA allocated | W&B |
|---|---:|---:|---|
| LOKI |95.6|3.95 GB|[jwjag44d](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/jwjag44d)|
| ROAD-Waymo |240.0|5.70 GB|[2mvj6nrx](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/2mvj6nrx)|

These are tiny operational attempts, not behavior comparisons or epoch forecasts. [Cold/warm integration record](../../outputs/experiments/E002a/model-integration/integration.json) · [final local/cloud artifact audit](../../outputs/experiments/E002a/model-integration-final/artifact-audit.json) · [native extraction acceptance](../../outputs/experiments/E002a/cache-acceptance-04/acceptance.json) · [six-configuration readiness](../../outputs/experiments/E002a/launch-readiness-02/readiness.json) · [CPU](../../outputs/experiments/E002a/model-cpu-tests-handoff.log) / [CUDA](../../outputs/experiments/E002a/model-cuda-tests-handoff.log) / [cu126 model tests](../../outputs/experiments/E002a/model-extractor-tests-handoff.log).

**Full 3× RGB cache complete:** 15,925 tracks /753,019 slots /366,653 valid CLS. Preparation took6,458.8 s;15,934 protected files were unchanged. All entries were then independently read/checksummed with15,925 hits, zero misses, zero writes and zero encoder calls; frozen per-track GT/class support matches setup. Active payload is2.62 GB (the root also retains older immutable namespaces). Missing RGB remains explicit:

| Frozen split | Tracks | Context slots | Accepted GT frames | RGB slots | RGB-supported GT frames |
|---|---:|---:|---:|---:|---:|
| LOKI training |8,574|270,355|262,071|124,961|118,451 (45.2%)|
| LOKI validation |1,861|61,900|60,231|26,079|24,784 (41.1%)|
| ROAD-Waymo training |4,380|332,940|167,280|169,660|167,280 (100%)|
| ROAD-Waymo validation |1,110|87,824|45,388|45,953|45,388 (100%)|

[Completed preparation](../../outputs/experiments/E002a/rgb-cache-preparation-03/cache.json) · [all-entry readback and class/group/track support](../../outputs/experiments/E002a/rgb-cache-audit/audit.json). These counts describe the frozen pilot cohorts, not unrestricted native populations. Full-population LiDAR hit rate and epoch time remain unknown until the manual pilot; the450 GB LRU fills lazily. No six-attempt comparison or held-out inference was launched.

## Full-population throughput check — 2026-10-22

Read-only inspection at03:53 UTC of the user-launched `pilot-20261021-183026-855079/e002a-road-waymo-g-s0`, code `6050233`. [Progress/cache/process snapshot](../../outputs/experiments/E002a/throughput-check-20261022/snapshot.json) · [actual W&B step timestamps](../../outputs/experiments/E002a/throughput-check-20261022/wandb-timing.json) · [run81h8shqe](https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/81h8shqe).

**Historical measurement scope:** the snapshot, diagnosis and timings below describe the named October21–22 builds, populations, hardware/backend and cache conditions. They are not timeless properties of E002a, D-DITR or future loaders. Preserve them as evidence; after an execution/cache change, record new measurements and their revision rather than carrying these rates forward as current facts.

| Measure | Observed |
|---|---:|
| Elapsed / completed epochs |9 h23 min /0|
| Completed optimizer updates |33 /69 in training epoch1|
| Median inter-update interval / latest five mean |14.82 min /13.56 min|
| GPU mean utilization / zero-utilization samples |20.3% /68.1%|
| Process block read/write counters |3.14 TB /619 GB|
| Point payload / retained entries |449.97 GB /23,424|

The first geometry arm computes the same measured LiDAR-availability flags as the other arms. `FeatureDataset.__getitem__` therefore calls full `iter_points` whenever point-cost metadata is unknown, loading or producing complete frozen feature unions merely to obtain crop counts. Random track visits also repeat native scene indexing/Parquet traversal; hits read/checksum/validate entire frame unions. The first epoch is dominated by feature supply. Once its metadata exists, geometry/RGB arms bypass point payloads; this does not establish usable LiDAR-arm throughput. Per-stage live timing is not persisted until epoch completion, so precise shares of decoding, encoding, cache reads/writes and synchronization are still unknown.

At the recent rate, the remaining36 training updates alone would take about8.1 h, plus validation; this is a conditional extrapolation, not an epoch measurement. Bounded three-track smoke acceptance checked correctness but was insufficient to establish practical population-scale throughput. The earlier declaration of pilot readiness is qualified by this finding; do not infer acceptable performance from cache-hit percentages or fitting model shapes.

Recommended next increment: derive exact availability/cost metadata from native voxel coordinates without encoder inference, verify parity with existing crop counts/flags, and keep geometry/RGB fitting independent of point-feature payloads. Then measure representative full64-track cold/warm multimodal updates and native/cache stage costs before proceeding with the sweep. Preserve existing valid cache entries, every track/token, frozen data and scientific settings. This inspection changes no training code or running process; execution-path changes need measured parity/throughput acceptance. No architecture change or new encoder is justified by this bottleneck.

### Build-scoped hit/miss costs

Measured on2026-10-21 at **`7940c21`**, RTX4080 Laptop GPU / pinned cu126 extractor environment, using `g-rgb-dditr` smoke attempts with three complete training tracks plus one source-validation track per dataset. These are averages over that small sample, not population latency bounds. A request is one pedestrian/timestep lookup into a shared frame union, not a whole track or optimizer update. [Recorded counters](../../outputs/experiments/E002a/model-integration-final/artifact-audit.json) · [source provenance: LOKI cold](../../outputs/experiments/E002a/model-integration/loki-cold/provenance.json) / [warm](../../outputs/experiments/E002a/model-integration/loki-warm/provenance.json), [ROAD cold](../../outputs/experiments/E002a/model-integration/road-waymo-cold/provenance.json) / [warm](../../outputs/experiments/E002a/model-integration/road-waymo-warm/provenance.json).

| Dataset | Warm hits measured | Mean hit read/check/load | Cold misses measured | Mean miss production | Mean write | Production + write |
|---|---:|---:|---:|---:|---:|---:|
| LOKI |148|43.0 ms|117|365.2 ms|61.3 ms|426.5 ms|
| ROAD-Waymo |310|73.9 ms|219|667.2 ms|72.7 ms|739.9 ms|

Hit timing includes file/OS-cache→RAM, SHA verification, NPZ loading/validation and LRU-index update; it excludes later crop construction and RAM→VRAM transfer. Files/OS cache were warm, with no page-cache flush: this is not isolated physical-disk latency. Miss production includes native frame decoding, encoder preprocessing/inference and union construction; additional scene indexing/planning outside that timer is excluded. ROAD's219 misses total146.1 s production, including58.9 s in the encoder wrapper, plus15.9 s writing. Its warm310 reads total22.9 s for8.84 GB. These counters do not partition the nine-hour full run. In build6050233's geometry arm, cached point features are used for counts/flags on the CPU and are not uploaded to the trainable spatial branch; this observation does not establish transfer cost or memory pressure for the LiDAR arm.

**Illustrative supply-only extrapolations, not measured epochs:** frozen input masks imply322,302 LOKI /380,994 ROAD requests per complete training+validation epoch ([trace counts](../../outputs/experiments/E002a/cache-simulation-02/simulation.json), build6729110). Assuming the small-sample mean costs apply uniformly:

| Dataset | All requests hit | Hypothetical10% misses | All-hit bytes read |
|---|---:|---:|---:|
| LOKI |3.85 h|7.28 h|5.14 TB|
| ROAD-Waymo |7.82 h|14.87 h|10.86 TB|

ROAD has302,264 training +78,730 validation lookups. At73.876 ms each, `380,994 ×0.073876 /3600 ≈7.82 h`; one shared frame union can be reread for several pedestrians. General formula: `requests × ((1−m) × hit_seconds + m × (production_seconds + write_seconds)) /3600`. Here `m=0.1` is an assumed miss fraction, not a measured full-population hit rate. Union sizes/locality may differ across the population; extra indexing, crop/transfer, model, validation and reporting work is omitted. The calculation warns that higher hit rate alone may be insufficient for this build, not that future implementations require these epoch times. [Exact inputs, source hashes and calculations](../../outputs/experiments/E002a/throughput-check-20261022/timing-budget.json).

### Storage-layout estimates — 2026-10-22

**Estimates for the October22 build6050233 feature contract, not implemented layouts or measured new-loader speeds.** Read only the active cache's small SQLite count/index records in short batches; no encoded feature payload reads, encoder calls, cache ownership or training changes. The07:31 UTC [count snapshot](../../outputs/experiments/E002a/storage-layout-estimate-20261022/count-snapshot.json) matches the current extraction namespace. ROAD training has220,140 measured crop requests over3,194/4,380 tracks (72.9% of tracks); its retained LRU sample contains23,331 training frame unions from226 groups. Observed mean crop size is26,028 points; mean retained union is19.29 MB. Validation sizing below uses the training mean because only one validation track is measured. The LRU sample is not random and may favor popular/dense frames.

All sizes below cover the complete frozen source training+validation populations, not the450 GB LRU ceiling. Decimal TB; uncompressed arrays. Full metadata/temp/output/backup/headroom and native datasets are additional. ROAD's52,265 unique frames extrapolate to1.01 TB shared payload;9.92 billion pedestrian/point instances extrapolate to2.78 TB model-ready per-track data. Frame-local uint32 indices add39.7 GB (uint64:79.3 GB).

| Proposed layout | ROAD FP32 | ROAD FP16 features | Both sources FP32 | Both sources FP16 features |
|---|---:|---:|---:|---:|
| Shared frame arrays + uint32 crop indices |1.05 TB|0.58 TB|1.51–2.09 TB|0.84–1.16 TB|
| Contiguous model-ready per-track copies |2.78 TB|1.51 TB|4.31–5.65 TB|2.34–3.07 TB|

The combined-source ranges are **low-confidence planning scenarios**, not confidence bounds or purchase-ready sizing. LOKI has only three training/one validation track with current20 m costs. Its bracket combines the older15 m random-group duplicated/union estimates with the sparse20 m crop counts and older full-scene estimate; more representative20 m evidence is needed. ROAD's unseen frames/validation may also differ from the observed mean.

Byte assumptions: shared point row256 feature +12 XYZ +8 native-index =276 bytes; model-ready per-track row256 feature +12 relative XYZ +4 membership +8 native-index =280 bytes. In the indexed option, relative geometry is derived after gathering; fully precomputing it as well adds16 bytes/selected point, approximately0.16 TB ROAD (shared totals then1.21 TB FP32 /0.74 TB FP16 features). Small per-frame box/pose/offset tables are additional. Frame-local uint32 assumes fewer than2³² rows/frame and must be checked on conversion. No point sampling or token removal is assumed.

**FP16 is a hypothetical storage change for the64 encoder channels only.** Coordinates, masks/geometry and identifiers retain their stated precision; shared/per-track rows become148/152 bytes. This saves about46% of payload, rather than exactly half of every cache field. The cache contains encoder features, not four-class behavior logits. FP16 storage could be read into FP32 training; AMP training is a separate change. Feature error, overflow and downstream effects have not been accepted. FP8/8-bit storage is not selected. Its byte-budget-only extension assumes one byte per feature channel: shared/per-track rows84/88 bytes before quantization scales/format metadata. This projects ROAD shared+uint32 indices to0.35 TB and per-track copies to0.87 TB; both-source planning ranges become0.50–0.69 TB and1.35–1.78 TB, respectively. ROAD shared storage would fit the450 GB payload ceiling on these assumptions, but sizing remains uncertain and no FP8 encoding, scaling, error or downstream accuracy has been validated. Storage quantization does not imply FP8 training or faster FP32 model execution.

**Loading-only ROAD epoch scenarios:** assume each selected point instance is read once, and an implemented contiguous layout sustains the stated effective read/load rate. These exclude native misses/preparation, crop assembly/dequantization, transfer, model/validation compute and reporting. Current get/read/check/load averages near0.4 GB/s came from the older warm-file probe, not these hypothetical formats;1/2 GB/s are assumed achieved rates, not SSD specifications or benchmarks.

| Per-track representation | At0.4 GB/s | At1 GB/s | At2 GB/s |
|---|---:|---:|---:|
| FP32,2.78 TB |116 min|46 min|23 min|
| FP16 features,1.51 TB |63 min|25 min|13 min|
| FP8 features,0.87 TB; byte-budget only |36 min|15 min|7 min|

Without RAM reuse, ideal indexed partial reads request a similar point-instance volume; sharing saves storage but does not automatically save epoch read traffic. Scattered indices can cause page/chunk amplification and approach whole-union traffic unless storage ordering/chunking supports the crop. Index lists alone do not fix the current NPZ path. Perfect cross-pedestrian reuse could reduce physical reads, but is not assumed under shuffled complete-track updates and bounded RAM.

Storage optimization alone does not establish the30-minute gate: linearly scaling the final FP32 dense-track calibration rate to the estimated ROAD **training** point count gives about30 minutes of model execution before validation/loading. This is shape/path-dependent extrapolation, not a measured epoch or hard lower bound. Benchmark representative loading, transfer, train/readout and validation stages, including overlap, before claiming a complete epoch time. [Exact formulas, byte assumptions, source hashes and numerical scenarios](../../outputs/experiments/E002a/storage-layout-estimate-20261022/estimate.json).

### Output-pooling measurement — 2026-10-22

Authorized bounded diagnostic, not a production feature-policy change. Measure existing frozen5 cm encoder outputs with constant10/20 cm output grids and an adaptive option: retain original tokens inside any represented pedestrian cuboid expanded0.5 m on every side, pool remaining context at20 cm. Pooling averages features/XYZ on native-origin grids after full-sweep encoding. Keep20 m context, every track and all timesteps. No behavior fitting or encoder inference; production cache/index reads only.

The [command](../../README.md#e002a-feature-caches) writes a fresh diagnostic record with source hashes, per-frame/group counts, target-boundary mixing, feature distortion, actual FP32/FP16 NPZ bytes and three warm read/SHA/NPZ/cast timings plus separate crop gathering. It includes shared-frame arrays and uint32 crop indices; member counts/representative native references replace a full inverse mapping. Diagnostic files are not compatible with the production cache schema. The completed bounded measurement and sizing scenarios follow. Warm files and synthetic formats cannot establish cold SSD throughput, behavior accuracy or the30-minute epoch gate. Existing cache/training settings remain unchanged.

**Measured at `c29d557`:**36 frames, CPU only,68.1 s;233 selected source files retain their hashes. ROAD:20 group-balanced retained training frames/groups plus two largest retained stress frames (22 frames/21 groups total). No validation entries were available. LOKI:14 saved spatial frames/four training scenarios, including six random frames/two scenarios and eight stress frames; no validation fixtures. The rolling ROAD cache and small LOKI sample limit representativeness. No test access, encoder calls, production cache writes or training changes.

| Output pooling | ROAD tokens retained¹ | LOKI tokens retained² | ROAD full train+validation FP32 estimate | ROAD FP16-feature estimate |
|---|---:|---:|---:|---:|
| Original5 cm outputs |100%|100%|1,048 GB|592 GB|
| Constant10 cm |65%|66%|680 GB|384 GB|
| Constant20 cm |32%|37%|338 GB|191 GB|
| Original target-near tokens +20 cm context |34%|39%|354 GB|200 GB|

¹ROAD measurements/sizing use the20 group-balanced frames, excluding size stress cases. ²LOKI includes all14 frames and is less representative. Full-population sizes are **projections**, applying measured diagnostic byte ratios to the earlier shared-frame+uint32 reference, not an inventory or guarantee. Diagnostic member counts and metadata differ slightly from the earlier assumed format. The unpooled FP16 projection therefore also differs from the earlier580 GB arithmetic. LOKI's existing low-confidence unpooled sizing bracket projects to **182–409 GB FP32 /108–243 GB FP16** with adaptive pooling; this is not a confidence interval. Full cache preparation, validation coverage and final metadata/headroom must be checked before promising a fit.

**Detail loss:** constant10/20 cm retains about69%/34% of original in-cuboid token counts on ROAD, and68%/34% on LOKI. Adaptive retains100% on both measured samples and produces no target-interior/exterior pooling mixtures. Constant20 cm mixes outside points into groups containing3.9% of ROAD /6.8% of LOKI original in-box tokens. These are geometric checks, not pedestrian semantic or behavior scores. Pooled-versus-original feature relative L2 is about6% at10 cm and11% at20 cm/adaptive; FP16 storage adds about0.021% relative L2 without sampled overflow. Neither establishes behavior equivalence. Pooling also changes token multiplicity and attention weighting; averaging features cannot reproduce the original attention exactly.

**Warm diagnostic loading, FP32:** ROAD mean read/SHA/NPZ is14.9→4.8 ms/frame and all represented crop gathering/geometry12.4→3.8 ms/frame for original→adaptive; LOKI16.7→6.8 ms and32.5→12.4 ms. Files are OS-warm immediately after writing, with three repeats; no cold SSD test, production SQLite lookup/full-validator overhead, GPU transfer or model work. These timings do not establish an epoch speed or the30-minute gate.

**Recommendation, not a changed protocol:** prefer original target-near tokens plus20 cm context as the next candidate. It appears capable of fitting ROAD in450 GB even with FP32 while retaining the target's existing detail. Constant10 cm with FP16 is an alternative with a finer environment but lower target resolution. Keep FP16 as a separately validated storage choice. Any production adoption needs a new cache identity/root, full sizing/preparation, packing recalibration, and a bounded source-validation behavior comparison; existing original-token caches remain valid historical evidence.

[Raw measurement/protected hashes](../../outputs/experiments/E002a/output-pooling-20261022/measurement.json) · [byte-ratio sizing arithmetic](../../outputs/experiments/E002a/output-pooling-20261022/sizing.json) · [standalone report](../../outputs/experiments/E002a/output-pooling-20261022/report.md). Acceptance: full CPU163 tests pass (nine CUDA skips); the four pooling/serialization checks also pass in the actual cu126 extractor environment without GPU work. Hand-calculated checks cover means, negative/grid boundaries, original-token preservation, independent inverse reconstruction, target mixing and FP32/FP16 serialization. No CUDA path or production feature code changed.

### Target-specific two-resolution epoch estimate — 2026-10-22

**Proposed design, not implemented:** cache shared20 cm coarse tokens plus original5 cm tokens for occupied coarse cells containing an original token inside any represented cuboid expanded0.5 m on each side. Each target's precomputed uint32 index selects original tokens from its own marked cells and coarse tokens elsewhere within20 m; no coarse/fine duplicate cells or repeated resolution search during training. Other pedestrians retain coarse resolution unless inside that target's own refined cells. Keeping both tiers increases storage over the earlier354/200 GB estimate; that exact layout has not been sized.

**Analytical full training+source-validation forecast on aalto, not a measured epoch or gate acceptance.** Assumes a completed persistent source cache, no encoder misses, spatially ordered uncompressed indexed arrays with efficient selected-row reads/checks, FP16 or FP32 storage read into the unchanged FP32 model, full tracks and no pipeline-overlap credit. The exact two-tier loader/token counts are unmeasured; previous shared-adaptive crop fractions are proxies, not guarantees.

| Dataset/scenario | FP16 feature storage | FP32 feature storage |
|---|---:|---:|
| ROAD optimistic |18 min|21 min|
| ROAD planning |27 min|34 min|
| ROAD conservative |54 min|75 min|
| LOKI planning, weak point-count bracket |19–32 min|23–40 min|

ROAD planning uses3.40 billion selected point instances: **490 GB FP16 /926 GB FP32 selected reads per epoch**, despite a much smaller shared resident cache. Overlapping pedestrians revisit rows. At an assumed effective1 GB/s, reading contributes8.2/15.4 min; CPU gathering/relative geometry/FP16 conversion3.6/3.2 min; training model+transfer13.0 min; validation model+transfer1.4 min; geometry/RGB/remaining collation/packing/checkpoint/logging allowance1 min. Transfer is included in model calibration, not added twice.

Model arithmetic fits nonnegative point/frame/microbatch terms to three original-token ROAD development shapes from the enabled-branches-v2 resource calibration at5efb36c. It preserves per-frame overhead rather than scaling the entire epoch only by token reduction. CPU gather rates come from the previous warm pooling diagnostic. Validation/train model-cost ratio0.4 and effective read bandwidth1 GB/s are assumptions. Optimistic uses2 GB/s and lower token/model costs; conservative0.4 GB/s, more tokens and slower execution/validation. These are planning scenarios, not confidence intervals or formal worst-case bounds. LOKI's broader underlying point-count bracket and absent validation sample limit its estimate further. One-time encoding/preparation is excluded only once complete; no production cache or attempt was changed for this forecast.

**Decision:** indexed FP16 storage makes the30-minute target plausible, but neither cache fit nor this forecast passes the conservative gate. Indexing alone cannot make whole-frame NPZ reads efficient. Next decisive check is representative complete-track partial-read loading plus actual pooled model forward/backward and validation, with integrity checks and sustained cold/limited-RAM I/O; avoid rebuilding the full cache before that check. [Exact formulas, source hashes and scenario budgets](../../outputs/experiments/E002a/epoch-estimate-20261022/estimate.json) · [standalone report](../../outputs/experiments/E002a/epoch-estimate-20261022/report.md).

### Target-specific indexed prototype — 2026-10-22

Authorized bounded implementation, separate from production training/caches. Native-origin20 cm cells use floor(XYZ/.2); coarse features/coordinates are means. A target marks occupied cells containing any original encoder token within its native cuboid expanded0.5 m per side (existing inclusive cuboid tolerance). Cache both coarse tokens and original tokens for the union of marked cells. Each precomputed target selection uses fine tokens in its own marked cells and coarse tokens elsewhere, then applies the existing inclusive20 m horizontal cutoff to token coordinates. Whole-cell refinement extends slightly beyond the box margin; the other targets do not change a target's refinement decisions. No coarse/fine duplicate cells, GT/error filtering or new model inputs. FP16 features reload into FP32; coordinates stay FP32, native reference int64, cell IDs/member counts/selection rows uint32, fine-tier indicator uint8. Coarse native references identify a contributing return, not a physical point at the centroid; no full inverse map is retained.

The [command](../../README.md#e002a-feature-caches) writes immutable append-only per-scene binary column shards and publishes a content-pinned JSON manifest last. Features/XYZ are ordered by spatial cell, with offsets for each frame and target index list. Load only indices and selected feature/XYZ rows; derive relative XYZ/inside-box on selected rows.8192-row checksummed chunks are verified on first touch, with a512-entry LRU of verification records per reader; file size/mtime/ctime/inode changes fail even after a prior check. No whole-scene SHA or spatial search per request. Failed/unpublished roots are not reusable; no fallback, retry or production-cache adoption.

Acceptance: hand-calculated nearby targets/other-target independence, coarse/fine exclusivity, means/coordinates, rotated/negative/boundary/empty cases, FP16 overflow/nonfinite failure, multi-frame/fresh-reader equality, chunk verification/memoization, corrupt/truncated/mutated files and unpublished-manifest rejection. Native execution reuses the36 earlier frames for full-cohort sizing and six complete frozen development tracks for shuffled loading and real FP32 CE/backward/AdamW and validation-forward timings. Compare native cuboids/anchors against original fixture offsets/membership; preserve missing slots/full tracks. Warm files/fresh verification, warm memoized and advisory-cold fresh verification are separate; `posix_fadvise(DONTNEED)` affects prototype files only and is not guaranteed physical-cold I/O. The six-track unions contain the selected fixture participants, not the full cohort; bounded page reuse and compute cannot certify a full-population epoch.

**Completed at `c09aa5f`:**36 population frames plus six complete development tracks (311 slots/218 usable point requests),30 scene shards/0.377 GB binary payload;465 checked source files unchanged. Native cuboid/anchor offsets and membership match original fixtures. All target in-box returns survive refinement; no coarse/fine duplicate cells. Two successful attempts produce identical shard hashes. Existing model/trainer/production caches remain unchanged; the active checkout stays5efb36c.

| Sizing sample | Selected token fraction | Projected full train+validation FP16 payload |
|---|---:|---:|
| ROAD20 group-balanced training frames/groups |30.9%|208 GB|
| LOKI14 frames/four training scenarios, including stresses |36.8%|112–251 GB|

These projections include shared coarse/fine features/XYZ, representative native references, cell/tier/member metadata and uint32 per-target indices. Small JSON/chunk-index allowance is additional. ROAD excludes the two largest-retained stress frames from sizing; there is no validation sizing sample. LOKI's bracket remains low-confidence, not a confidence interval. Each source appears feasible separately; both together are not guaranteed within450 GB. This supersedes using354/200 GB from the earlier globally adaptive layout for the target-specific design.

**Measured loading in the final attempt:** LOKI's three complete tracks load in0.187 s warm-memoized /0.274 s advisory-cold with fresh verification; ROAD's three take0.182/0.281 s. These include selective integrity checks, row gathering, FP16 conversion, relative geometry/membership and CPU tensors. The full-cohort sweep check gives ROAD0.66/0.55 GB/s and LOKI0.67/0.59 GB/s logical selected-byte throughput for warm/advisory-cold. These are coupled CPU/loading rates, not physical SSD bandwidth; all bounded files fit in RAM, and global/cold-I/O pressure was not measured. The six-track tables contain selected fixture participants, while the36-frame check uses full-cohort unions.

**Real-model checks:** unchanged deterministic FP32 architecture, actual source GT CE/backward/clipping/AdamW, complete tracks, one/two-track physical batches and separate no-grad validation forward. Maximum tested allocation is3.01 GiB with two tracks. CPU169 tests pass (nine CUDA skips), CUDA169 pass, actual cu126 backend six indexed-storage and15 model tests pass. The full suites ran at09da04e; later probe-only changes add input hashes and the full-cohort loading pass, accepted by native execution. No backbone inference, W&B, held-out evaluation, comparison fitting or checkpoint selection.

**Updated forecast:** the final measured shape/load rates project ROAD32 min warm /40 min advisory-cold per training+full-validation epoch. The earlier successful run projects28/35 min: use roughly **30–40 min** for current ROAD planning, not a measured epoch or conservative upper bound. LOKI's weak count bracket gives24–35/28–42 min in the final run. Loading/CPU work and model/H2D each enter once; an additional1 min for remaining supply/packing/checkpoint/logging is assumed, with no overlap credit. No conservative30-minute gate acceptance or behavior-quality equivalence. Next: bounded asynchronous loading/GPU overlap and recalibrated spatial frame concurrency, then remeasure before full preparation; do not silently adopt the prototype as production supply.

[Final numerical/provenance record](../../outputs/experiments/E002a/indexed-prototype-20261022-03/prototype.json) · [sizing/loading/epoch arithmetic](../../outputs/experiments/E002a/indexed-prototype-20261022-03/analysis.json) · [report](../../outputs/experiments/E002a/indexed-prototype-20261022-03/report.md) · [acceptance/log hashes](../../outputs/experiments/E002a/indexed-prototype-20261022-03/acceptance.json). Retained attempts: `indexed-prototype-20261022` failed after preparation/loading because the temporary checkout lacked its standard output-root link; `-02` completed after that environment fix; `-03` adds full-cohort loading. Fresh roots, preserved evidence, no automatic retry/resume.

### Standalone indexed cache builder — 2026-10-22

User-authorized full preparation command, separate from training wiring: [usage](../../README.md#e002a-feature-caches), `scripts/build-e002a-point-cache.py`. One source per fresh root, all frozen eligible training/validation tracks including missing/unlabeled context; held-out archives are not read. Reuses pinned native readers, associations, transforms, initial ego anchors and full-sweep frozen D-DITR5 cm input preprocessing. Each requested sweep is encoded once across participants, directly into the accepted target-specific20 cm coarse/.5 m expanded-box fine-cell layout above; no old LRU and no RGB inference. Storage stays FP16 features/FP32 XYZ, with precomputed uint32 selections and FP32 model readout. Geometry/RGB preparations are unchanged.

Per-scene immutable shard manifests are content-pinned by a final source manifest. It retains population/encoder/backend/implementation hashes, source stat identities, complete track source keys/times/usable-slot masks, per-track shard group and frame references, stored sizes, encoder calls, elapsed time and failure evidence. Every binary-file SHA and target readback is checked after writing; the last frame additionally compares serialized features against fresh in-memory construction. Frozen collection/setup/archives/assets are rehashed before completion. Native annotation checks reuse the existing frozen provenance; raw sensor bytes use before/after scene stat identities, not new full-dataset checksums. `--min-free-gb` defaults20 and checks a full frame payload plus headroom before writing; it does not reserve the entire future cache. Source-by-source actual sizing must confirm both populations fit. All paths are fresh, no retry/resume, and a final durable `manifest.json` is published only on success. An interrupted root may retain completed scene shards but is not a complete source cache. `cache.json` retains progress/failure evidence. Bounded `--max-scenes`/`--max-frames-per-scene` checks explicitly publish `bounded-native-check`, never full-population acceptance.

Acceptance: synthetic checks cover one forward per shared sweep, integer native keys/string manifest references, missing-slot/empty-GT-independent retention, unavailable tracks, root protection, low-space refusal, file corruption, native-source mutation, missing boxes/frames, failed completion and explicit bounded scope. At `afbdb86`, CPU175 tests pass (nine CUDA skips), CUDA175 pass and actual cu126 extractor12 indexed tests pass. Native builds use two sweeps each: ROAD `train_00001`,38 target requests/8.90 MB,25 complete track records; LOKI `scenario_000`,four requests/4.67 MB,11 track records. Each source loads D-DITR once and calls it exactly twice, with zero RGB calls. Every binary SHA/target readback passes;5,496 ROAD and10,441 LOKI protected files remain unchanged. End-to-end checks took28.8/15.0 s including full catalog/asset verification and initialization; these tiny checks are not full-build or epoch-time forecasts. [Acceptance and pinned manifests/test logs](../../outputs/experiments/E002a/indexed-builder-20261022/acceptance.json) are retained locally and on aalto; native binaries stay on aalto. The isolated builder checkout is `/home/user20/projects/pedestrian-behavior-labeling-indexed-build`, with canonical shared outputs and explicit `PYTHONPATH` in the command. The primary checkout's unrelated edits/revision were preserved. No full-population build, training run or training integration was performed.

### Indexed training wiring — 2026-10-22

Training now consumes prepared inputs only. `PreparedRGB` opens the existing SQLite index read-only and verifies requested NPZ identities/checksums; missing entries fail without calling a producer. `PreparedPoints` requires a completed source manifest, frozen population/archive/slot identity and pinned D-DITR release/config/weights. It checks small scene manifests and binary lengths at startup, extracts all packing counts without reading feature payloads, then loads only selected rows through the existing chunk-verifying reader. Eight open scene readers maximum; eviction closes mmap handles, not persistent cache entries. FP16 features become FP32; relative XYZ/membership and missing-slot semantics remain unchanged. No new features, GT-dependent selection, encoder instances or old point-cache writes.

Full fitting rejects partial/in-progress caches. Bounded smoke may accept a partial root only when every selected track's usable slots are present. Run/checkpoint/prediction metadata pins the source manifest/protocol; final protection checks include source and scene manifests, not a wasteful full feature-store rehash each epoch. Geometry and RGB-only arms remain independent of point caches. Fitting settings/targets/splits/normalization and enabled-only fusion are unchanged. Paired summaries reject cross-source mixing of old unpooled and new pooled LiDAR protocols.

Calibration, smoke and the six-attempt launcher now accept indexed cache roots and retain native source-validation-only safeguards. New LiDAR profiles must carry the indexed point protocol and new reader code hashes. Calibration uses the existing six complete development tracks with their new pooled points, keeps78% allocated-VRAM headroom, and reserves zero VRAM for encoder misses. Existing unpooled profile/fixture reproduction remains separate. [Current commands](../../README.md#e002a-source-validation-pilot).

At `78c3ccd`, both local CPU and aalto cu118 CPU suites pass182 tests (nine CUDA skips); GPU is explicitly hidden on aalto to avoid competing with the active builder. At `31dde32`, actual cu126 CPU tests pass all15 training checks (two CUDA skips) plus seven prepared-input checks; the final startup-limit guard additionally passes the15 local training checks. Coverage includes read-only hits/no inference on misses, source/population/slot integrity, metadata-only packing costs, bounded scene-reader reuse, corruption/mutation failures, partial complete-track coverage, full-population claims even in smoke mode and pre-W&B missing-cache/oversized-frame failure. Shared-run tests train all three arms, reload checkpoints/reconstruct saved metrics, and preserve local evidence on W&B logging failure.

Read-only native checks use the already-completed two-sweep fixtures, not active full-build outputs:38 ROAD/241,988 tokens and four LOKI/51,064 tokens decode correctly;11 files per source remain unchanged. One fully covered LOKI track also matches its precomputed cost across complete-slot replay. Existing RGB reads hit for99 ROAD/28 LOKI slots, without encoder construction. [Acceptance/test logs/native counts](../../outputs/experiments/E002a/indexed-training-20261022/acceptance.json) are retained locally and on aalto. The separate training checkout is `/home/user20/projects/pedestrian-behavior-labeling-indexed-train`; builder checkout/revision/processes and primary unrelated edits are preserved. GPU resource calibration, actual W&B smoke and full-population throughput acceptance wait for completed source preparation. No full training launched; this is CPU/reader acceptance, not CUDA or epoch-throughput acceptance.

### Completed caches and GPU readiness — 2026-10-23

Both native builds at `d96e213` publish complete source train/validation manifests with zero changed protected inputs: ROAD52,265 sweeps/5,490 tracks/436 scenes,156.219 GB in8.29 h; LOKI31,290 sweeps/10,435 tracks/523 scenes,117.521 GB in4.79 h. Those are one-time preparation costs, not epoch times. The caches cover4,380/1,110 ROAD and8,574/1,861 LOKI training/validation tracks; maximum complete lengths100/102 slots.

At training revision `0384812`, all182 cu118 CUDA tests pass. The indexed cu126 calibration completes52 workloads without exceeding the predeclared ceiling: max64 tracks,665,856 temporal cells,1,460,024 physical-microbatch tokens,1,512,760 ordinary padded tokens, spatial concurrency2 and524,288 per-chunk padded-token limit. Both sources' sampled complete tracks and a524,288-token frame stress retain FP32 fitting and zero encoder residency. Four actual W&B source-validation smoke attempts (`indexed-integration-20261023`) complete first/repeated prepared reads, checkpoint/metric reconstruction and plot/protection audits. These are operational smoke results, not behavior comparisons.

Final timing check: `scripts/check-e002a-throughput.py` reuses the production assembly/packing/model/loss/AdamW operations with seed0, full source catalog and exactly one complete64-track logical update per case (or the actual population if smaller). Cases are seeded shuffle, highest total point counts and longest tracks, chosen from input metadata only, independently in training/validation. Each is measured with advisory eviction of only its own selected point shards and repeated OS-file reads; eviction is not a physical-cold guarantee. New random model/optimizer per case; validation uses no gradients, and all weights are discarded. No W&B or held-out scoring. Record exact selections, load counters, physical batches, seconds and peak allocation. A deliberately pessimistic forecast applies the maximum measured seconds/track to every training/validation track, plus an explicit assumed60 s routine epoch overhead. It is a stress envelope, not a measured epoch or proven worst-case bound. Small tests verify deterministic unique input-only selections and forecast arithmetic. Full training remains pending timing acceptance.

Timing at `3fd039a`, same calibrated model/loader: ROAD's shuffled64-track update takes30.89 s with advisory-cold point files /29.68 s with repeated files (30.83 million tokens,22 physical microbatches,4.20 GB peak CUDA allocation). The dense64-track stress takes90.09/85.50 s (121.40 million tokens,64 physical microbatches); longest64 takes68.54/65.96 s. The first shuffled64-track validation group takes28.13 s. Straight shuffled-rate extrapolation across the actual4,380 training/1,110 validation tracks gives43–45 min including the explicit assumed60 s overhead; this is an illustrative sample estimate, **not a measured full epoch**. It does not establish the under30-minute gate.

The representative training group selects4.44 GB of feature/geometry/index bytes but reads12.38 GB for chunk verification; repeated files still repeat those checks under the current per-reader bounded memo and eight-scene reader budget. RGB reads contribute only0.07–0.11 s/group. These counters identify a plausible input-supply cost; there is no measured loader-versus-GPU timing breakdown yet. Do not attribute all remaining time to disk bandwidth, checksum work or GPU computation from these counters alone.

The remaining timing cases were deliberately stopped after this evidence, preserving `KeyboardInterrupt`/failed operational status in [partial timing evidence](../../outputs/experiments/E002a/indexed-throughput-20261023-road-waymo-g-rgb-dditr/throughput.json); no computation/corruption failure is implied. Later arms/sources were not timed. [Readiness summary](../../outputs/experiments/E002a/indexed-readiness-20261023/readiness.json), [calibration](../../outputs/experiments/E002a/indexed-calibration-20261023/profile.json) and [actual W&B smoke audit](../../outputs/experiments/E002a/indexed-integration-20261023/integration.json) retain numerical evidence. CPU184 tests pass (nine CUDA skips); all182 pre-utility cu118 CUDA tests pass, plus two new throughput arithmetic/selection checks on CPU and the real GPU timing execution. Full comparison attempts remain unstarted. Next: isolate input loading/verification versus model work and improve throughput before claiming the runtime gate; the completed feature caches need no rebuild for loader/execution optimization.

### Load/model timing breakdown — 2026-10-23

At `f53f538`, the exact same seeded shuffled64-track ROAD training group is measured twice with lightweight timers around input iteration, point read/verification/gather, total point-crop preparation and RGB reads. Same calibrated packing,30.83 million tokens/22 physical microbatches, unchanged model/loss/optimizer and immutable feature caches. Weights are discarded; no W&B/test scoring. The outer input timer includes dataset assembly of each microbatch/collation; the remaining time includes H2D, forward/backward/loss/clipping/AdamW and host bookkeeping, **not just GPU kernel time**. GPU completion is synchronized at the update boundary; the unchanged training loop already synchronizes loss values before advancing the loader.

| Seconds per64-track update | Advisory-cold files | Repeated files |
|---|---:|---:|
| Input preparation |21.54|20.34|
| Model, transfer and optimizer remainder |8.62|8.69|
| Total |30.16|29.04|
| Within input: point reads + verification + row gathering |15.05|13.46|
| Within input: further point crop geometry/conversion/assembly |4.09|4.45|
| Within input: RGB reads |0.26|0.15|
| Within input: remaining metadata/geometry-dataset/collation work |2.14|2.28|

Input preparation accounts for70–71% of this update and point preparation dominates it. The point read/check/gather timer does not separate raw disk reads from checksum computation or NumPy row copying; do not label its whole duration as hashing or SSD time. The repeated pass remains slow. Next optimization should target point-input supply and potential loading/compute overlap, with the same cache content and integrity semantics; no cache rebuild is needed. This measurement changes no scientific settings and demonstrates neither optimized speed nor a full-epoch gate pass. [Numerical record](../../outputs/experiments/E002a/indexed-breakdown-20261023/throughput.json) · [protected-input check](../../outputs/experiments/E002a/indexed-breakdown-20261023/protected.json). The `--breakdown` command runs only these two updates; its selection/timer wrapper checks pass.

### Point-loader optimization investigation and plan — 2026-10-23

Read-only investigation at `f53f538`, using `python -m cProfile` around the existing `--breakdown` command on aalto. Same shuffled64 ROAD training tracks as above,30.83 million tokens/22 physical microbatches; two discarded updates, no W&B/held-out scoring. Profiled totals are30.09/26.86 s, of which input takes21.64/18.14 s and model/transfer/optimizer8.44/8.72 s. Variation from earlier timings is **not an implemented speedup**. Peak host RSS is5.01 GB; no CUDA OOM. The command-level profile includes startup/teardown, so the table uses caller edges to isolate recurring operations and averages the two updates. CPU function durations are attribution measurements, not asynchronous GPU kernel timings.

| Recurring operation | Mean seconds/update | Finding |
|---|---:|---|
| SHA-256 inside point `take` |5.28|Checksums are a substantial measured cost |
| Verification `read` calls inside `take` |4.91|Includes OS/page-cache/I/O wait and copying into Python bytes; not pure physical SSD time |
| Crop `astype` calls |2.01|Includes required FP16→FP32 conversion and avoidable copies; this whole cost cannot be removed |
| JSON parsing on scene-reader reopen |1.53|The eight-reader LRU repeatedly decodes manifests already checked at startup |
| Chunk discovery (`np.unique` internal hash) |1.05|Three calls per crop; feature/coordinate chunks coincide and selected rows are sorted |
| Crop concatenation (`np.c_`) |0.70|Builds a full result before another unconditional FP32 copy |
| Relative XYZ + box membership, combined |1.08|Required geometry, lower priority than the above |
| Model-side CPU `torch.isfinite` on point tensors |1.87|Separate from loader time; scans points already checked during crop construction |

These selected operations are not a complete additive wall-time decomposition. Full crop time includes conversion/geometry; model checks belong to the model remainder. [Profile summary/caller edges](../../outputs/experiments/E002a/indexed-loader-profile-20261023/summary.json), [raw cProfile](../../outputs/experiments/E002a/indexed-loader-profile-20261023/profile.pstats), [two-update timings](../../outputs/experiments/E002a/indexed-loader-profile-20261023/update/throughput.json) and [unchanged protected inputs](../../outputs/experiments/E002a/indexed-loader-profile-20261023/update/protected.json) are retained locally and on aalto.

**Verification reuse has different first-epoch and later-epoch effects.** Each update selects4.44 GB but verifies12.38 GB in28,807 chunks, identically in both passes: reader eviction loses verification state. The64 tracks cover48 scenes but cause60 reader opens. A read-only manifest/index replay finds27,308 unique chunks/11.54 GB: only0.84 GB (6.8%) of first-pass verification bytes is avoidable with complete reuse. Therefore persistent verification alone cannot remove most first-epoch hashing. It could avoid previously verified chunks in later visits/epochs, while selected features must still be read/gathered. Removing verification reads may also expose page faults previously hidden by those reads; subtracting the full read+hash cost is not a valid speed forecast. [Reuse calculation](../../outputs/experiments/E002a/indexed-loader-profile-20261023/verification-reuse.json).

Full ROAD verification state for the three consumed columns needs597,774 one-byte flags (0.60 MB), plus file identities/bookkeeping; this is not feature caching. Conversely all436 scene manifests contain308.6 MB encoded JSON (88.3 MB for these48 scenes); decoded Python objects can be much larger. Retain compact or byte-bounded metadata, rather than assuming unlimited decoded metadata is free. [Metadata sizing](../../outputs/experiments/E002a/indexed-loader-profile-20261023/metadata-sizing.json).

**Proposed implementation order; not implemented by this investigation:**

1. **Reduce work in each crop.** Allocate the final `[N,68]` FP32 buffer once, write converted feature/geometry columns into it and remove redundant advanced-index/`np.array`/final `astype` copies where ownership permits. Discover sorted row chunk boundaries once for features/coordinates instead of hashing the same row list twice. Reuse bounded, pinned scene metadata independently of the eight open mmap readers. Keep required coordinate semantics, finite checks and exact selected rows/order. Measure each change separately; the table gives current costs, not promised savings.
2. **Separate verification state from mmap lifetime.** Use source-bounded chunk flags keyed by pinned manifest/file identity, surviving reader close/reopen and epoch boundaries in the same process. Check the original file stamp on every use, including reopen; mutation/replacement must fail rather than inherit a verified flag. No persistent cross-process trust file or skipped first verification. Keep the reader bound. Test first-touch, repeat, reader eviction, corruption before first read and mutation/replacement after eviction. Measure fresh state separately from previously verified state.
3. **Overlap one CPU-prepared physical microbatch with GPU work.** Start with one bounded producer/queue; preserve logical-update membership, physical order, generator/dropout determinism and failure propagation. Give the producer exclusive ownership of its SQLite connection/readers; do not share the current thread-affine RGB connection blindly. Bound queued plus in-progress RAM and close/join on cancellation. This hides loading rather than removing it. For the original21.54 s input/8.62 s remainder, even perfect overlap has an optimistic21.54 s floor for that update. If this input rate applied across all5,490 training+validation tracks, it would mean30.8 min before routine epoch overhead, not a measured epoch/lower bound. Loader work must also shrink to establish30 minutes; actual host/GPU contention may limit overlap.
4. **Then inspect remaining model/transfer overhead.** Benchmark consolidated finite validation at the appropriate trust boundary without dropping nonfinite failures or weakening the public model interface. Consider pinned staging/asynchronous transfer only after measuring transfer wait and extra host memory; the8.6 s remainder is not all PCIe transfer. Leave architecture, precision, packing thresholds, dropout and token population unchanged.

Acceptance: compare supplied tensors/token counts/flags exactly on both sources and missing-input cases; preserve reader/manifest mutation failures and complete tracks. For overlap, check identical batch order and fixed-seed outputs/loss/gradients/updates. Run CPU/CUDA regression and regenerate resource calibration for any hashed execution-code changes. Benchmark fresh verification and retained verification, shuffled/dense/long groups on both datasets, with RAM, read/verified bytes and queue wait recorded. Finish with a complete source training+validation operational epoch including routine logging/checkpoint/plot work before claiming the under30-minute gate. No new feature cache or encoder inference is required. Leave RGB NPZ format, storage precision, new disk layouts and broad worker/GPU tuning out of the first increment.

### Cache inventory and retirement — 2026-10-22

User-authorized deletion of the **old** `E002a/point-cache/`:23,305 entries/449.97 GB encoded payload,450.26 GB allocated directory. Ownership lock was free and no training/preparation/prototype process was active. Removed the directory while holding its lock; retained its229 MB SQLite index as historical unpooled cost evidence under `E002a/cache-inventory-20261022/retired-point-index.sqlite3`. This snapshot is not a usable cache or a source of new pooled-token costs. Free space increased to518 GB. No other cache, dataset, model run/checkpoint or native input was deleted. [Removal evidence](../../outputs/experiments/E002a/cache-inventory-20261022/retirement.json) · [full allocated-size inventory](../../outputs/experiments/E002a/cache-inventory-20261022/inventory.json).

Canonical **aalto** output root: `/home/user20/projects/pedestrian-behavior-labeling/outputs/`. Review/viewer outputs and the older experiment checkouts' E001/E001b/E001c directories alias this same storage; they are not duplicate dataset caches. ThinkPad outputs are separate local copies (about5.8 GB total experiments), unchanged.

| Path relative to `outputs/experiments/` | On-disk size | Role in E002a training |
|---|---:|---|
| `E001/reader-preparation/` |0.486 GB|Keep: frozen tracks, labels, masks and native references; all arms |
| `E001/data-setup/` |0.015 GB|Keep: frozen populations/splits/source normalization; all arms |
| `E001b/bbox-native/` |0.116 GB|Keep: current geometry's native2D bbox extension; all arms |
| `E002a/rgb-cache/` |2.95 GB|Keep: complete3× DINO CLS for RGB-enabled arms;2.62 GB current payload/15,925 tracks, plus2,377 older namespace entries/0.254 GB and index/block overhead |
| `E002a/rgb-cache-preparation-03/cache.json` |small|Keep: full-population RGB preparation identity/gate |
| `E002a/geometry-revision-calibration/profile.json` |small|Current model execution profile; recalibrate when the indexed adapter/execution changes |
| Full target-specific indexed LiDAR cache |not built|Planned supply for LiDAR arm; production root TBD; prototype208 GB ROAD projection is not a prepared dataset |
| `E002a/indexed-prototype-20261022[-02,-03]/` |0.38 GB each|Bounded shared coarse/fine shards, correctness/performance evidence; not full training populations |
| `E001b/bbox/`, `E001c/box-cache/` |0.116/0.190 GB|Historical alternative preparations; E002a uses neither |
| `E002a/model-integration/points/` |7.63 GB|Historical bounded LiDAR acceptance cache; not current training supply |
| `E002a/cache-acceptance-data/`, `cache-acceptance-02-data/` |4.15 GB each|Historical acceptance caches; not current training supply |
| Other `cache-acceptance-*-data/` |about0.022 GB each|Small acceptance fixtures |
| `E002a/execution-probe/`, `execution-probe-02/`, `execution-probe-04/` |0.75/1.38/0.039 GB|Resource fixtures; `-04` references `-02` files, so directory names alone do not establish redundancy |
| `E002a/native-probe-06/`, `native-probe-07/` |1.32 GB each|Historical native fixtures; current RGB/extraction wiring still reads `-07/probe.json` and `loki-transform.json` |
| `E002a/point-pca-comparison-02/`, `-03/`, `linear-semantic-probe/`, DINO galleries |several GB combined|Diagnostic features/visualizations; not training supply |

`outputs/inspection/prediction-viewer/context/` adds0.374 GB cached RGB/BEV context; viewer indexes about0.007 GB. Neither feeds training. `E002a/weights/` (1.45 GB: DINO0.343, D-DITR0.555, deferred Utonia0.549 GB) and `extractor-env/` (9.04 GB) are model assets/environment, not dataset caches. Preserve experiment `runs/` separately: checkpoints, predictions and W&B/local evidence are results, not disposable feature caches.

Read-only machine audit on22 October, after retirement: Linux root has233 GB used/518 GB available; the separate native-data NTFS partition has only35 GiB available. Docker reports57.51 GB images (20.22 GB reclaimable),10.3 GB stopped-container layers and28.9 GB build cache; these accounting categories can overlap. All five containers are stopped, with reported exits16–22 months earlier; `speech_to_speech` alone has9.7 GB writable layers. Other candidates are pip downloads4.32 GB, Hugging Face Xet transfer cache3.13 GB and historical acceptance point caches listed above. The16.89 GB uv cache shares14.59 GB through hardlinks outside the cache; its inode audit estimates only2.29 GB reclaimable. Torch's0.343 GB DINO checkpoint is SHA256-identical to the retained project weight. Conda environments `pedestrian-behavior` (6.39 GB) and `zod-iac` (18.84 GB) are explicitly **retained by user instruction**. No additional cleanup was performed. Root-owned paths were not fully readable, so the filesystem-wide accounting is incomplete; Docker's report is verified independently, not a complete reconciliation of `df` and `du`.

**Current executable behavior:** geometry opens no RGB/point cache; geometry+RGB uses the existing RGB cache read-only. The [indexed training wiring](#indexed-training-wiring--2026-10-22) replaces the legacy full-arm supply; fitting no longer creates a point LRU or runs encoders. Standalone historical cache-preparation/acceptance commands still describe the old format; they are not indexed preparation commands. Existing prototype shards remain usable for loading/model work. Rebuilding the prototype from raw frozen outputs now requires re-extracting its retired ROAD source entries; historical protected hashes remain evidence, not a promise those deleted files still exist. No additional cleanup is implied by this inventory.

### Throughput gate — agreed 2026-10-22

**Before further full training, target strictly under30 minutes for a conservative worst-case complete training+source-validation epoch on aalto**, for both source populations and every pilot arm. Include loading/native decoding, cache verification/writes and eviction misses, transfer, forward/backward/optimizer, full validation and routine checkpoint/logging costs. Verify representative complete64-track updates, long/dense cases and cold as well as sustained LRU-pressure conditions; an all-hit toy example cannot pass this gate. This is an acceptance target, not an achieved result or universal wall-time guarantee. Build6050233 fails it.

One-time preparation may be budgeted separately if it is explicitly completed before fitting; report that cost and include every miss still possible during training in the epoch estimate. After cache/loader/backend/hardware changes, rerun timing and parity checks, record the tested revision/configuration and replace the current forecast with new evidence. Do not meet the target by silently dropping tracks/tokens or changing scientific settings. The gate is a documented execution prerequisite; no automatic runtime cutoff or launcher enforcement is implemented by this documentation update.

## Enabled-branch fusion revision — 2026-10-22

Implemented `enabled-branches-v2`: only enabled descriptors/flags, parameters and feature-cache ownership. Fusion inputs are65/194/323; temporal width stays256. Trainable parameter counts are **1,603,460 /1,734,916 /1,975,044** for `g`/`g-rgb`/`g-rgb-dditr`. Geometry keeps its six partial-validity masks and derived aggregate flag. RGB validity remains distinct from numerical bbox validity. Normalization, populations, targets, fitting settings, frozen feature identities and storage formats are unchanged.

At implementation revision `cc04f8169ea7777769adef61bebfbcea0722dcd7`, the CPU suite passes159 tests (nine CUDA skips), the CUDA suite passes all159, and all15 model tests pass in the actual cu126 extractor backend. Tests cover each fusion width/flag order, absent branch parameters, checkpoint reload, missing enabled descriptors/padding/future context, unequal accumulation, and ordinary/recomputed/offloaded gradients. Fixture integration covers all three arms with checkpoint/metric reconstruction, enabled-only observed strata, no disabled feature/cache/cost access and W&B-failure propagation through the existing fake logging boundary. Saved-prediction summaries accept separate old/new protocols and reject mixing them or mismatched common RGB availability. [Acceptance record/log references](../../outputs/experiments/E002a/enabled-branches-20261022/acceptance.json) are local; CUDA ran in a separate temporary worktree. The original actual-W&B/native smoke evidence above remains historical; no new native inference or actual W&B attempt was needed for this fusion revision.

No experiment attempt was launched, cache rebuilt or active GPU checkout changed for this revision; the latter remained at6050233 after verification. New resource calibration and the under30-minute full-epoch gate remain required; removing unused reads does not establish LiDAR-arm throughput or modality utility.

## Before implementation and execution

Scientific specification and bounded correctness/memory acceptance are complete. The population-scale throughput follow-up above must be addressed before further execution:

| Item | Status/evidence |
|---|---|
| Population/preparation | Complete full3× RGB cache and frozen GT/class/missing-input readback above; existing populations/GT unchanged |
| Model/trainer | Implemented; masks, equal-track accumulation, update warmup and source-validation selection accepted |
| Feature supply | Persistent RGB complete; lazy point supply/configurable450 GB LRU and persistent input-only cost references accepted |
| Resource calibration | Profile05 belongs to historical shared-availability-v1; new model hashes require recalibration. Conservative under30 min full-epoch gate remains unmet |
| Evidence/handoff | Local/cloud audit and manual commands accepted for bounded correctness; first full attempt exposed poor throughput in the2026-10-22 snapshot |

Acceptance before the six exploratory attempts:

1. Check65/194/323 fusion inputs, coordinate/feature pairing, missing-enabled-branch zeros, enabled-only flags/parameters and no disabled cache reads. GT must not affect spatial selection or enter model inputs; padding masks exclude nonexistent tokens without removing missing/unlabeled context. Verify full-track future context and RoPE positions through gaps.
2. With dropout disabled, compare unequal microbatch/final-partial aggregation, gradients and optimizer updates against a logical-batch reference. With dropout0.1 and fixed physical shapes, compare ordinary/recomputed outputs, gradients and updates at rtol1e-4/atol1e-5; preserve RNG/masks. Do not require identical dropout across different packing shapes.
3. Verify cache byte/LRU accounting, shared-frame reuse, capacity reduction, atomic writes, corruption/nonfinite failures, hit/miss feature equality and unchanged training RNG/backend settings. Preserve frozen source/encoder hashes.
4. Verify exact warmup/update counts, padding/unlabeled loss exclusion, nonfinite/empty-GT failures, full validation, F1/CE/earlier-epoch tie-breaking, patience/budget stops and selected-checkpoint predictions.
5. Run required CPU/CUDA suites and a clearly labeled bounded integration smoke with actual W&B logging, source validation only, checkpoint/plot/prediction reconstruction and logging-error propagation. Smoke is operational evidence, not a comparison result. Save resource/throughput evidence before the manual six-attempt launcher; no automatic full-run launch.

The earlier worst-case OOM and~1.15 TiB estimate remain historical warnings against assuming full preparation fits. Numeric ceilings and bounded20 m I/O are recorded in the integrated calibration above; full-population epoch cost remains unknown. Utonia remains deferred. Never silently cap tokens, truncate tracks, change precision or relax deterministic training. Preserve failures; fresh attempt outputs, no automatic retry/resume. Local checkpoint/prediction backup outside this laptop remains **TBD** and separate from the cache budget.

**Results:** no completed full pilot comparison attempts. Bounded behavior-training smoke attempts are operational evidence only; see acceptance above. Native/resource/semantic preparatory evidence is preserved below. Architectural novelty and modality utility remain hypotheses, not consequences of integrating these components.

## Preparatory work

Authorized **2026-10-19** as a bounded feasibility gate, including semantic BEV inspection. The following records preserve the executed probe protocols and evidence; their 15 m radius, benchmark dimensions and other historical settings do not override the current pilot specification above. No behavior training, held-out scoring, W&B runs or full-cache preparation. Existing E001-family archives, setup, normalization and results stay unchanged. This probe may complete partially if gated DINOv3 weights are unavailable; that state is explicit and cannot pass the RGB execution gate.

## Fixed scope and sample

Seed 0 selects eight training and two validation recording groups **per dataset**, then three native frames from each group's union of frozen selected frames. Add training-only crowded-group and longest-track context stress frames. Stress frames are excluded from random-sample extrapolation. Group selection and stress choices use identities, lengths and counts, never behavior classes or model errors. No test archives are opened. Reuse the accepted clip/scenario independence assumption.

Extract each scene once per selected timestamp and reuse features for all represented eligible pedestrians. Small pickle-free fixtures preserve coordinates, decoded features/logits, native indices/inverse mappings and separate identity references. Sampled source archives and frozen setup/collection manifests are hashed before/after. This is a bounded protection check, not a full filesystem audit.

## Encoder contracts

- **RGB:** frozen DINOv3 ViT-B/16 LVD-1689M; 768-channel normalized CLS. Centered square crop, side `1.5 × max(native bbox width, height)`, sampled to 256² with bicubic resampling. Mean-color out-of-image padding; never shift the crop. No native-box/image intersection means missing RGB. Log crop padding and intersection fractions. E001b native boxes only; no ROAD fallback.
- **Points:** frozen nuScenes D-DITR distilled + segmentation-fine-tuned checkpoint, strict state-dict loading, 64 decoded channels before the 16-class semantic head. Entire selected native scene scan, no temporal accumulation; Waymo retains all sensors/both returns and TOP ego-motion compensation.
- The author's [nuScenes loader](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/datasets/nuscenes.py) uses raw LiDAR XYZ and strength/255; the [Waymo preprocessing](https://github.com/RWTHVision/ditr/blob/f66d1dadb82e97ded7750567a043e4783305b01c/pointcept/datasets/preprocessing/waymo/preprocess_waymo.py) uses tanh(strength). Sampled LOKI PLY strengths are already [0,1] (2026-10-19 native audit); retain them and validate this bound on each probe frame. This aligns numeric ranges, not sensor reflectance physics.
- Our ego XYZ axes are forward/left/up. Rotate encoder input to nuScenes raw-LiDAR right/forward/up: `(-y,x,z)`; [official nuScenes KITTI conversion](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/kitti.py) documents the 90° difference. Do not invent a nuScenes roof-sensor translation. Different sensor origin/height remains a domain-shift limitation.
- Deterministic first native return per 5 cm voxel, author FNV ordering, exact inverse mapping; fixed serialization orders. These declared changes replace random validation sampling/order. No segmentation test-time augmentation or silent point truncation. Point extraction uses FP32 sparse/linear operations and the author's internal FP16 FlashAttention cast, matching its tester; eval autocast produces incompatible sparse value/weight dtypes. DINO uses CUDA FP16 autocast. Save float32 descriptors; test repeatability separately from deterministic FP32 trainable operations.

Point neighborhoods: **15 m horizontal radius in world-horizontal/fixed initial-heading axes**, no height cutoff. Tokens carry `[feature64, relativeXYZ3, inside_native_cuboid1]`. Native heading/full dimensions are used only for membership. A cuboid membership flag is geometric containment, not verified point ownership. No interpolation across missing native boxes/poses.

Pinned author revisions and hashes are enforced in [the adapter](../../src/pedestrian_behavior/inspection/pretrained.py). Weight: `nuscenes/semseg-ptv3_weight=distill-ptv3_nuscenes_dino-L/model/model_best.pth` at Hugging Face revision `a656b4385aa28060b30152676023903a923b4a17`; SHA256 `93dbb9d3fda247adf56e1c1ebd419d6ada4ebe4dc141110c606926d3ad67b30d`. Matching config SHA256 `7cce1838385d3f14830b5d61c88db98cbb62951cea39665311c61bca23f253e3`. [Pretraining/exposure qualifications](../../docs/literature/papers/ditr.md) remain applicable.

## Visual and numerical checks

Save raw-height BEV, semantic BEV and an oblique height-preserving point view with native target boxes and neighborhood circles. These are qualitative predictions from the existing nuScenes segmentation head, **not behavior outputs or measured segmentation accuracy**. Preserve a legend. Save full-image native-box/crop overlays beside the exact model crop, including naturally missing/partial observations. Inspect alignment, sparse/distant targets and gross semantic domain failures before interpreting features.

[CPU acceptance](../../tests/test_feature_probe.py): boundary/outside/missing crops, mean padding, rotated cuboids, fixed-heading transforms, point/strength mask correspondence, voxel inverse mapping, reordered feature-coordinate pairs, empty neighborhoods, finite gradients and development-only repeatable sampling. Native execution additionally checks finite outputs, 768/64 shapes, strict checkpoint loading and repeatability. Existing command behavior is covered by the full repository suite. Frozen extraction may use nondeterministic sparse kernels: recorded numerical repeatability is a measurement, not a cross-platform identity guarantee.

## Resource workload and estimates

Temporary benchmark graph only; no optimizer updates or trained behavior labeler. Shared unconditioned learned seeds, one spatial cross-attention block/four heads/FFN; internal widths 128/256 × one/four seeds; output256. Geometry64 + RGB256 + LiDAR256 + three flags =579 →256. Two bidirectional pre-norm temporal layers, width 256/four heads/FFN1024/dropout0.1, sinusoidal elapsed grid time, four-class readout. Profile FP32 forward/backward for whole-track microbatch shapes 1/2/4/8. Repeat the largest observed neighborhood at every slot of the longest eligible training track: this conservative shape test is **not measured training throughput or actual complete-track extraction**. Record OOMs without retry/truncation.

Estimate training/validation float32 payload and point-GPU extraction time per dataset using the random sample, scaling frame means by eligible scene-slot counts then averaging sampled group totals. Compare whole-scene sharing, per-frame neighborhood union sharing and duplicated pedestrian neighborhoods. Report RGB separately. Estimates exclude test preparation, index/inverse-map overhead, compression and native I/O; warm/cold kernel effects and small sample uncertainty limit precision. Cache representation, spatial width, any token limit and disk ceiling remain undecided until this report is reviewed.

## Environment and command

Official HF `model.safetensors` is also accepted: SHA256 `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` at model revision `5931719e67bbdb9737e363e781fb0c67687896bc`. Its parameter names are converted back using the [official Transformers mapping](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py), then strictly loaded into the pinned original backbone. Reconstruct only deterministic RoPE/bias-mask buffers and the masked zero key bias; learned weights must all be present. No alternate model implementation or Transformers dependency is introduced.

Use an isolated ignored `outputs/experiments/E002a/extractor-env`; never change E001 environments or add CUDA encoder dependencies to the main project. [Pinned direct dependencies](requirements.txt); save the fully resolved freeze/check output with execution evidence. CUDA 12.6 PyTorch wheels and matching prebuilt FlashAttention/torch-scatter avoid the host's older CUDA 11.7 compiler. Author PTv3 source is loaded without its eager imports of unrelated custom-op architectures; source files are not patched. The adapter verifies code revisions and weight/config hashes.

Code moves through Git; verify the remote revision. Native mount on aalto is `/run/media/user20/F47C60057C5FC152/projects`, not the old `/media` path. Native data remains read-only. Example from repository root (set paths to the pinned author clones/weights):

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/probe-e002a.py \
  --device cuda --output "$probe_root/native-probe" \
  --ditr-code "$probe_root/dependencies/ditr" \
  --ditr-config "$probe_root/weights/ditr-nuscenes-config.py" \
  --ditr-weights "$probe_root/weights/ditr-nuscenes.pth" \
  --dino-code "$probe_root/dependencies/dinov3" \
  --dino-weights /path/to/official-dinov3-vitb16.pth \
  --loki-root /run/media/user20/F47C60057C5FC152/projects/loki_data \
  --waymo-root /run/media/user20/F47C60057C5FC152/projects/waymo_v2
```

`--road-index` and `--outputs-root` override frozen-reference storage locations without rewriting manifests. Output directory must be fresh and under `outputs-root/experiments/E002a`. Omitting `--dino-weights` explicitly produces partial LiDAR/crop evidence; there is no alternate RGB checkpoint fallback. Results are local `probe.json`, `report.md`, `index.html`, `panels/`, `fixtures/` and `workloads.json`; failures retain traceback/status and propagate. No resume/retry framework.

## Execution evidence — 2026-10-19

The bounded LiDAR/crop attempt **native-probe-06** completed in 269 s on aalto at revision `f9ff228`, with explicit partial status because RGB weights were unavailable at execution. **75 native development frames**: 38 LOKI / 37 ROAD-Waymo, including separate training-only stress cases. First-frame repeated extraction was **identical (max error 0)** in both datasets. All **663 protected** sampled archives/setup/collection manifests retained their hashes. No behavior training, held-out inference, W&B runs or full caches.

| Measurement | LOKI | ROAD-Waymo |
|---|---:|---:|
| Native points/frame, median |231,936|174,021|
| Spatial tokens/frame, median |105,344|150,063|
| Encoder wall seconds/frame, median |0.196|0.232|
| Peak PyTorch allocated GiB during extraction |0.91|1.07|
| Represented eligible track/slots |474|418|
| Valid native RGB crops |156|120|
| Nonempty-neighborhood tokens, median / maximum |17,453 /66,991|15,434.5 /70,056|

Counts describe the bounded sample, not full-population coverage estimates. Some observed targets have no target-box returns; 29 LOKI /40 ROAD track/slots have zero target-box voxel support, including missing/unusable boxes. Preserve their context/other branches rather than filtering them out.

The inspected LOKI and ROAD panels show recognizable scene structure and plausible object/surface regions, with mixed road/other-flat/sidewalk/terrain predictions. This is a **qualitative observation on selected development images**, not verified segmentation correctness or evidence of behavior recoverability. Native image/crop overlays inspected so far match the declared square crop. Different sensor origin/height remains a domain-shift limitation; do not tune a coordinate translation against semantic appearance alone.

### Resource gate

All 16 conservative whole-track shapes (102 slots,70,056 tokens at every slot, widths128/256 × seeds1/4 × tracks1/2/4/8) reached CUDA OOM. This does **not** establish that every actual track fails. A separate **single-frame synthetic-zero workload at the measured maximum count** passed all 16 configurations with finite backward: one-frame/one-track peak about 0.30 GiB at width 128,0.50 GiB at 256; eight frames about 1.12/2.03 GiB. Its first-call timing includes warm-up and is not steady training throughput. The spatial adapter itself fits; retaining its activations across long tracks is the direct-graph bottleneck. Frame streaming/checkpointing or a declared token reduction need a decision, not silent truncation.

Float32 payload estimates from the random groups, **excluding test preparation, indexes/inverse-map overhead and native I/O**:

| Dataset / split | Full-scene GiB | Neighborhood-union GiB | Duplicated track-neighborhood GiB | RGB CLS GiB |
|---|---:|---:|---:|---:|
|LOKI training|821.78|362.82|1302.26|0.374|
|LOKI validation|85.88|36.49|84.59|0.026|
|ROAD training|1575.75|462.21|1490.07|0.299|
|ROAD validation|562.20|318.03|1313.70|0.161|

Even union sharing totals about **1,180 GiB / 1.15 TiB**; the laptop has about 511 GiB free after extraction environment setup. RGB CLS is approximately 0.86 GiB total and has no comparable storage concern. These small-sample group-scaled estimates can vary substantially; they are sufficient to reject assuming unbounded float32 point caches fit. Point-encoder wall-time extrapolation sums to about 5.9 hours; native I/O, transfer, indexes and caching can add time. Cache layout, budget, numeric precision and spatial token limits remain open; none was applied automatically.

### Artifacts and acceptance

Full evidence is under `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/native-probe-06/` on aalto. [Local report](../../outputs/experiments/E002a/native-probe-06/report.md), [panel gallery](../../outputs/experiments/E002a/native-probe-06/index.html), [numerical/provenance record](../../outputs/experiments/E002a/native-probe-06/probe.json), [whole-track workloads](../../outputs/experiments/E002a/native-probe-06/workloads.json) and `single-frame-workloads/` are ignored outputs. All 27 spatial fixtures, panels and numerical records have been collected locally; generated artifacts stay outside Git. Independent fixture checks reconstruct semantic counts exactly and verify finite float32 data, native selected-coordinate correspondence and pickle-free loading.

Earlier manual attempts are preserved alongside `execution-audit.json`: restricted NumPy checkpoint metadata (01), old sparse runtime NumPy SIGFPE (02/03), evaluation-autocast sparse dtype mismatch (04), and residual pooling-order shuffling (05). Fixed using specific safe scalar allowlisting, spconv-cu126 2.3.8 / cumm-cu126 0.7.11, author-style inference precision and disabling **all** serialization-order shuffling. No automatic retries or checkpoint substitutions.

CPU suite: 87 tests with six CUDA skips; CUDA suite: all 87; isolated extractor probe checks: all 9. The official conversion round-trip preserves all 188 full-backbone state tensors exactly (random initialized weights for this loading check, not pretrained inference). Main lock/dependency and isolated dependency checks pass. All 211 affected-owner links/anchors and `git diff --check` pass. [CPU](../../outputs/experiments/E002a/cpu-acceptance.log), [CUDA](../../outputs/experiments/E002a/cuda-acceptance.log), [isolated runtime](../../outputs/experiments/E002a/extractor-acceptance.log), [fixture acceptance](../../outputs/experiments/E002a/artifact-acceptance.json) and [independent conversion](../../outputs/experiments/E002a/dino-conversion-acceptance.json) retain the checks. DINOv3 access was granted after the point/crop probe. That historical attempt did not execute RGB; the [completion below](#rgb-checkpoint-completion--2026-10-19) records the subsequent original-checkpoint execution.

### RGB checkpoint completion — 2026-10-19

**native-probe-07** completes the fixed-sample probe using the original DINOv3 ViT-B/16 LVD-1689M `.pth`, with strict loading into the pinned original implementation. Matched the published `73cec8be` SHA prefix and pinned full SHA256 `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`; the file hash is unchanged after execution. Execution revision: `738d124edf41812e9ebb42030ecb5a9b857d8ac1`. The earlier partial attempt and failure evidence remain intact.

| Dataset | Native frames | Valid RGB crops / saved descriptors | RGB encoder wall seconds | Co-resident peak GiB | Incremental peak GiB | Repeat max error |
|---|---:|---:|---:|---:|---:|---:|
| LOKI |38|156|0.678|0.629|0.103|0|
| ROAD-Waymo |37|120|0.279|0.664|0.137|0|

All **276 descriptors** are finite float32 vectors of length 768, aligned with the exact native track/frame/crop identities in 60 pickle-free archives. Dataset-specific first-batch repeated extraction matches exactly on this hardware/runtime. Encoder times include input transfer and first-call overhead, exclude native I/O/crop preparation, and are not full-preparation throughput. Peaks are PyTorch allocated memory during RGB extraction with both encoders resident; the incremental column subtracts the pre-call live allocation. Point-extraction peaks in the new report also include resident DINO weights and should not be treated as an increase in point-model cost.

The sample matches **all 75 prior native frames**, with identical selected groups/archives and all 663 protected hashes unchanged. Both native point repeats also remain identical. Float32 cache payload estimates remain unchanged: RGB CLS approximately 0.86 GiB for both datasets' training/validation; point-neighborhood union approximately 1.15 TiB. All 16 conservative full-track readout shapes still reach OOM. Both encoder interfaces are now verified; this does **not** resolve the point-cache/activation budget or establish behavior performance. No training, held-out scoring, W&B run or full-cache preparation.

[Completed report](../../outputs/experiments/E002a/native-probe-07/report.md) · [Panels](../../outputs/experiments/E002a/native-probe-07/index.html) · [RGB acceptance](../../outputs/experiments/E002a/native-probe-07/rgb-acceptance.json) · [Provenance](../../outputs/experiments/E002a/native-probe-07/probe.json). RGB numerical fixtures and all panels are collected locally; the new duplicate spatial fixtures stay on aalto, while the prior 27 verified local spatial fixtures remain available in `native-probe-06`. Full new outputs remain at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/native-probe-07/`.

Validation after RGB measurement: CPU 87 tests (six CUDA skips), CUDA all 87, isolated extractor all 9; dependency/lock checks, owner links and `git diff --check` pass. These checks validate native extraction/correspondence and existing behavior, not annotation quality or cross-platform bitwise identity.

### DINO patch visual check — 2026-10-19

Authorized qualitative follow-up only; no behavior training or segmentation-head download. Reuse the completed probe's rendered development frames, displaying at most two valid crops plus one explicit missing example per frame. Native images/boxes must reproduce the earlier crop records. Preserve the original probe and frozen data; use a fresh output directory.

For each valid example, save exact float32 crop pixels, normalized CLS, normalized **16×16×768 patch tokens**, per-crop three-component PCA colors and cosine similarity to the patch at the centre of the image-visible native box. The seed is a geometric choice, not a verified pedestrian mask; it may contain background or occlusion. Fit PCA separately per crop, fix component signs and scale colors by its 1st/99th percentiles. Colors cannot be compared across crops. Maps are displayed at native patch resolution or enlarged overlays, never claimed as segmentation, pose or behavior predictions. Patch structure does not validate the CLS descriptor used by E002. [Official visualization examples](https://github.com/facebookresearch/dinov3#notebooks).

The original gallery selected the first three represented tracks regardless of crop validity. Three apparent zero-crop ROAD frames actually contain **3, 4 and 5** valid descriptors; `train_00028-1509232266190995` genuinely has zero. Its two native 3D tracks extend approximately 0.6/0.4 s beyond their last FRONT boxes; no ROAD pedestrian GT exists at that timestamp, and one identity has a native FRONT_RIGHT association. ROAD-origin identities do not imply FRONT boxes throughout their complete native context. The new gallery shows valid/represented counts and valid examples first. Prior numerical extraction is unaffected; original evidence remains intact.

From the isolated GPU checkout:

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/check-e002a-rgb.py \
  --probe "$probe_root/native-probe-07" \
  --output "$probe_root/dino-visual-check-new" --device cuda
```

The command reuses pinned checkpoint/native paths from `probe.json`; it does not decode LiDAR. Match the original crop order and batches of up to16 for FP16 comparison. Raw saved crop pixels retain bicubic overshoot; display clips to [0,1] without changing encoder inputs.

**Completed `dino-visual-check-03`** at revision `cff408200c9f87b0f05be1d737ac7d741fbfcb89` in **38.3 s** on aalto: **27 development frames, 45 valid patch examples and 20 explicit missing-crop examples**. All selected CLS descriptors match the previous extraction exactly (max absolute error0); the first repeated patch/CLS batch also matches exactly. All **725 protected files** remain unchanged: the prior 663 frozen references, probe record, original checkpoint and 60 RGB descriptor archives. [Gallery](../../outputs/experiments/E002a/dino-visual-check-03/index.html) · [numerical/provenance record](../../outputs/experiments/E002a/dino-visual-check-03/visual-check.json) · [independent fixture/link checks](../../outputs/experiments/E002a/dino-visual-check-03/acceptance.json). Panels have PNG/SVG exports; pickle-free fixtures retain exact inputs/features/maps/identities. Outputs are available locally and at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/dino-visual-check-03/`.

Purposive visual inspection shows distinguishable visual regions in some examples and broad context similarity in others. These views support extraction sanity only; they do not establish foreground segmentation, pose sensitivity, CLS behavior information or modality benefit. No segmentation head, behavior training, target-test access or W&B run.

The original [LiDAR-first gallery](../../outputs/experiments/E002a/native-probe-06/index.html) and [completed point/RGB gallery](../../outputs/experiments/E002a/native-probe-07/index.html) now also show the corrected crops alongside **visible original BEV panels**, with counts per frame and missing-crop context collapsed and labeled. Each keeps `index-before-crop-fix.html` as presentation evidence. Only HTML was refreshed; native/probe numerical files and encoder predictions are unchanged. The same presentation helper serves the DINO gallery; no inference rerun is needed.

### Crop-size visual comparison — 2026-10-19

Authorized diagnostic: compare centered square sides **1.5, 2 and 3 × max(native bbox width, height)** on the same 45 valid development examples. Every input remains 256×256, with unchanged bicubic sampling, mean padding, native boxes and no crop shifting. Larger crops add context while reducing target pixel size; no training setting or baseline cache is changed. Missing boxes stay missing. Show crops side by side, with padding fractions, and keep the three-scale DINO PCA/similarity comparison in an expandable panel. PCA colors are independently fitted per crop/scale, not matched semantic classes.

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/check-e002a-rgb.py \
  --probe "$probe_root/native-probe-07" \
  --output "$probe_root/dino-crop-scales" --device cuda --compare-crop-scales
```

The default command still uses only 1.5×. The optional comparison preserves the original 1.5× batch composition/checks and saves new 2×/3× float32 crops, CLS, patch features, maps and scale-specific crop/seed metadata in separate pickle-free fixtures.

Completed at revision `792c9050653b74305758d9bb3853a9f96ece1b50` in **97.7 s** on aalto: **45 pedestrians × three crop sizes =135 feature fixtures** across the same 27 development frames. The repeated baseline patch batch matches exactly; all 725 protected files are unchanged. [Comparison gallery](../../outputs/experiments/E002a/dino-crop-scales/index.html) · [record](../../outputs/experiments/E002a/dino-crop-scales/visual-check.json) · [independent acceptance](../../outputs/experiments/E002a/dino-crop-scales/acceptance.json). PNG/SVG comparisons and pickle-free arrays are available locally and under the same ignored output path on aalto. CPU93 (six CUDA skips), CUDA93 and isolated extractor15 checks pass. Larger crops visibly trade target detail for scene context; this does not identify the best behavior-classification input. No training or held-out access.

The original native06/native07 and DINO gallery pages also link these side-by-side crop panels and expandable feature comparisons. Their numerical/provenance records retain the original1.5× evidence; the2×/3× data live only in `dino-crop-scales/`. At this preparatory stage the working crop choice was1.5×; the later agreed E002a pilot uses3×, as defined in the [RGB branch](#rgb-branch). No old cache is reinterpreted as3×.

### Whole-track window videos — 2026-10-19

Authorized qualitative extension: **ten complete pedestrian tracks, five per dataset**, drawn only from the existing probe's development groups. Prefer tracks with the most paired native 2D/3D slots and distinct scenes, breaking ties by paired coverage and stable identity. This is purposive observation coverage, not a representative sample or behavior-based selection. Preserve every frozen 5 Hz slot through each full native track extent, including unlabeled/missing context; no interpolation, trimming or last-frame reuse.

Left: one centered **3× max-box-side RGB window**, native target box in white, thin 1.5×/2×/3× crop outlines in cyan/yellow/magenta. Display samples to640² for readability; model crops/defaults remain256²/1.5×. Right: the pinned frozen nuScenes D-DITR semantic head on each full native scan, then pedestrian-centered BEV of decoded 5 cm voxel points within **30 m horizontal radius**, no height cutoff. Draw **15/20/30 m radius circles**, the native target cuboid footprint and class legend. Use the same fixed initial ego-heading/world-horizontal coordinates as E002a; up is initial forward, screen-right is initial right. Scene inference uses the original axis/strength/pose contracts. The views are synchronized by saved native references and labeled grid/native times. Missing RGB or usable 3D observations produce explicit placeholders independently.

```bash
probe_root="$PWD/outputs/experiments/E002a"
uv run --no-project --python "$probe_root/extractor-env/bin/python" python scripts/video-e002a-tracks.py \
  --probe "$probe_root/native-probe-07" \
  --output "$probe_root/track-window-videos" --device cuda
```

Outputs: ten H.264 MP4s at 1440×900/5fps, first/middle/last PNGs, `index.html`, per-video encoder logs and `videos.json` with track selection, per-frame identities/times/crop bounds/support, checkpoint/revision references and failure evidence. Geometry tests use known nested squares and metric offsets/radius boundaries; missing observations never inherit another timestep. No DINO feature extraction, training, held-out access or segmentation accuracy claim.

### Frozen point-feature PCA — 2026-10-20

Authorized comparison of **frozen nuScenes D-DITR versus original Utonia** on the same ten development tracks above. Qualitative feature inspection only; no behavior training, segmentation GT scoring, held-out access, new full cache or backbone selection. [Command](../../README.md#e002a-pretrained-feature-feasibility).

- Utonia inference source `de1f9deef4ee65e508d74347f8d0eb848d878abb`; original `utonia.pth` SHA256 `a0a2e5e234ce943082ae9ba26bcff74e285f16ed8d04fed5c36ffe8dc30ea353`, verified against official HF LFS metadata. Frozen strict author loading, all serialization shuffling disabled. Official [outdoor demo](https://github.com/Pointcept/Utonia/blob/de1f9deef4ee65e508d74347f8d0eb848d878abb/demo/7_pca_outdoor.py) recipe: 0.2 coordinate scale, 0.01 scaled grid (5 cm physical), ego origin retained, no scene centering, absent color/normals zero. The [actual NormalizeColor](https://github.com/Pointcept/Utonia/blob/de1f9deef4ee65e508d74347f8d0eb848d878abb/utonia/transform.py) divides RGB by 255; it does not subtract one. No LiDAR RGB coloring or normal estimation. D-DITR's existing XYZ/strength recipe is unchanged.
- Utonia colors use its official outdoor visualization's last three hierarchy levels (216+432+576=1224 channels), inverse-mapped to native rows. D-DITR uses its 64 decoded channels before the semantic head. These feature levels, supervision and input channels differ; this is a practical released-representation comparison, not an isolated architecture/pretraining ablation. Different grid axes/rounding are permitted by author recipes; both displays use the exact same native XYZ indices through their respective inverse maps.
- One PCA per encoder, fitted once on **16,384 rows**: four existing probe training scenes per dataset, first two selected probe frames per scene, 1024 deterministic native returns/frame within 30 m world-horizontal distance of an eligible track. No behavior label or error-based selection. Identical sampled native indices for both encoders. Three leading components, deterministic signs, fixed training-sample 1st/99th percentile channel bounds; no channel whitening or per-frame scaling. Colors do not correspond between encoders. Pooled calibration uses both datasets and cannot support strict zero-shot evaluation.
- Full-scene encoding, then the same target-centered 30 m world-horizontal neighborhood for both panels; initial-ego-heading BEV, no height cut, 15/20/30 m rings, native target footprint. Display maps features to all native returns; it does not recover detail lost in grid/hierarchical reduction. Reuse the complete frozen 5 Hz grids and RGB windows, with independent missing observations and no temporal interpolation/accumulation. Utonia's published pretraining includes Waymo training/validation; [exposure qualifications](../../docs/literature/papers/utonia.md#verified-pretraining-partitions-2026-10-15) remain applicable.
- Fresh output: `index.html`, ten 2048×850/5fps MP4s, first/middle/last PNGs, `comparison.json`, pickle-free `pca.npz`, calibration features/native indices and per-frame point offsets/PCA colors. Original probe/video records, checkpoints, setup and frozen source evidence are checksum-protected. Record revisions, settings, dependency/backend versions, hardware, repeat errors, encoder wall time/peak allocated CUDA memory and failures. Errors propagate with local evidence retained; no automatic retry.

Acceptance: known PCA axes/variance and fixed out-of-range clipping; training-only balanced calibration; gather equivalence to the official full hierarchy upcast; missing-modality values and coordinate scaling; identical rendered point positions, metric rings and missing panels; failure propagation/local preservation. Real execution checks finite/repeated features on both datasets, exact original grid identities/times, all 865 video frames, pickle-free artifact correspondence, and unchanged protected sources; evidence follows below.

Attempt history is retained: `point-pca-comparison` failed on an unused SciPy augmentation import before extraction; the adapter now imports only the pinned inference modules. `point-pca-comparison-02` was manually stopped and marked **invalid-preprocessing** after discovering incorrect -1 missing-color inputs. Its Utonia colors/variance are excluded. The actual pinned author normalization/scaling class bodies were then executed independently ([acceptance](../../outputs/experiments/E002a/point-pca-author-preprocessing-acceptance.json)); missing RGB is zero, white RGB is one, outdoor XYZ scale is 0.2. Only the corrected fresh `point-pca-comparison-03` is eligible for review. No automatic retries, existing result edits or native reprocessing.

**Corrected execution completed** at revision `18ce056ad6e4303015e86340bf4336bd5b0536f4` in **962.1 s** on the RTX 4080. Ten videos / **865 frames**: LOKI369 (316 valid RGB crops), ROAD-Waymo496 (382 valid RGB crops). Both encoders' repeated calibration features match exactly on each dataset (max absolute error0). All **721 protected files** remain unchanged. PCA bases/bounds reconstructed exactly from the saved 16,384 calibration feature rows per encoder. The three displayed components explain **60.12% D-DITR / 22.14% Utonia** variance; unequal projection coverage and differing feature stages preclude ranking encoder quality from these colors.

| Dataset | D-DITR median extraction s/frame | Utonia median extraction s/frame | D-DITR peak allocated GiB | Utonia peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| LOKI | 0.190 | 0.496 | 1.37 | 3.65 |
| ROAD-Waymo | 0.225 | 0.518 | 1.47 | 3.06 |

Times include grid preparation, inference, gathering/projection and CPU transfer; exclude native I/O and artifact encoding. Peaks include both resident encoders. Author precision: FP32 sparse/linear operations, internal FlashAttention FP16 for D-DITR / BF16 for Utonia, no outer autocast. These are practical probe costs, not an isolated architecture benchmark. Utonia's global PCA color distribution visibly differs across these two datasets; neither that observation nor a sharper-looking D-DITR map establishes behavior transfer quality. No backbone or E002 training setting was changed.

[Corrected gallery / ten MP4s](../../outputs/experiments/E002a/point-pca-comparison-03/index.html) · [Execution/provenance](../../outputs/experiments/E002a/point-pca-comparison-03/comparison.json) · [PCA reconstruction](../../outputs/experiments/E002a/point-pca-comparison-03/pca-reconstruction-acceptance.json) · [Artifact acceptance](../../outputs/experiments/E002a/point-pca-comparison-03/acceptance.json). Copies live locally and on `aalto` under the isolated review checkout's `outputs/experiments/E002a/point-pca-comparison-03/`. Final CPU102 (six CUDA skips), CUDA102 and isolated22 checks pass; [CPU](../../outputs/experiments/E002a/point-pca-cpu-acceptance.log), [CUDA](../../outputs/experiments/E002a/point-pca-cuda-acceptance.log), [isolated](../../outputs/experiments/E002a/point-pca-extractor-acceptance.log). Original E001-family archives/results and existing visual evidence are retained unchanged.

Independent local acceptance verifies all865 encoded frame counts/rates, exact original native keys/times/crop validity, **863 aligned finite float32 point/color fixtures**, 30 PNGs, and all16 calibration fixture hashes. Two ROAD-Waymo timesteps lack usable target 3D context; their missing panels are retained independently of RGB. [LOKI contact sheet](../../outputs/experiments/E002a/point-pca-comparison-03/loki-contact.png) · [ROAD-Waymo contact sheet](../../outputs/experiments/E002a/point-pca-comparison-03/road-waymo-contact.png).

Completed at revision `ee3a32557fc181d2b7181366f68d9075593a3698` in **365.1 s** on aalto: **ten videos / 865 frames / 173 s playback**, with five distinct scenes per dataset. Per-frame native key/timestamp lists exactly match each complete frozen track; `ffprobe` verifies encoded frame counts, dimensions and 5fps. All **666 protected files** are unchanged. CPU96 (six CUDA skips), CUDA96 and isolated extractor16 checks pass; native/encoded-frame and gallery checks are retained with the outputs. Spot-checked middle frames on both datasets show readable nested windows/rings and the linked target boxes. Semantic predictions remain qualitative, including the existing sensor/domain limitations.

[Video gallery](../../outputs/experiments/E002a/track-window-videos/index.html) · [per-frame record](../../outputs/experiments/E002a/track-window-videos/videos.json) · [native-reference check](../../outputs/experiments/E002a/track-window-videos/native-reference-acceptance.json) · [video/link acceptance](../../outputs/experiments/E002a/track-window-videos/acceptance.json). Available locally and at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/track-window-videos/`; logs live beside that folder as `track-video-{cpu,cuda,extractor}-acceptance.log` (CPU log local).

Retained earlier manual attempts: `dino-visual-check` failed the CLS comparison because single-crop FP16 inference changed batch composition; `dino-visual-check-02` reproduced all LOKI examples but failed on a serialized native-root string passed to the ROAD reader. Matching original batches and restoring a `Path` fixed those failures; regression checks cover both. No automatic retry or overwritten evidence. Full CPU **91 tests / six CUDA skips**, CUDA **91**, and final isolated extractor **13** checks pass; numerical fixtures reconstruct cosine maps and all four questioned ROAD counts, and documentation links/`git diff --check` pass. Logs: `outputs/experiments/E002a/dino-visual-{cpu,cuda,extractor}-acceptance.log`.

### Frozen semantic-head transfer check — 2026-10-20

Authorized bounded follow-up: evaluate the existing frozen nuScenes D-DITR **backbone and segmentation head**, without fitting either. Native `waymo_v2/{training,validation}/lidar_segmentation/` contains 798/202 Parquet files; schema and sampled nonzero values verified. Exactly **seven of the existing 37 Waymo development frames, from five clips**, match semantic-label timestamps. Use all seven, selected before inference; no new scene search, held-out access, behavior training or full-cache preparation.

The [official schema](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/segmentation.py) stores `[instance_id, semantic_class]`, TOP only, with sparse annotated timestamps/points and both returns. Keep the existing full multi-sensor scan as encoder input. Align labels through the same laser/return order and positive-finite range mask as native decoding; score only TOP returns with supported GT, at all ranges. Preserve native pixel indices, inverse voxel association and raw classes. Check the native strength sequence independently against concatenated range images. No interpolation or voxel-majority replacement of native GT.

**Taxonomy frozen before inference:** nine coarse shared classes: vehicle (Waymo car/truck/bus/other_vehicle; nuScenes car/truck/bus/construction_vehicle/trailer), pedestrian, bicycle, motorcycle, cone, drivable (Waymo road+lane_marker), sidewalk, manmade (Waymo buildings/signs/lights/poles) and vegetation (including Waymo trunks). Native IDs follow [Waymo's enum](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/segmentation.proto) and the [pinned nuScenes config](https://huggingface.co/rwthvision/ditr/blob/a656b4385aa28060b30152676023903a923b4a17/nuscenes/semseg-ptv3_weight=distill-ptv3_nuscenes_dino-L/config.py). Exclude Waymo undefined, riders, curb, other_ground and walkable, whose categories do not map cleanly. Predict native 16-class argmax first, then map; nuScenes barrier/other_flat/terrain become an explicit unmatched prediction column and remain errors on scored GT. Excluded GT contributes no false positives. Category boundaries remain approximate; this is not official Waymo mIoU.

Report pooled native-return confusion, per-class IoU/precision/recall/F1 and point/frame/group support, accuracy, supported-GT-class macro-IoU, per-frame scores and excluded/unlabeled counts. Absent classes remain explicit; no fabricated uncertainty intervals from seven correlated frames. Save raw/row-normalized PNG/SVG confusion, seven GT/prediction/error BEVs, pickle-free reconstruction fixtures, mapping/provenance/protected hashes/status/failures, `report.md` and `index.html` in a fresh ignored attempt. BEV bounds are display-only. Existing 5 cm deterministic sampling/axis/intensity/precision contracts remain unchanged; no TTA or new classifier. [Acceptance tests](../../tests/test_semantic_check.py) cover range masks/channels, sensor/return ordering, unmatched-prediction penalties, hand-computed metrics, empty support and aligned BEV pixels. [Command](../../README.md#e002a-pretrained-feature-feasibility).

This diagnoses the released head on a small Waymo development sample. It cannot establish behavior usefulness, frozen-feature linear separability, or LOKI transfer. No model winner or later experiment setting is selected from these scores.

**Completed `semantic-transfer-02`** at revision `85d51f0cce48438221a56104fd022d2510ac12e8` in **71.3 s** on aalto (summed encoder calls **2.47 s**, remainder native reads/checks/artifacts). Seven frames from five clips: six source-training frames and one source-validation frame. **994,982 scored native returns**, **78.30% accuracy**, **40.40% supported-class macro-IoU** (8/9 classes supported). These are mapped development diagnostics, not official benchmark or behavior results.

| Class | GT points | Supporting frames / clips | IoU % | Precision % | Recall % |
|---|---:|---:|---:|---:|---:|
| Vehicle |109,345|7 /5|61.73|89.67|66.45|
| Pedestrian |29,654|7 /5|22.05|97.29|22.19|
| Drivable |179,183|7 /5|73.22|88.97|80.53|
| Sidewalk |62,629|7 /5|15.20|57.46|17.12|
| Manmade |351,752|7 /5|76.58|83.51|90.22|
| Vegetation |261,951|7 /5|74.46|83.93|86.83|
| Bicycle |359|1 /1|0.00|0.00|0.00|
| Cone |109|2 /2|0.00|0.00|0.00|
| Motorcycle |0|0 /0|0.00|0.00|unknown|

The unsupported motorcycle class has 121 false-positive predictions; its recall is undefined and it is excluded from supported-class macro-IoU. Rare bicycle/cone scores have little support. Pedestrian recall ranges from 0–42.9% across these frames; most missed pedestrian points map to manmade (13,503) or vegetation (7,882). Sidewalk often maps to unmatched native categories (38,273 points). Broad scene structure transfers on this small sample, but the existing head is not a reliable pedestrian/sidewalk segmenter here. This does not imply the frozen features cannot support a learned behavior readout; it also supplies no evidence for LOKI semantic performance. No preprocessing or class mapping was tuned after scoring.

[Full table/report](../../outputs/experiments/E002a/semantic-transfer-02/report.md) · [GT/prediction/error gallery](../../outputs/experiments/E002a/semantic-transfer-02/index.html) · [Numerical/provenance record](../../outputs/experiments/E002a/semantic-transfer-02/semantic-check.json) · [Independent acceptance](../../outputs/experiments/E002a/semantic-transfer-02/acceptance.json) · [Confusion PNG](../../outputs/experiments/E002a/semantic-transfer-02/confusion.png) / [SVG](../../outputs/experiments/E002a/semantic-transfer-02/confusion.svg). Available locally and at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/semantic-transfer-02/`.

All **671 protected files** retained their hashes. Independent acceptance checks every **1,165,732 native TOP label/pixel association**, exact metric reconstruction from pickle-free arrays, and identical coordinates/voxel mappings/predictions for all four frames with retained original spatial fixtures (the other three have no prior spatial fixture). Seven BEV panels and raw/row-normalized confusion exports were checked. CPU107 (six CUDA skips), CUDA107 and isolated semantic5 checks pass; [CPU](../../outputs/experiments/E002a/semantic-cpu-tests.log), [CUDA](../../outputs/experiments/E002a/semantic-cuda-tests.log). The first `semantic-transfer` attempt failed before inference because a validation check wrongly rejected legal -1 **instance IDs**; semantic IDs remained nonnegative. The fix validates the semantic channel separately and is regression-tested; the failed evidence is retained. No automatic retry or existing result changes.

### Frozen-feature linear semantic probe — 2026-10-20

Authorized after the existing head's low pedestrian/sidewalk recall. Train **only a fresh affine `Linear(64,9)`** over the unchanged frozen/eval D-DITR representation. This is a supervised Waymo semantic diagnostic, not behavior-model training or strict zero-shot segmentation; no LOKI supervision, held-out access, W&B or full-population preparation. Earlier inspected development images/scores informed this probe; validation is exploratory, not an independent final estimate.

**Frozen before training:** retain the original probe's eight random training/two random validation clips, excluding extra stress-only groups. All ten have 29–40 semantic timestamps. Select six timestamps/clip, evenly spaced through sorted available native semantic timestamps (`linspace` integer positions): **48 training /12 validation frames**, no clip overlap. These are native semantic timestamps, not a new behavior-grid cache. Training: up to8192 uniformly sampled GT-supported native TOP returns/frame, deterministic NumPy seed0, fixed sample across epochs, no class balancing. Validation: every GT-supported TOP return at all ranges; no sampling. Preserve both returns, full scene encoder input and prior 5 cm/axis/intensity contracts. Same nine-class mapping/exclusions as the unadapted check.

Normalize the sampled **training** features by per-channel mean/std only (std floor1e-6); no validation fitting. New head585 parameters, FP32, AdamW LR0.001, weight decay0.0001, default betas(.9,.999)/epsilon1e-8, unweighted point CE, batch8192 with final partial batch retained, clip1. Seed0 and deterministic algorithms for head fitting; encoder extraction retains the documented sparse-kernel precision/repeatability qualifications. Budget50 epochs, patience10 on strict validation supported-class macro-IoU improvements; exact score ties prefer lower validation CE, then earlier epoch. No scheduler, hyperparameter sweep, backbone update, automatic retry or resume. Save atomic head-only `best.pt`/`last.pt` using shared checkpoint operations and reload the selected head.

Compare on identical validation points: (1) original 16-class argmax mapped with unmatched-prediction errors; (2) **taxonomy-only control**, summing native class probabilities per common class and choosing among the nine common classes without fitting; (3) new learned nine-class head. The control prevents attributing gains from discarding terrain/other_flat/barrier or merging vehicles to head learning. This remains head adaptation with Waymo labels, not a matched pretraining/method benchmark.

Save source references, class support, selected identities/pixels, compact train/full-validation float32 feature fixtures, original/control/new predictions, before/after backbone-state hashes, head checkpoints, epoch history, per-frame/pooled metrics, GT/prediction/error panels, confusion PNG/SVG and CE/IoU curves in a fresh ignored directory. No changes to E001-family data/results or previous probes. [Command](../../README.md#e002a-pretrained-feature-feasibility); [tests](../../tests/test_semantic_probe.py) check group separation/even sampling, taxonomy-only probability merging, source-only normalization, final partial batches and checkpoint reload/prediction equality. Native acceptance must reconstruct scores, verify cached labels against native pixels, confirm unchanged backbone/protected files and disclose unsupported/rare classes. Results are unknown until execution; success supports accessible common semantics, not behavior utility or LOKI transfer.

**Completed `linear-semantic-probe`** at revision `03447b164c851c53cdf3ee66efe88e42148fa7a0` in **186.7 s** on the RTX 4080. **393,216 sampled training points /1,923,893 full validation points**, eight/two disjoint clips, 48/12 frames. Only the **585 head parameters** were optimized; the full released model's before/after extraction state hashes match. Selected epoch37, patience stop47. No failures or restarts. The following scores all use these same 12 validation frames; they must not be directly contrasted with the previous seven-frame pooled scores.

| Class / metric | Original head | Taxonomy-only control | Learned linear head |
|---|---:|---:|---:|
| Vehicle IoU % |52.55|54.02|61.03|
| Pedestrian IoU % |28.20|27.60|54.07|
| Drivable IoU % |57.91|69.42|73.10|
| Sidewalk IoU % |12.72|15.96|35.15|
| Manmade IoU % |73.19|72.99|78.30|
| Vegetation IoU % |51.22|50.50|60.49|
| Supported-class macro-IoU % |34.78|36.63|45.27|
| Accuracy % |70.95|75.70|81.24|

Learned pedestrian precision/recall **85.38% /59.59%**; sidewalk **58.42% /46.87%**. Gains occur in both validation clips: pedestrian IoU original→learned **28.60→54.38** (`train_00528`) and **17.12→46.43** (`train_00575`); sidewalk **16.78→43.21** and **4.04→24.43**. These are two already-inspected development clips, not an independent final test. The large-class gains beyond the taxonomy-only control support accessible common semantic information in the frozen features; domain/head mismatch contributes to the original failures. This does not establish behavior utility, LOKI transfer, or a need for backbone fine-tuning. No architecture/optimizer search followed the results. Any later representation choice based on this probe is Waymo-label-informed development, including for LOKI→Waymo claims.

**Quality qualification / nuScenes context:** sidewalk IoU **35.15% remains weak**, despite improvement. The [official model zoo](https://github.com/RWTHVision/ditr#model-zoo) reports **80.9% nuScenes validation mIoU** after full segmentation fine-tuning. The released experiment's [last logged training-validation pass](https://huggingface.co/rwthvision/ditr/blob/a656b4385aa28060b30152676023903a923b4a17/nuscenes/semseg-ptv3_weight%3Ddistill-ptv3_nuscenes_dino-L/train.log) reports **76.05% sidewalk /80.44% pedestrian /96.81% drivable IoU**, averaging **79.56%**; these are not verified class scores for the final80.9% evaluation. Different datasets, taxonomies, preprocessing and full-model versus small frozen-feature training prevent a direct performance comparison. Our probe establishes accessible semantic information, **not adequate segmentation quality or that a frozen backbone will suffice for the final behavior labeler**.

Rare-class limits: sampled training contains **4 bicycle /0 motorcycle /174 cone points**; validation **2,154 /0 /93**. Learned bicycle/cone IoU is0; motorcycle has no validation GT and is excluded from supported-class macro-IoU. These are not decisive tests of rare-class recoverability. Full/cached class supports and per-frame confusion remain in the record; source-only uniform sampling was not changed after observing this limitation. The diagnostic head is separate from the behavior inputs and does not replace the frozen pretrained representation.

[Report/full table](../../outputs/experiments/E002a/linear-semantic-probe/report.md) · [Gallery/36 panels](../../outputs/experiments/E002a/linear-semantic-probe/index.html) · [Provenance/history](../../outputs/experiments/E002a/linear-semantic-probe/linear-probe.json) · [Acceptance](../../outputs/experiments/E002a/linear-semantic-probe/acceptance.json) · [CE/IoU PNG](../../outputs/experiments/E002a/linear-semantic-probe/history.png) / [SVG](../../outputs/experiments/E002a/linear-semantic-probe/history.svg). Results/checkpoints/compact fixtures are collected locally and retained at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/linear-semantic-probe/` (about500 MiB, not a population cache).

Independent acceptance verifies all **2,317,109 cached GT/native-pixel associations**, complete validation support, source-only normalization, selected-checkpoint predictions, replayed selection/patience and all three reconstructed confusion matrices. All **676 protected files** retained their hashes. CPU111 (six CUDA skips), CUDA111 and isolated semantic9 checks pass; [CPU](../../outputs/experiments/E002a/linear-cpu-tests.log), [CUDA](../../outputs/experiments/E002a/linear-cuda-tests.log). No E001 results, native data, prior probe outputs, shared behavior-training code, dependencies or pretrained weights were changed.
