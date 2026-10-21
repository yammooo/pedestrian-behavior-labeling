# E002a — First multimodal behavior pilot

**Working specification, updated 2026-10-21.** First behavior pilot within [E002](../E002-multimodal-reference/README.md). The three D-DITR-first arms, branch interfaces and complete-track memory strategy below are agreed in direction; **working default** and **TBD** values remain provisional. No behavior model, pilot command, full feature cache or behavior-training run exists yet. Completed extraction/resource/semantic probes are [preparatory evidence](#preparatory-work), not separate behavior experiments. Later E002 comparisons may use different architectures.

The existing directory name is retained for stable commands, artifact references and links. This record owns the pilot specification, preparation, fitting settings and eventual behavior results.

## Question and interpretation

Do appearance and local 3D scene features improve offline pedestrian behavior labeling beyond geometry, including missing RGB and cross-dataset transfer? Use an established multimodal reference before considering a new retrieval mechanism. No modality benefit or methodological novelty is assumed; [literature positioning](../../docs/literature/offline-labeling-research-direction.md).

E001b geometry is a pragmatic reference, not a universal winner. Its BiLSTM track-weighted macro-F1, mean ± sample SD in percentage points: ROAD→ROAD **52.18 ±1.55**, ROAD→LOKI **45.57 ±1.10**, LOKI→LOKI **62.03 ±1.61**, LOKI→ROAD **47.53 ±0.91**. Other inputs win or tie elsewhere; [five-seed review](../E001b-2d-bbox-contribution/results-review.md). This representation choice followed inspection of both datasets' tests: disclose **test-informed development**.

## Next increment: behavior utility, not another segmentation gate

The [frozen-feature semantic probe](#frozen-feature-linear-semantic-probe--2026-10-20) reached pedestrian/sidewalk IoU54.1/35.2% on two Waymo development validation clips. This shows accessible semantics, not adequate segmentation or behavior usefulness. Our labeler receives pedestrian tracks/boxes; its point branch supplies additional target/scene evidence. A temporal model cannot restore information discarded by a frozen encoder, while spatial readout/fusion can also be bottlenecks. A negative behavior result cannot identify the encoder alone as the cause.

Prepare one bounded pilot on each source:

| Arm | Enabled descriptors |
|---|---|
| `g` | Geometry |
| `g-rgb` | Geometry + RGB |
| `g-rgb-dditr` | Geometry + RGB + D-DITR |

Every arm receives the same three branch-availability flags. Keep tracks, targets, geometry/RGB preprocessing, fusion/temporal widths and fitting protocol matched. **Utonia is deferred**, replacing the earlier four-arm proposal: first establish RGB/D-DITR behavior utility and a practical memory/storage strategy. Its feature specification and exposure/resource evidence below are retained for a separately agreed later comparison, not a fourth current arm. Frozen encoders are the initial pilot policy, not a final freezing decision. The Waymo-adapted semantic head is diagnostic only; neither its logits nor semantic GT enter the behavior inputs.

The earlier LR/query sweep is not the next increment. No further standalone segmentation optimization is planned before the behavior pilot.

## Retained data and evaluation contract

- Reuse E001/E001b's frozen eligible complete tracks, 5 Hz grids, four-state targets, native associations and group splits. Preserve 3D-only LOKI tracks, internal gaps and unlabeled context.
- Generate four logits at every non-padding timestep. Only accepted GT contributes to equal-track CE and metrics; retain E001 selection/aggregation semantics.
- Geometry normalization uses source-training statistics only. Frozen image transforms are fixed; no pooled cross-dataset feature fitting or target PCA/compression.
- Source training/validation only in the pilot. Train separate models per source; each source's validation selects its own checkpoints/settings. Pilot group/track budgets are **TBD**; no held-out inference.
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

Working MLP: `Linear(20,64) → GELU → Linear(64,64) → GELU`, output `[B,T,64]`. Internal flags preserve partial missingness. Derive `geometry_valid = any(six flags)` and zero the final descriptor if all groups are missing. Append this derived flag at fusion alongside RGB/LiDAR availability; retain the six internal flags for partial missingness.

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

**Agreed 2026-10-21:** cache the frozen RGB CLS at **3×** crop scale, separately from earlier1.5× probe outputs. Store float32 `[T,768]`, explicit validity, track/native-time identities, crop/transform settings and checkpoint provenance; zero missing rows without filtering tracks. Never reuse a1.5× CLS cache as3×. Full pilot population/cache scope remains to be frozen.

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

Working per-point input and projection:

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
A = softmax(Q K^T / sqrt(D/4) + padding_mask)  # [4,M,N]
U = S + ConcatHeads(A V) W_O
Z = U + FFN(LN(U))          # token-wise; agreed FFN 128→512→128, GELU
LiDAR descriptor = Z[0]     # [128]; no extra output projection
```

This is cross-attention, not prepended-token unrestricted self-attention: no `N × N` attention in this readout. Spatial dropout is **TBD**. Geometry enters through point coordinates/membership; no learned semantic class predictions are required. The single seed output forms one128-channel descriptor/frame; it is not an extra temporal timestep. The spatial summary is made before temporal reasoning; future frames do not change its point selection. [PMA/Set Transformer precedent](https://proceedings.mlr.press/v97/lee19d.html).

Multiple seeds remain a possible later control, not a first-pilot setting. With the same128-channel width, four seeds would produce `[4,128]` instead of `[1,128]`: more summary slots, not the same total descriptor size. Different seeds could preserve different spatial evidence, but specialization is not guaranteed and seeds may become redundant. Cross-attention score/value aggregation scales with seed count; point encoding/projection is shared. Flattening would require a larger fusion input; compressing back to128 reinstates an output bottleneck, though aggregation is more expressive. Passing spatial tokens separately through time would change the temporal architecture. No extra seeds can recover information absent from frozen encoder features or discarded point sampling.

### Spatial readout objection and revised proposal (2026-10-16)

The initial pooling discussion motivated retaining coordinates until learned aggregation. Shared unconditioned seeds were chosen on19 October; fixed mean/max remains a possible control, not the default. DINO CLS is also learned compression. Full-track-conditioned retrieval remains a conditional later hypothesis; none of these summaries is assumed lossless or novel.

## Missing inputs and padding

| Branch | Validity rule | Invalid output |
|---|---|---|
| Geometry | Any existing feature-group flag true; partial groups retain their masks | Zero final64-channel descriptor if all groups missing |
| RGB | `rgb_valid`: usable native image/box crop | Zero final128-channel descriptor |
| LiDAR | `lidar_valid`: usable20 m crop with at least one retained spatial token | Skip attention; zero final128-channel descriptor |

Append **`geometry_valid`, `rgb_valid` and `lidar_valid`** as three per-frame scalar feature channels. The geometry flag is the deterministic OR of the six existing geometry flags, not an additional observation. Their meanings/site policy must be encoder-independent and identical across arms. Disabling a branch zeros its descriptor but retains measured observation flags. No extra target/context availability bits enter the model; retain those counts as diagnostics. A crop with context but no inside-box points can still be valid LiDAR.

Gate **after** all branch projections: learned seeds, residuals and biases otherwise produce nonzero outputs for missing input. Never invoke attention with every spatial key masked. Genuine missing observations are retained; corrupt/nonfinite inputs and missing required decoding metadata fail visibly.

Spatial padding masks cover nonexistent point tokens; temporal padding masks cover nonexistent track timesteps. Missing modalities and unlabeled slots do **not** remove time positions. GT masks affect loss/metrics only.

## Fusion and temporal head

```text
geometry [64] + RGB [128] + LiDAR [128] + availability [3]
→ concat [323] → trainable Linear(323,256)
→ fused tokens [B,T,256]; no additive positional encoding
→ two full-track bidirectional pre-norm Transformer layers, with temporal RoPE on Q/K
→ Linear(256,4) → logits [B,T,4]
```

Agreed widths: geometry64, RGB128, LiDAR128, fusion input323 and temporal256. Temporal working settings: two layers, four heads, FFN1024, dropout0.1; full noncausal attention within each track, never across tracks. Mask padding and retain missing/unlabeled context. Keep inactive descriptor slots zero so fusion/temporal shapes match every arm. `g` is geometry plus shared availability controls with a new Transformer, not a pure geometry or historical BiLSTM score.

**Temporal positional encoding, agreed 2026-10-21:** use [RoPE](https://arxiv.org/abs/2104.09864) on Q/K in each temporal self-attention layer, not on V; no added sinusoidal/learned position vectors. Keep full bidirectional attention. Working implementation convention: rotate all64 channels/head with frequency base10,000 and position `(nominal_grid_time − track_start_time) /0.2`, equivalent to slot index at5 Hz. Preserve positions through missing modalities/GT; mask temporal padding. No track-length normalization or absolute recording timestamps. This replaces the earlier additive sinusoidal proposal; no behavior advantage is yet measured.

Initially frozen: DINOv3 and the selected point encoder, in evaluation mode. Trainable: geometry MLP, RGB projection, point normalization/projection, spatial seed/attention/FFN, fusion, temporal layers and classifier. Trainable operations use deterministic FP32; frozen extraction precision is separately recorded. Only behavior supervision trains these modules; source semantic-head probe weights are not reused.

## Access and comparison safeguards

The deferred Utonia alternative is a **Waymo-exposed reference, not an upper bound**. Its [pretraining includes Waymo training/validation](../../docs/literature/papers/utonia.md#verified-pretraining-partitions-2026-10-15), potentially overlapping E001 source-test recordings; exact intersections are unmeasured. Source-only fitting does not establish dataset-excluded LOKI→Waymo transfer or unseen Waymo recordings. A shared encoder choice informed by both sources' development scores must be disclosed. LOKI behavior utility remains untested for both encoders.

D-DITR is LiDAR-only at inference but inherits DINOv2 image distillation and nuScenes semantic fine-tuning; [lineage](../../docs/literature/papers/ditr.md). Different DINO teachers do not imply aligned RGB/LiDAR features or complementarity. Waymo semantic-probe decisions are additional label-informed development. Freezing does not erase pretraining access; retain [shared evaluation safeguards](../README.md#strict-zero-shot-access).

Later modality ablations remain deferred: geometry+availability, +RGB, +LiDAR without RGB, +both. With five seeds/two sources/four variants they would require40 attempts/80 test cells, separately authorized after pilot settings freeze. No held-out tests in the current pilot.

## Deferred training pilot

One seed0 attempt/arm/source gives **six planned exploratory attempts** across three arms/two sources; no execution command yet. These diagnose effects, not reliable winners. Repeat central comparisons before final claims. Do not multiply the pilot into a backbone/query/LR search.

Working fitting defaults, to ratify before execution:

| Setting | Value |
|---|---|
| Loss | Equal-track CE over accepted GT; no unlabeled/padding contribution |
| Optimizer | AdamW, LR **TBD**, betas(.9,.999), epsilon1e-8, weight decay1e-4 |
| Updates | FP32, global gradient clip1; effective batch64 complete tracks via correctly weighted accumulation; retain final partial batch |
| Schedule/budget | Three-epoch warmup then constant LR; maximum100 epochs, patience15 |
| Selection | Highest source-validation track-weighted macro-F1; exact ties lower CE, then earlier epoch; patience resets only for strict F1 improvement |
| Augmentation | Natural missingness and ordinary network dropout; no modality/crop augmentation initially |

Earlier LR candidates1e-4/3e-4/1e-3 and one/four spatial seeds are historical development options, not a required sweep; the first pilot now fixes one spatial seed. Reuse E001 loss/checkpoint/evaluation operations where applicable; do not change existing data/results or shared scientific semantics.

## Complete-track memory and batching policy — agreed 2026-10-21

Keep every selected complete track and every encoder voxel token in its crop. Do not drop long/dense tracks, split them into independent training windows, truncate temporal attention or change inference context to meet a memory limit. Numerical VRAM ceilings and frame concurrency must follow measurement, not the analytical estimates below.

- **Frozen feature supply:** encode scenes without encoder gradients, share scene outputs across represented pedestrians where practical, and discard frozen internal activations. Cached outputs allow encoders to be unloaded during behavior training. On-demand extraction from native data is an alternative, not rejected; it repeats decoding/encoding on cache misses and adds encoder residency/workspaces to the VRAM budget. RGB CLS caching is agreed at3×; full disk cache, bounded pilot cache or on-demand extraction for **D-DITR** remain a measured choice, not a settled requirement. Freeze extraction semantics and verify equivalent feature/identity pairing for either path.
- **Two batching levels:** whole-track microbatches vary with estimated point cost `sum(N_bt)` and temporal padding/attention cost. Spatial frame chunks control the number of point sets resident at once; they do not split the temporal sequence. Shuffle/bucket deterministically from seed0 using lengths/token counts, never behavior labels or model errors. Each selected track appears once/epoch, including expensive tracks; grouping and calibrated ceilings are fixed before a run and recorded.
- **Gradients, conditional memory paths:** ordinary execution is the default for workloads that fit. Use spatial activation recomputation and/or CPU-input staging/offloading only for workloads that require them, according to preflight-calibrated, recorded shape/memory thresholds; no OOM-triggered retry. Retain the small descriptors for full-track temporal attention. Recompute only the trainable readout from frozen features during backward, not the frozen encoder. Preserve RNG behavior and delay updates until all contributions are accumulated; do not detach the readout from behavior supervision. Ordinary GPU-input checkpointing alone is insufficient. Exact implementation details must pass gradient equivalence before execution; [checkpointing](https://docs.pytorch.org/docs/2.7/checkpoint.html) and [saved-tensor offloading](https://docs.pytorch.org/docs/2.7/autograd.html#torch.autograd.graph.save_on_cpu).
- **Effective batch:** accumulate summed individual track losses and divide by actual tracks per optimizer update (working target64 complete tracks; retain the final partial update). Do not average microbatch means equally when microbatch sizes differ. Use matched track order/update membership across arms; physical microbatches may differ. Partial-GT masks and E001 equal-track semantics remain unchanged.
- **Preflight:** measure the agreed20 m block on short, typical and long/dense development tracks in both datasets, including naturally missing branches. Time ordinary execution, recomputation and offloading on fitting fixtures, choosing the least overhead path that meets the budget; verify outputs, parameter gradients and one optimizer update within declared FP32 tolerances, including unequal accumulation and temporal padding. Record peak allocated/reserved VRAM, host RAM, transfer/I/O, tracks and point tokens per second, and timings after warm-up. Derive safe microbatch and frame-chunk ceilings from these measurements. If the largest selected example cannot fit, fail visibly and revisit the agreed execution strategy; no OOM-driven retry, silent cap or dropped track.

This settles the execution direction, not its numeric resource limits. Disk feasibility is a separate gate: recomputation saves activations, not the stored frozen-feature payload. Feature-supply strategy and any pilot-cache scope/sharing/disk budget must be fixed before full preparation. On-demand timings must include native reads/decoding, transfer, scene reuse and frozen encoding; a small forward-only encoder timing is not end-to-end epoch throughput. No additional sampling or precision reduction is authorized by this policy.

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

Manual attempts remain: `execution-probe` failed on integer timestamp filename concatenation; `execution-probe-02` completed native extraction but was manually interrupted before unguarded offload stress, with interruption also preventing its final evidence write; `execution-probe-03` failed the GPU optimizer-update equivalence check with default biased spatial attention. The final graph uses the specified bias-free spatial projections; equivalence passes without relaxed tolerances. Attempt04 independently hashes/verifies the reused fixtures and original sources. Original1.5×/15 m probes and all E001-family data/results are unchanged.

[Local report](../../outputs/experiments/E002a/execution-probe-04/report.md), [numerical evidence](../../outputs/experiments/E002a/execution-probe-04/execution.json), [artifact audit](../../outputs/experiments/E002a/execution-probe-04/artifact-audit.json), [nonempty compression audit](../../outputs/experiments/E002a/execution-probe-04/compression-audit.json), [selective on-demand audit](../../outputs/experiments/E002a/execution-probe-04/selective-on-demand-audit.json), [GPU system samples](../../outputs/experiments/E002a/execution-probe-04/system.csv) and [validation/failure logs](../../outputs/experiments/E002a/execution-validation/) are collected locally.3× RGB fixtures are also local in `outputs/experiments/E002a/execution-probe-02/rgb/`. Full point fixtures remain on aalto at `/home/user20/projects/pedestrian-behavior-labeling-review/outputs/experiments/E002a/execution-probe-02/`; final resource evidence is in sibling `execution-probe-04/`.

## Before implementation and execution

| Open item | Required definition |
|---|---|
| Pilot population | Group/track budget per source; fixed complete tracks/GT across arms; class/missing-input support; validation coverage |
| Resource calibration | Measure20 m all-voxel costs and throughput; fix track-memory and concurrent-frame ceilings, padding/bucketing details and host-RAM limit |
| Readout | Choose spatial dropout; D=128, M=1, feature LayerNorm and spatial GELU FFN128→512→128 are settled |
| Feature supply | Compare native on-demand extraction with bounded/shared caches; include encoder residency, native I/O, scene reuse, epoch throughput, host RAM and disk ceiling; stored descriptors remain FP32 unless a separate change is agreed |
| Fitting | Ratify one LR and schedule/budget/effective batch; no hidden optimization differences |
| Evidence/next decision | Primary/per-class behavior metrics, missing-RGB/point-support strata, paired track changes, parameters/runtime/resources; criteria for repeat seeds or adaptation |

The earlier preparatory direct worst-case workload OOM and~1.15TiB D-DITR float32 union estimate prohibit assuming full preparation fits. Measure20 m D-DITR costs; Utonia costs remain a deferred gate; do not silently cap tokens, truncate tracks or reduce precision. Validate coordinate/feature pairing, missing-branch zeros, spatial/temporal/GT masks, full-track gradients/aggregation, checkpoint reload and existing-pipeline compatibility before execution. Preserve failures; fresh output directories, no automatic retry/resume.

**Results:** no behavior runs. Native/resource/semantic evidence is preserved below. Architectural novelty and modality utility remain hypotheses, not consequences of integrating these components.

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

The original native06/native07 and DINO gallery pages also link these side-by-side crop panels and expandable feature comparisons. Their numerical/provenance records retain the original 1.5× evidence; the 2×/3× data live only in `dino-crop-scales/`. The E002 working crop choice remains 1.5× pending a behavior-based comparison.

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
