# Code architecture and data flow

Agreed 2026-10-08. This owns code responsibilities and implementation flow. [Dataset notes](docs/datasets/README.md#reader-and-prepared-track-schema) own saved/native schemas; [E001](experiments/E001-kinematic-transfer/README.md) owns scientific settings and acceptance requirements; [README](README.md) owns commands.

## Package layout

```text
src/pedestrian_behavior/
  datasets/                 Native readers, identities, units and transforms
  data/
    preparation.py          Native observations → saved tracks
    features.py             Physical feature construction and normalization
    loading.py              Saved-track Dataset and batch collation
    splits.py               Grouped splitting and split validation
  models/
    kinematic.py            MLP/BiLSTM and their shared encoder/classifier
  training.py               Training loop and checkpoint handling
  evaluation.py             Loss, metrics and condition analysis
  experiments/
    e001.py                 E001 targets, eligibility, models and condition policy
    run.py                  Shared attempt lifecycle, artifacts, plots and W&B
    sweep.py                Shared bounded child-process scheduling
    e001_run.py             E001 data/settings assembly and command
    e001_sweep.py           E001 seed matrix and command
    e001b_run.py            Bbox data/settings assembly and verified-bbox strata
    e001b_sweep.py          E001b BiLSTM-only seed matrix and command
  inspection/               Human inspection tools
```

Native readers, preparation, E001 setup, features, splitting, normalization, batch loading and kinematic models are implemented. Training, evaluation, checkpoints and local/W&B evidence are implemented. The [saved-track format](docs/datasets/README.md#reader-and-prepared-track-schema) is unchanged. The repository-root `experiments/` remains the home of comparison records/configs/accepted overrides; `src/pedestrian_behavior/experiments/` contains executable experiment code.

## Data flow

**Preparation, already implemented:** native readers → verified transforms/frame selection/kinematics → saved collection with unchanged native supervision. Its present rules serve E001; their location does not make them universal defaults for later studies.

**Experiment setup, implemented for E001:** audit target policy/accepted overrides → determine eligible tracks and exclusion reasons → freeze recording groups/splits → fit normalization on source-training tracks only. Save manifests/statistics with collection and policy references under ignored experiment outputs. These artifacts reference saved tracks rather than copying trajectories. E001 uses one population/split assignment across all feature/model combinations.

**On-the-fly samples:** the Dataset loads a selected track through the existing `load_track()`, derives experiment targets and selected physical features, applies frozen normalization, and returns CPU tensors plus track references. It never fits statistics, chooses splits or silently skips samples. Feature construction receives physical arrays/masks; native behavior annotations go only to the target policy.

An E001 sample contains `inputs[T,D]` (float32, including feature-validity flags), `targets[T]` (int64), `gt_valid[T]` (bool), locator and saved-track reference. The collator pads to the batch's longest track and retains lengths, a padding mask and references. Internal missing observations remain sequence positions. The training loop moves tensors to the device; temporal models exclude batch padding from processing. Exact feature/GT/loss semantics remain in E001.

**Models, implemented:** shared `KinematicClassifier` owns a two-layer MLP encoder, optional single-layer BiLSTM and linear readout. E001's `build_model()` supplies dimensions/classes/dropout and chooses A/B; shared model code imports no experiment policy. Forward accepts padded `[B,T,D]` inputs and CPU int64 lengths, returns `[B,T,C]` logits, masks padding and packs recurrence without carrying state between calls. Ground truth is never a model input. [Hand-calculated output and padding/gradient acceptance](tests/test_e001_models.py) owns checks; [E001](experiments/E001-kinematic-transfer/README.md#model-implementation-and-acceptance-2026-10-08) owns settings/parameter counts.

Future RGB/LiDAR representations can retrieve native observations through existing references and add justified caches. Their sample layouts and any schema changes remain open.

## Keep it tidy

- Shared operations live in data/models/training/evaluation; experiment modules own target semantics, eligibility, selected components and run settings. Shared modules do not import experiment modules; pass ordinary functions/settings where a policy differs.
- Reuse existing functions, dictionaries, NumPy and PyTorch. Add abstractions only when concrete implementations require them; no plugin registry, speculative base classes or empty scaffolding.
- Group models by family. Keep the two kinematic baselines and shared components together; add a new family file when implemented.
- Agree each increment's input/output and readable acceptance tests before coding it. Reuse existing fixture/inspection tools where suitable; rerun existing tests for regressions.
- Update affected imports, commands, tests and owner documents together when moving code. Keep schemas in dataset notes, protocols/results in comparison records, and architecture here; link rather than duplicate.

The one-off `scripts/audit-road-associations.py` reuses the native reader and preview renderer to verify frozen ROAD sources/official links and generate a purposive review pack. E001 owns its acceptance record; the audit does not modify associations, targets or saved archives.

**Training and evaluation:** `training.py` seeds deterministic execution, trains equal-track minibatches, reports each update through an optional experiment callback, aggregates changing-weight epoch CE, selects from full source validation and atomically replaces state-dictionary checkpoints. `evaluation.py` owns shared masked loss, inference, raw/track-weighted confusion metrics, slice reweighting and pickle-free prediction persistence. It imports no E001 policy. `e001.py` owns four classes and the ten observation axes; conditions are derived from physical saved archives after inference. No GT/condition metadata enters model inputs or changes saved-track schema.

`experiments/run.py` owns the shared attempt lifecycle: fresh directory/failure evidence, seeding, one persistent training loader, source validation, checkpoint reload, final inference, local artifacts, plots and W&B. It receives ordinary assembly/condition functions, settings, class labels and stratum definitions; it imports no experiment policy. Local step/history/provenance writes precede external logging, and failures propagate. It keeps counter axes hidden, step/epoch curves separate, provenance in config and detailed tables in their namespace. Checkpoints/full predictions remain local. The restricted smoke uses source validation only; comparisons evaluate both held-out tests. No resume/restart/retry framework.

`e001_run.py` remains the E001 command/settings/data wrapper and preserves its existing helper interfaces/defaults. `e001b_run.py` supplies bbox datasets, 14/20-column BiLSTMs, both explicit source normalizations, bbox manifest/path/hash provenance and additional verified-box conditions. Both reuse `training.py`, `evaluation.py`, `KinematicClassifier` and E001's existing ontology/ten stratum definitions. E001b conditions are computed after inference from verified caches and saved with predictions; they never enter model inputs as GT/condition metadata.

`experiments/sweep.py` schedules a supplied attempt list/command function in bounded independent subprocesses, with separate logs/status and failure propagation. `e001_sweep.py` supplies the existing 80-attempt matrix; `e001b_sweep.py` supplies 20 BiLSTM availability/geometry attempts. Both use 75/8 and seeds 0–4 by default. All requested seeds reach model initialization, persistent loaders, checkpoints, provenance and W&B names. No shared model state, population changes or automatic retries. [Commands](README.md#e001b-training-and-evaluation) own tmux and isolated-checkout details.

**E001b bbox extension:** `datasets/bbox.py` retrieves native annotated boxes/dimensions and verifies original source checksums; `data/bbox.py` owns geometry, immediate-neighbor speeds, cache validation and numeric input construction. `experiments/e001b.py` owns the native-only geometry policy (Waymo FRONT without ROAD fallback; native LOKI), frozen E001 population references, coverage audit, source-training statistics and additive sample assembly. An explicit preparation-only Waymo root uses the existing native reader path override and records resolved component paths/checksums, allowing relocated mounts without editing frozen collections. Separate pickle-free caches reference existing E001 tracks; preparation never rewrites their arrays, setup or splits. Existing E001 sample/model/training/evaluation semantics stay unchanged; its orchestration now uses the shared runner. [E001b](experiments/E001b-2d-bbox-contribution/README.md) owns protocol/schema/gates; its thin runner and launcher reuse the shared operations above.
