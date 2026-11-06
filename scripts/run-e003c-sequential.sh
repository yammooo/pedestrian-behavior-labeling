#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
outputs_root="$repo_root/outputs"
python_path="${E003C_PYTHON:-$outputs_root/experiments/E002a/extractor-env/bin/python}"
device="${E003C_DEVICE:-cuda}"
attempt="$(date -u +%Y%m%d-%H%M%S)"
run_root="${E003C_RUN_ROOT:-$outputs_root/experiments/E003c/runs/temporal-adaptation-$attempt}"
feature_cache="$run_root/fusion-features"
probe_path="${E003C_PROBE:-$outputs_root/experiments/E002a/native-probe-07}"
rgb_cache_root="${E003C_RGB_CACHE_ROOT:-$outputs_root/experiments/E002a/rgb-cache}"
rgb_preparation="${E003C_RGB_PREPARATION:-$outputs_root/experiments/E002a/rgb-cache-preparation-03/cache.json}"
smoke_selection="${E003C_SMOKE_SELECTION:-$outputs_root/experiments/E003a/acceptance-20261103/selection.json}"
export PYTHONPATH="$repo_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

common=(--device "$device" --outputs-root "$outputs_root" --probe "$probe_path"
    --rgb-cache-root "$rgb_cache_root" --rgb-preparation "$rgb_preparation")
smoke=()
if [[ "${E003C_SMOKE:-0}" == "1" ]]; then
    smoke=(--smoke --smoke-selection "$smoke_selection")
fi

uv run --no-project --python "$python_path" python scripts/extract-e003c-features.py \
    "${common[@]}" --output "$feature_cache" "${smoke[@]}"

for arm in head temporal-head; do
    uv run --no-project --python "$python_path" python scripts/train-e003c.py \
        --arm "$arm" --device "$device" --outputs-root "$outputs_root" \
        --probe "$probe_path" --rgb-cache-root "$rgb_cache_root" \
        --rgb-preparation "$rgb_preparation" --feature-cache "$feature_cache" \
        --output "$run_root/$arm" "${smoke[@]}"
done
