#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
outputs_root="$repo_root/outputs"
python_path="${E003C_PYTHON:-$outputs_root/experiments/E002a/extractor-env/bin/python}"
device="${E003C_DEVICE:-cuda}"
attempt="$(date -u +%Y%m%d-%H%M%S)"
run_root="${E003C_RUN_ROOT:-$outputs_root/experiments/E003c/runs/source-adaptation-$attempt}"
default_feature_cache="$outputs_root/experiments/E003c/fusion-features-road-waymo"
feature_cache="${E003C_FEATURE_CACHE:-$default_feature_cache}"
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
    if [[ -z "${E003C_FEATURE_CACHE:-}" || "$feature_cache" == "$default_feature_cache" ]]; then
        echo "E003C smoke requires E003C_FEATURE_CACHE pointing to a separate fresh cache." >&2
        exit 2
    fi
    smoke=(--smoke --smoke-selection "$smoke_selection")
fi

if [[ -e "$feature_cache" ]]; then
    if [[ ! -f "$feature_cache/status.json" ]]; then
        echo "Feature cache exists without status.json; refusing to retry or overwrite it: $feature_cache" >&2
        exit 2
    fi
    cache_status="$(uv run --no-project --python "$python_path" python -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$feature_cache/status.json")"
    if [[ "$cache_status" != "complete" ]]; then
        echo "Feature cache is not complete ($cache_status); choose a fresh E003C_FEATURE_CACHE." >&2
        exit 2
    fi
    echo "Reusing complete feature cache; each trainer will verify its lineage and checksums."
else
    uv run --no-project --python "$python_path" python scripts/extract-e003c-features.py \
        "${common[@]}" --output "$feature_cache" "${smoke[@]}"
fi

for arm in head temporal-head; do
    uv run --no-project --python "$python_path" python scripts/train-e003c.py \
        --arm "$arm" --device "$device" --outputs-root "$outputs_root" \
        --probe "$probe_path" --rgb-cache-root "$rgb_cache_root" \
        --rgb-preparation "$rgb_preparation" --feature-cache "$feature_cache" \
        --output "$run_root/$arm" "${smoke[@]}"
done
