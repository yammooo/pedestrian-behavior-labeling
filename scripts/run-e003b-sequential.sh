#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
outputs_root="$repo_root/outputs"
python_path="${E003B_PYTHON:-$outputs_root/experiments/E002a/extractor-env/bin/python}"
device="${E003B_DEVICE:-cuda}"
attempt="$(date -u +%Y%m%d-%H%M%S)"
run_root="${E003B_RUN_ROOT:-$outputs_root/experiments/E003b/runs/$attempt}"
probe_path="${E003B_PROBE:-$outputs_root/experiments/E002a/native-probe-07}"
rgb_cache_root="${E003B_RGB_CACHE_ROOT:-$outputs_root/experiments/E002a/rgb-cache}"
rgb_preparation="${E003B_RGB_PREPARATION:-$outputs_root/experiments/E002a/rgb-cache-preparation-03/cache.json}"
smoke_selection="${E003B_SMOKE_SELECTION:-$outputs_root/experiments/E003a/acceptance-20261103/selection.json}"
export PYTHONPATH="$repo_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_case() {
    local reference="$1"
    local source="$2"
    local target="$3"
    local command=(uv run --no-project --python "$python_path" python scripts/train-e003b.py
        --source "$source" --reference "$reference" --device "$device"
        --outputs-root "$outputs_root" --output "$run_root/$reference-$source-to-$target"
        --probe "$probe_path" --rgb-cache-root "$rgb_cache_root" --rgb-preparation "$rgb_preparation")
    if [[ "${E003B_SMOKE:-0}" == "1" ]]; then
        command+=(--smoke --smoke-selection "$smoke_selection")
    fi
    "${command[@]}"
}

run_case e002d road-waymo loki
run_case e003a road-waymo loki
run_case e002d loki road-waymo
run_case e003a loki road-waymo
