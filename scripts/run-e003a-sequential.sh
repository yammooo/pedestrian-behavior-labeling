#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
outputs_root="$repo_root/outputs"
profile_path="${E003A_PROFILE:-$outputs_root/experiments/E003a/calibration/profile.json}"
rgb_preparation="${E003A_RGB_PREPARATION:-$outputs_root/experiments/E002a/rgb-cache-preparation-03/cache.json}"
probe_path="${E003A_PROBE:-$outputs_root/experiments/E002a/native-probe-07}"
device="${E003A_DEVICE:-cuda}"
attempt="$(date -u +%Y%m%d-%H%M%S)"
run_root="${E003A_RUN_ROOT:-$outputs_root/experiments/E003a/runs/$attempt}"
smoke_selection="${E003A_SMOKE_SELECTION:-}"
control_transfer_results="${E003A_CONTROL_TRANSFER_RESULTS:-}"
python_path="$outputs_root/experiments/E002a/extractor-env/bin/python"
export PYTHONPATH="$repo_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_source() {
    local source="$1"
    local control="$outputs_root/experiments/E002d/runs/fusion-20261030-140937-$source-rgbdrop50"
    local command=(uv run --no-project --python "$python_path" python scripts/train-e003a.py
        --source "$source" --device "$device"
        --profile "$profile_path" --control-run "$control"
        --rgb-preparation "$rgb_preparation" --opposite-rgb-preparation "$rgb_preparation"
        --probe "$probe_path" --outputs-root "$outputs_root"
        --output "$run_root/$source")
    if [[ -n "$control_transfer_results" ]]; then
        command+=(--control-transfer-results "$control_transfer_results")
    fi
    if [[ -n "$smoke_selection" ]]; then
        command+=(--smoke --smoke-selection "$smoke_selection")
    fi
    "${command[@]}"
}

run_source road-waymo
run_source loki
