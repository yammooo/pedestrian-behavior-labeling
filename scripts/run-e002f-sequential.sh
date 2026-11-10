#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
outputs_root="$repo_root/outputs"
python_path="${E002F_PYTHON:-$outputs_root/experiments/E002a/extractor-env/bin/python}"
device="${E002F_DEVICE:-cuda}"
attempt="$(date -u +%Y%m%d-%H%M%S)"
run_root="${E002F_RUN_ROOT:-$outputs_root/experiments/E002f/runs/temporal-architecture-$attempt}"
feature_root="${E002F_FEATURE_ROOT:-$outputs_root/experiments/E002f/features/temporal-architecture-$attempt}"
profile_path="${E002F_PROFILE:-$outputs_root/experiments/E002f/calibration/full-batch-$attempt/profile.json}"
reference_profile="${E002F_REFERENCE_PROFILE:-$outputs_root/experiments/E002d/calibration-20261030-sequential-ready/profile.json}"
probe_path="${E002F_PROBE:-$outputs_root/experiments/E002a/native-probe-07}"
rgb_cache_root="${E002F_RGB_CACHE_ROOT:-$outputs_root/experiments/E002a/rgb-cache}"
rgb_preparation="${E002F_RGB_PREPARATION:-$outputs_root/experiments/E002a/rgb-cache-preparation-03/cache.json}"
smoke_selection="${E002F_SMOKE_SELECTION:-$outputs_root/experiments/E003a/acceptance-20261103/selection.json}"

if [[ -e "$run_root" || -e "$feature_root" ]]; then
    echo "E002f run/feature roots must be fresh: $run_root, $feature_root" >&2
    exit 2
fi
export PYTHONPATH="$repo_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [[ -e "$profile_path" ]]; then
    profile_status="$(uv run --no-project --python "$python_path" python -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$profile_path")"
    if [[ "$profile_status" != "complete" ]]; then
        echo "E002f calibration exists but is not complete; refusing to retry: $profile_path" >&2
        exit 2
    fi
else
    uv run --no-project --python "$python_path" python scripts/calibrate-e002f.py \
        --device "$device" --outputs-root "$outputs_root" --reference-profile "$reference_profile" \
        --output "$profile_path"
fi

mkdir -p "$run_root" "$feature_root"

smoke=()
if [[ "${E002F_SMOKE:-0}" == "1" ]]; then
    smoke=(--smoke --smoke-selection "$smoke_selection")
fi
for architecture in transformer-1layer bilstm-96; do
    source_run="$run_root/source/$architecture"
    cache="$feature_root/$architecture"
    uv run --no-project --python "$python_path" python scripts/train-e002f.py source \
        --architecture "$architecture" --profile "$profile_path" --output "$source_run" \
        --outputs-root "$outputs_root" --probe "$probe_path" --rgb-cache-root "$rgb_cache_root" \
        --rgb-preparation "$rgb_preparation" --device "$device" "${smoke[@]}"
    uv run --no-project --python "$python_path" python scripts/train-e002f.py extract \
        --architecture "$architecture" --source-run "$source_run" --output "$cache" \
        --outputs-root "$outputs_root" --probe "$probe_path" --rgb-cache-root "$rgb_cache_root" \
        --rgb-preparation "$rgb_preparation" --device "$device" "${smoke[@]}"
    for arm in head temporal-head; do
        uv run --no-project --python "$python_path" python scripts/train-e002f.py adapt \
            --architecture "$architecture" --arm "$arm" --source-run "$source_run" \
            --feature-cache "$cache" --output "$run_root/adaptation/$architecture/$arm" \
            --outputs-root "$outputs_root" --device "$device" "${smoke[@]}"
    done
done
echo "E002f completed under $run_root (feature caches: $feature_root)"
