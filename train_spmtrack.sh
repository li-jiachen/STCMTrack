#!/usr/bin/env bash
# Train the SPMTrack baseline (independent of STCMTrack; official structure of WenRuiCai/SPMTrack @ c581fe2).
#
#   DEVICE_IDS=0 ./train_spmtrack.sh                      # Anti-UAV410
#   DATASET=antiuav300 DEVICE_IDS=0 ./train_spmtrack.sh   # Anti-UAV
#
# Single stage. Each sample has three templates (frames sampled with the upstream "interval" rule) and two
# search frames; the query state is propagated from the first search frame to the second. The budget (80 epochs,
# global batch 4, 2048 samples per epoch) is the stage-1 budget of Sec. 3.1, not upstream's 170 epochs / batch 128.
# The result is a partial checkpoint (trainable parameters only) that carries `_spmtrack_port_version`.
# Extra arguments are passed to boot.sh. Train each dataset separately; do not mix their weights.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

resolve_conda_sh() {
    local conda_base=""

    if [[ -n "${CONDA_SH:-}" ]]; then
        if [[ -f "$CONDA_SH" ]]; then
            printf '%s\n' "$CONDA_SH"
            return 0
        fi
        echo "Conda activation script not found: $CONDA_SH" >&2
        return 1
    fi

    if [[ -n "${CONDA_EXE:-}" ]]; then
        conda_base="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    elif command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
    else
        for candidate in "$HOME/miniconda" "$HOME/miniconda3" "$HOME/anaconda3"; do
            if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
                conda_base="$candidate"
                break
            fi
        done
    fi

    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        printf '%s\n' "$conda_base/etc/profile.d/conda.sh"
        return 0
    fi

    echo "Conda activation script not found. Set CONDA_SH or make conda available on PATH." >&2
    return 1
}

DATASET="${DATASET:-antiuav410}"
case "$DATASET" in
    antiuav410) dataset_mixins=() ;;
    antiuav300) dataset_mixins=(dataset_antiuav300) ;;
    *) echo "Unsupported DATASET: $DATASET (expected antiuav410 or antiuav300)" >&2; exit 1 ;;
esac

CONDA_SH="$(resolve_conda_sh)"
CONDA_ENV="${CONDA_ENV:-stcmtrack}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found after activating the conda environment" >&2
    exit 1
fi
printf 'Using python3: %s\n' "$(command -v python3)"
printf 'CONDA_DEFAULT_ENV: %s\n' "${CONDA_DEFAULT_ENV:-}"

DEVICE_IDS="${DEVICE_IDS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/spmtrack_train_${DATASET}}"

mixin_names=(disable_torch_compile ${dataset_mixins[@]+"${dataset_mixins[@]}"})
exp_name="SPMTrack-Train-${DATASET}"

printf 'REPO_ROOT: %s\n' "$REPO_ROOT"
printf 'DATASET: %s | DEVICE_IDS: %s\n' "$DATASET" "$DEVICE_IDS"
printf 'OUTPUT_DIR: %s\n' "$OUTPUT_DIR"
printf 'Mixins: %s\n' "${mixin_names[*]}"

mkdir -p "$OUTPUT_DIR"

first_device_id="${DEVICE_IDS%%,*}"
first_device_id="${first_device_id//[[:space:]]/}"
if [[ ! "$first_device_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid DEVICE_IDS: $DEVICE_IDS (must start with a numeric GPU index)" >&2
    exit 1
fi

python3 - "$first_device_id" <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    print(f"Python path: {sys.executable}", file=sys.stderr)
    print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', '')}", file=sys.stderr)
    print(f"torch import failed: {exc}", file=sys.stderr)
    print("Check the GPU status with nvidia-smi.", file=sys.stderr)
    raise SystemExit(1)

device_id = int(sys.argv[1])
if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
    print(f"CUDA preflight failed: available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()} requested={device_id}", file=sys.stderr)
    print("Check the GPU status with nvidia-smi.", file=sys.stderr)
    raise SystemExit(1)
torch.cuda.set_device(device_id)
_ = torch.zeros(1, device=f"cuda:{device_id}") + 1
torch.cuda.synchronize()
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
PY

boot_args=()
for mixin_name in "${mixin_names[@]}"; do
    boot_args+=(--mixin "$mixin_name")
done

"$REPO_ROOT/boot.sh" SPMTrack dinov2 \
    --output_dir "$OUTPUT_DIR" \
    --device_ids "$DEVICE_IDS" \
    --disable_wandb \
    --timm_offline \
    --project_name STCMTrack \
    --exp_name "$exp_name" \
    "${boot_args[@]}" \
    "$@"
