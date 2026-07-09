#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

CONDA_SH_DEFAULT="/root/miniconda/etc/profile.d/conda.sh"
CONDA_SH="${CONDA_SH:-$CONDA_SH_DEFAULT}"
CONDA_ENV="${CONDA_ENV:-spmtrack}"

if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
elif [[ "$CONDA_SH" == "$CONDA_SH_DEFAULT" && -f /root/miniconda/bin/activate ]]; then
    source /root/miniconda/bin/activate "$CONDA_ENV"
else
    echo "找不到 Conda 激活脚本: $CONDA_SH" >&2
    echo "备用脚本也不可用: /root/miniconda/bin/activate" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Conda 环境激活后未找到 python3" >&2
    exit 1
fi
printf 'Using python3: %s\n' "$(command -v python3)"
printf 'CONDA_DEFAULT_ENV: %s\n' "${CONDA_DEFAULT_ENV:-}"

DEVICE_IDS="${DEVICE_IDS:-0}"
BASE_WEIGHT="${BASE_WEIGHT:-$REPO_ROOT/weights/base_finetuning.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/stcmtrack_train}"

printf 'REPO_ROOT: %s\n' "$REPO_ROOT"
printf 'CONDA_SH: %s\n' "$CONDA_SH"
printf 'CONDA_ENV: %s\n' "$CONDA_ENV"
printf 'DEVICE_IDS: %s\n' "$DEVICE_IDS"
printf 'BASE_WEIGHT: %s\n' "$BASE_WEIGHT"
printf 'OUTPUT_DIR: %s\n' "$OUTPUT_DIR"

if [[ ! -f "$BASE_WEIGHT" ]]; then
    echo "BASE_WEIGHT not found: $BASE_WEIGHT" >&2
    echo "可通过环境变量 BASE_WEIGHT 覆盖该路径" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

first_device_id="${DEVICE_IDS%%,*}"
first_device_id="${first_device_id//[[:space:]]/}"
if [[ ! "$first_device_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid DEVICE_IDS: $DEVICE_IDS" >&2
    echo "DEVICE_IDS 必须以数字 GPU 编号开头" >&2
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
    print(f"torch.__version__: unavailable ({exc})", file=sys.stderr)
    print("torch.version.cuda: unavailable (torch import failed)", file=sys.stderr)
    print("torch.cuda.is_available(): unavailable (torch import failed)", file=sys.stderr)
    print("torch.cuda.device_count(): unavailable (torch import failed)", file=sys.stderr)
    print(f"DEVICE_IDS: {os.environ.get('DEVICE_IDS', '')}", file=sys.stderr)
    print("请先运行 nvidia-smi 确认 GPU 状态。", file=sys.stderr)
    raise SystemExit(1)

device_id = int(sys.argv[1])
device_ids = os.environ.get("DEVICE_IDS", "")
python_path = sys.executable
conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    print(f"Python path: {python_path}", file=sys.stderr)
    print(f"CONDA_DEFAULT_ENV: {conda_env}", file=sys.stderr)
    print(f"torch.__version__: {torch.__version__}", file=sys.stderr)
    print(f"torch.version.cuda: {torch.version.cuda}", file=sys.stderr)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", file=sys.stderr)
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}", file=sys.stderr)
    print(f"DEVICE_IDS: {device_ids}", file=sys.stderr)
    print("请先运行 nvidia-smi 确认 GPU 状态。", file=sys.stderr)
    raise SystemExit(1)

if not torch.cuda.is_available():
    fail("CUDA preflight failed: torch.cuda.is_available() is False")

device_count = torch.cuda.device_count()
if device_count <= 0:
    fail("CUDA preflight failed: torch.cuda.device_count() <= 0")

if device_id >= device_count:
    fail(
        f"CUDA preflight failed: requested GPU index {device_id} is out of range for device_count={device_count}"
    )

try:
    torch.cuda.set_device(device_id)
    test_tensor = torch.zeros(1, device=f"cuda:{device_id}")
    _ = test_tensor + 1
    torch.cuda.synchronize()
except Exception as exc:
    fail(f"CUDA preflight failed during tensor test: {exc}")

print(f"Python path: {python_path}")
print(f"CONDA_DEFAULT_ENV: {conda_env}")
print(f"torch.__version__: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {device_count}")
print(f"Current GPU index: {device_id}")
print(f"Current GPU name: {torch.cuda.get_device_name(device_id)}")
print(f"DEVICE_IDS: {device_ids}")
PY

mixin_names=(
    disable_torch_compile
    ltcp
    ltcp_memory_2
)

printf 'Training entry command summary: %s\n' "$REPO_ROOT/boot.sh STCMTrack dinov2 --output_dir \"$OUTPUT_DIR\" --device_ids \"$DEVICE_IDS\" --disable_wandb --timm_offline --project_name STCMTrack --exp_name STCMTrack-Train --mixin disable_torch_compile --mixin ltcp --mixin ltcp_memory_2 --weight_path \"$BASE_WEIGHT\""
printf 'Mixin list: %s\n' "${mixin_names[*]}"

"$REPO_ROOT/boot.sh" STCMTrack dinov2 \
    --output_dir "$OUTPUT_DIR" \
    --device_ids "$DEVICE_IDS" \
    --disable_wandb \
    --timm_offline \
    --project_name STCMTrack \
    --exp_name STCMTrack-Train \
    --mixin disable_torch_compile \
    --mixin ltcp \
    --mixin ltcp_memory_2 \
    --weight_path "$BASE_WEIGHT" \
    "$@"
