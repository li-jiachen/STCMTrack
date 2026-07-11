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

EVAL_SCOPE="${EVAL_SCOPE:-short}"
case "$EVAL_SCOPE" in
    full|short)
        ;;
    *)
        echo "Unsupported EVAL_SCOPE: $EVAL_SCOPE (expected full or short)" >&2
        exit 1
        ;;
esac
printf 'Evaluation scope: %s\n' "$EVAL_SCOPE"

BASE_WEIGHT="${BASE_WEIGHT:-$REPO_ROOT/weights/stcmtrack_base.bin}"
LTCP_WEIGHT="${LTCP_WEIGHT:-$REPO_ROOT/weights/stcmtrack_ltcp.bin}"
ANTIUAV_GT_DIR="${ANTIUAV_GT_DIR:-/root/lanyun-fs/antiuav410/test}"
OUTPUT_ROOT="${OUTPUT_DIR:-$REPO_ROOT/output/stcmtrack_test_${EVAL_SCOPE}}"
DEVICE_IDS="${DEVICE_IDS:-0}"
REPORT_TAG="${REPORT_TAG:-stcmtrack_test_${EVAL_SCOPE}}"

if [[ ! -f "$BASE_WEIGHT" ]]; then
    echo "缺失 BASE_WEIGHT: $BASE_WEIGHT" >&2
    echo "可通过环境变量 BASE_WEIGHT 覆盖该路径" >&2
    exit 1
fi
if [[ ! -f "$LTCP_WEIGHT" ]]; then
    echo "缺失 LTCP_WEIGHT: $LTCP_WEIGHT" >&2
    echo "可通过环境变量 LTCP_WEIGHT 覆盖该路径" >&2
    exit 1
fi
if [[ ! -d "$ANTIUAV_GT_DIR" ]]; then
    echo "ANTIUAV_GT_DIR 不存在: $ANTIUAV_GT_DIR" >&2
    echo "可通过环境变量 ANTIUAV_GT_DIR 覆盖该路径" >&2
    exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"

printf 'OUTPUT_ROOT: %s\n' "$OUTPUT_ROOT"
printf 'RUN_ID: %s\n' "$RUN_ID"
printf 'RUN_OUTPUT_DIR: %s\n' "$RUN_OUTPUT_DIR"

mkdir -p "$RUN_OUTPUT_DIR"

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

device_id = int(sys.argv[1])
device_ids = os.environ.get("DEVICE_IDS", "")
python_path = sys.executable
conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

try:
    import torch
except Exception as exc:  # pragma: no cover - defensive runtime guard
    print(f"Python path: {python_path}", file=sys.stderr)
    print(f"CONDA_DEFAULT_ENV: {conda_env}", file=sys.stderr)
    print(f"torch.__version__: unavailable ({exc})", file=sys.stderr)
    print(f"torch.version.cuda: unavailable ({exc})", file=sys.stderr)
    print("torch.cuda.is_available(): unavailable (torch import failed)", file=sys.stderr)
    print("torch.cuda.device_count(): unavailable (torch import failed)", file=sys.stderr)
    print(f"DEVICE_IDS: {device_ids}", file=sys.stderr)
    print("请先运行 nvidia-smi 确认 GPU 状态。", file=sys.stderr)
    raise SystemExit(1)

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
    ctr
    ctr_foreground_correction
    ctr_threshold_040
    evaluation
)
case "$EVAL_SCOPE" in
    short)
        mixin_names+=(eval_short)
        ;;
esac

boot_mixin_args=()
for mixin_name in "${mixin_names[@]}"; do
    boot_mixin_args+=(--mixin "$mixin_name")
done

"$REPO_ROOT/boot.sh" STCMTrack dinov2 \
    "${boot_mixin_args[@]}" \
    --output_dir "$RUN_OUTPUT_DIR" \
    --device_ids "$DEVICE_IDS" \
    --disable_wandb \
    --timm_offline \
    --project_name STCMTrack \
    --exp_name "STCMTrack-Test-${EVAL_SCOPE}" \
    --weight_path "$BASE_WEIGHT" \
    --weight_path "$LTCP_WEIGHT"

results_zip="$(
    python3 - "$RUN_OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
candidates = [
    path
    for path in root.rglob("results.zip")
    if path.as_posix().endswith("/eval/epoch_0/results.zip")
    or path.as_posix().endswith("/eval/epoch_1/results.zip")
]

if not candidates:
    print(f"results.zip not found under current run dir: {root}", file=sys.stderr)
    raise SystemExit(1)

for path in candidates:
    print(f"candidate results.zip: {path} (mtime={path.stat().st_mtime:.0f})", file=sys.stderr)

selected = max(candidates, key=lambda path: (path.stat().st_mtime, path.as_posix()))
print(f"selected results.zip: {selected}", file=sys.stderr)
print(selected)
PY
)"

printf 'Evaluating results from current run: %s\n' "$results_zip"

python3 "$REPO_ROOT/tools/evaluate_antiuav_iou_p20.py" "$results_zip" \
    --gt-dir "$ANTIUAV_GT_DIR" \
    --sequence-csv "$REPO_ROOT/docs/${REPORT_TAG}_sequence.csv"
