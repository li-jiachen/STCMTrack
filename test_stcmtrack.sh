#!/usr/bin/env bash
# Evaluate STCMTrack or one of the ablation variants of Table 2.
#
#   EVAL_SCOPE=full DEVICE_IDS=0 ./test_stcmtrack.sh                      # full model on Anti-UAV410
#   EVAL_SCOPE=full VARIANT=mcc_rgtc DEVICE_IDS=0 ./test_stcmtrack.sh     # Table 2, row 7
#   EVAL_SCOPE=full DATASET=antiuav300 DEVICE_IDS=0 ./test_stcmtrack.sh   # Anti-UAV
#
# VARIANT (Table 2 rows): baseline(1) ltcp(2) mcc(3) rgtc(4) ltcp_mcc(5) ltcp_rgtc(6) mcc_rgtc(7) full(8)
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
CONDA_SH="$(resolve_conda_sh)"
CONDA_ENV="${CONDA_ENV:-stcmtrack}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found after activating the conda environment" >&2
    exit 1
fi
printf 'Using python3: %s\n' "$(command -v python3)"

EVAL_SCOPE="${EVAL_SCOPE:-short}"
case "$EVAL_SCOPE" in
    full|short) ;;
    *) echo "Unsupported EVAL_SCOPE: $EVAL_SCOPE (expected full or short)" >&2; exit 1 ;;
esac

VARIANT="${VARIANT:-full}"
case "$VARIANT" in
    baseline)  variant_mixins=();                             use_ltcp=false ;;
    ltcp)      variant_mixins=(ltcp);                         use_ltcp=true ;;
    mcc)       variant_mixins=(ctr ctr_no_rgtc);              use_ltcp=false ;;
    rgtc)      variant_mixins=(ctr ctr_no_mcc);               use_ltcp=false ;;
    ltcp_mcc)  variant_mixins=(ltcp ctr ctr_no_rgtc);         use_ltcp=true ;;
    ltcp_rgtc) variant_mixins=(ltcp ctr ctr_no_mcc);          use_ltcp=true ;;
    mcc_rgtc)  variant_mixins=(ctr);                          use_ltcp=false ;;
    full)      variant_mixins=(ltcp ctr);                     use_ltcp=true ;;
    *)
        echo "Unsupported VARIANT: $VARIANT" >&2
        echo "Expected one of: baseline ltcp mcc rgtc ltcp_mcc ltcp_rgtc mcc_rgtc full" >&2
        exit 1
        ;;
esac

DATASET="${DATASET:-antiuav410}"
case "$DATASET" in
    antiuav410)
        dataset_mixins=()
        default_base_weight="$REPO_ROOT/weights/stcmtrack_base.bin"
        default_ltcp_weight="$REPO_ROOT/weights/stcmtrack_ltcp.bin"
        default_gt_dir="$REPO_ROOT/../antiuav410/test"
        ;;
    antiuav300)
        dataset_mixins=(dataset_antiuav300)
        default_base_weight="$REPO_ROOT/weights/stcmtrack_antiuav300_base.bin"
        default_ltcp_weight="$REPO_ROOT/weights/stcmtrack_antiuav300_ltcp.bin"
        default_gt_dir="$REPO_ROOT/../antiuav300_ir/test"
        ;;
    *) echo "Unsupported DATASET: $DATASET (expected antiuav410 or antiuav300)" >&2; exit 1 ;;
esac
printf 'Evaluation scope: %s | variant: %s | dataset: %s\n' "$EVAL_SCOPE" "$VARIANT" "$DATASET"

BASE_WEIGHT="${BASE_WEIGHT:-$default_base_weight}"
LTCP_WEIGHT="${LTCP_WEIGHT:-$default_ltcp_weight}"
ANTIUAV_GT_DIR="${ANTIUAV_GT_DIR:-$default_gt_dir}"
OUTPUT_ROOT="${OUTPUT_DIR:-$REPO_ROOT/output/stcmtrack_test_${DATASET}_${VARIANT}_${EVAL_SCOPE}}"
DEVICE_IDS="${DEVICE_IDS:-0}"
REPORT_TAG="${REPORT_TAG:-stcmtrack_${DATASET}_${VARIANT}_${EVAL_SCOPE}}"

if [[ ! -f "$BASE_WEIGHT" ]]; then
    echo "BASE_WEIGHT not found: $BASE_WEIGHT (override with BASE_WEIGHT=...)" >&2
    exit 1
fi
if [[ "$use_ltcp" == true && ! -f "$LTCP_WEIGHT" ]]; then
    echo "LTCP_WEIGHT not found: $LTCP_WEIGHT (override with LTCP_WEIGHT=...)" >&2
    exit 1
fi
if [[ ! -d "$ANTIUAV_GT_DIR" ]]; then
    echo "ANTIUAV_GT_DIR not found: $ANTIUAV_GT_DIR (override with ANTIUAV_GT_DIR=...)" >&2
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
    echo "Invalid DEVICE_IDS: $DEVICE_IDS (must start with a numeric GPU index)" >&2
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
    print(f"torch import failed: {exc}", file=sys.stderr)
    print(f"DEVICE_IDS: {device_ids}", file=sys.stderr)
    print("Check the GPU status with nvidia-smi.", file=sys.stderr)
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
    print("Check the GPU status with nvidia-smi.", file=sys.stderr)
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
print(f"Current GPU: {device_id} ({torch.cuda.get_device_name(device_id)})")
PY

mixin_names=(disable_torch_compile ${dataset_mixins[@]+"${dataset_mixins[@]}"} ${variant_mixins[@]+"${variant_mixins[@]}"} evaluation)
if [[ "$EVAL_SCOPE" == short ]]; then
    mixin_names+=(eval_short)
fi
printf 'Mixins: %s\n' "${mixin_names[*]}"

boot_args=()
for mixin_name in "${mixin_names[@]}"; do
    boot_args+=(--mixin "$mixin_name")
done
boot_args+=(--weight_path "$BASE_WEIGHT")
if [[ "$use_ltcp" == true ]]; then
    boot_args+=(--weight_path "$LTCP_WEIGHT")
fi

"$REPO_ROOT/boot.sh" STCMTrack dinov2 \
    "${boot_args[@]}" \
    --output_dir "$RUN_OUTPUT_DIR" \
    --device_ids "$DEVICE_IDS" \
    --disable_wandb \
    --timm_offline \
    --project_name STCMTrack \
    --exp_name "STCMTrack-Test-${DATASET}-${VARIANT}-${EVAL_SCOPE}"

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
    --sequence-csv "$RUN_OUTPUT_DIR/${REPORT_TAG}_sequence.csv"
