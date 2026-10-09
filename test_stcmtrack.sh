#!/usr/bin/env bash
# Evaluate the eight configurations in Table 2 of the paper.
#
#   DEVICE_IDS=0 ./test_stcmtrack.sh                       # full model on the Anti-UAV410 test set
#   VARIANT=baseline BASE_WEIGHT=/path/to/spmtrack.safetensors DEVICE_IDS=0 ./test_stcmtrack.sh  # row 1
#   VARIANT=mcc_rgtc DEVICE_IDS=0 ./test_stcmtrack.sh      # Table 2, row 7
#   DATASET=antiuav300 DEVICE_IDS=0 ./test_stcmtrack.sh    # Anti-UAV
#   EVAL_SCOPE=short DEVICE_IDS=0 ./test_stcmtrack.sh      # quick run: 4 sequences, at most 200 frames each
#
# VARIANT (module combinations in Table 2): baseline(1) ltcp(2) mcc(3) rgtc(4) ltcp_mcc(5)
#   ltcp_rgtc(6) mcc_rgtc(7) full(8, default).
#   baseline             independent SPMTrack, used by Table 2 row 1 and the Table 1 comparison:
#                        three templates, propagated query state,
#                        head-input re-weighting, Hann window; official WenRuiCai/SPMTrack @ c581fe2), without
#                        LTCP, MCC and RGTC. It uses its own weight file (spmtrack_baseline.safetensors);
#                        STCMTrack weights are refused. ALLOW_UNMARKED_SPMTRACK_WEIGHTS=1 declares a file
#                        without the marker of this code as trained with the official SPMTrack code.
#   ltcp ... full        STCMTrack (method STCMTrack) with the components of the row switched on.
# Rows 2-8 share the STCMTrack base network and non-component settings. Row 1 retains the original
# SPMTrack structure and inference; it requires separate SPMTrack weights.
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

EVAL_SCOPE="${EVAL_SCOPE:-full}"
case "$EVAL_SCOPE" in
    full|short) ;;
    *) echo "Unsupported EVAL_SCOPE: $EVAL_SCOPE (expected full or short)" >&2; exit 1 ;;
esac

VARIANT="${VARIANT:-full}"
method_name=STCMTrack   # model family that boot.sh builds: STCMTrack or SPMTrack
case "$VARIANT" in
    baseline) method_name=SPMTrack; variant_mixins=();         use_ltcp=false ;;
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
if [[ "$method_name" == SPMTrack ]]; then
    # The separate SPMTrack comparison needs weights trained with its own structure.
    case "$DATASET" in
        antiuav410) default_base_weight="$REPO_ROOT/weights/spmtrack_baseline.safetensors" ;;
        antiuav300) default_base_weight="$REPO_ROOT/weights/spmtrack_antiuav300_baseline.safetensors" ;;
    esac
fi
printf 'Evaluation scope: %s | variant: %s | model: %s | dataset: %s\n' "$EVAL_SCOPE" "$VARIANT" "$method_name" "$DATASET"

BASE_WEIGHT="${BASE_WEIGHT:-$default_base_weight}"
LTCP_WEIGHT="${LTCP_WEIGHT:-$default_ltcp_weight}"
# Read the same constants as the tracker; explicit overrides are exported to its dataset seed.
if [[ -z "${ANTIUAV_GT_DIR:-}" ]]; then
    ANTIUAV_GT_DIR="$(python3 - "$DATASET" <<'PYCODE'
import sys
from pathlib import Path
from trackit.core.runtime.global_constant import get_global_constant
key = 'ANTIUAV410_PATH' if sys.argv[1] == 'antiuav410' else 'ANTIUAV300_TEST_PATH'
print(Path(get_global_constant(key)).expanduser().resolve())
PYCODE
)"
fi
export ANTIUAV_GT_DIR
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

if [[ "$method_name" == SPMTrack ]]; then
    spmtrack_check_args=(--weights "$BASE_WEIGHT")
    if [[ "${ALLOW_UNMARKED_SPMTRACK_WEIGHTS:-0}" == 1 ]]; then
        spmtrack_check_args+=(--allow-unmarked)
    fi
    python3 "$REPO_ROOT/tools/check_spmtrack_weights.py" "${spmtrack_check_args[@]}"
else
    weight_check_args=(--base "$BASE_WEIGHT")
    if [[ "$use_ltcp" == true ]]; then
        weight_check_args+=(--ltcp "$LTCP_WEIGHT")
    fi
    python3 "$REPO_ROOT/tools/check_stcmtrack_weights.py" "${weight_check_args[@]}"
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
if [[ "$method_name" == SPMTrack && "${ALLOW_UNMARKED_SPMTRACK_WEIGHTS:-0}" == 1 ]]; then
    mixin_names+=(spmtrack_allow_unmarked_weights)
fi
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

"$REPO_ROOT/boot.sh" "$method_name" dinov2 \
    "${boot_args[@]}" \
    --output_dir "$RUN_OUTPUT_DIR" \
    --device_ids "$DEVICE_IDS" \
    --disable_wandb \
    --timm_offline \
    --project_name STCMTrack \
    --exp_name "${method_name}-Test-${DATASET}-${VARIANT}-${EVAL_SCOPE}"

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

metric_args=()
if [[ "$EVAL_SCOPE" == short ]]; then
    metric_args+=(--allow-partial)
    echo "Partial evaluation (EVAL_SCOPE=short): the first 4 sequences, at most 200 frames each."
elif [[ "$DATASET" == antiuav410 ]]; then
    metric_args+=(--expected-sequences 120)
fi
python3 "$REPO_ROOT/tools/evaluate_antiuav_iou_p20.py" "$results_zip" \
    --gt-dir "$ANTIUAV_GT_DIR" \
    --sequence-csv "$RUN_OUTPUT_DIR/${REPORT_TAG}_sequence.csv" \
    --report-json "$RUN_OUTPUT_DIR/${REPORT_TAG}_metrics.json" \
    "${metric_args[@]}"
