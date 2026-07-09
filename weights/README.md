# Model Weights

This directory stores the checkpoint files required for STCMTrack inference and training.

Large weight files are intentionally not committed to Git. Download them manually or fetch them from release artifacts, then place them at the paths expected by the scripts.

By default:

- `test_stcmtrack.sh` loads `weights/base_finetuning.bin` first, then loads `weights/ltcp_epoch09_model_compat.bin`.
- `train_stcmtrack.sh` uses `weights/base_finetuning.bin` as the initialization checkpoint.

## Weight Files

| File | Required by | Purpose | SHA256 | Notes |
|---|---|---|---|---|
| `weights/base_finetuning.bin` | `test_stcmtrack.sh`, `train_stcmtrack.sh` | Base STCMTrack/SPMTrack-compatible checkpoint | `71eb8f20cbdec8bf2dff5177806923eca0fa08b694c2b40cf5399f2d9ad2d2d0` | In the current development environment, this file is a symlink to `/root/lanyun-fs/fine-tuning-Base.bin`. For a fresh clone, place or symlink the downloaded base checkpoint at `weights/base_finetuning.bin`. |
| `weights/ltcp_epoch09_model_compat.bin` | `test_stcmtrack.sh` | LTCP incremental checkpoint with compatible key names | `ed4e063bcac94e6fd98de14abb1416965f4a26ad92f482602234807228b291b7` | Created by renaming `ttcp.gate.weight` to `ltcp.gate.weight` and `ttcp.gate.bias` to `ltcp.gate.bias` from the original LTCP checkpoint. The original checkpoint was not modified. |

## How to Place the Weights

Use placeholders for your own download locations:

```bash
mkdir -p weights

wget -O weights/base_finetuning.bin <BASE_CHECKPOINT_URL>
wget -O weights/ltcp_epoch09_model_compat.bin <LTCP_COMPAT_CHECKPOINT_URL>

sha256sum weights/base_finetuning.bin
sha256sum weights/ltcp_epoch09_model_compat.bin
```

A symlink is also acceptable when the checkpoint files are stored outside the repository.

## Environment Variable Overrides

`test_stcmtrack.sh` supports these overrides:

- `BASE_WEIGHT=/path/to/base.bin`
- `LTCP_WEIGHT=/path/to/ltcp_compat.bin`
- `DEVICE_IDS=0`
- `EVAL_SCOPE=short` or `EVAL_SCOPE=full`

Example:

```bash
BASE_WEIGHT=/path/to/base_finetuning.bin \
LTCP_WEIGHT=/path/to/ltcp_epoch09_model_compat.bin \
DEVICE_IDS=0 \
./test_stcmtrack.sh
```

`train_stcmtrack.sh` supports these overrides:

- `BASE_WEIGHT=/path/to/base.bin`
- `DEVICE_IDS=0`
- `OUTPUT_DIR=/path/to/output`

Example:

```bash
BASE_WEIGHT=/path/to/base_finetuning.bin \
DEVICE_IDS=0 \
OUTPUT_DIR=/path/to/train_output \
./train_stcmtrack.sh
```

## Missing Keys

The base checkpoint and the LTCP incremental checkpoint are loaded sequentially.

The LTCP compatibility file only contains a small set of incremental parameters, so loading it by itself will naturally produce many `missing keys` messages. This is expected.

During sanity checks, the important condition is that `unexpected keys` should remain 0. Do not change the model structure or checkpoint format just to remove `missing keys` warnings.

## Do Not Commit Weights

Weight files are ignored by Git. Do not commit `.bin`, `.pth`, `.pt`, `.ckpt`, or `.safetensors` files.

Use GitHub Releases or external download locations for checkpoint distribution.
