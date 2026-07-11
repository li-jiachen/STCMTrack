# Model Weights

This directory stores the checkpoint files required for STCMTrack inference and training.

Weight files are intentionally not committed to Git.

By default:

- `test_stcmtrack.sh` loads `weights/stcmtrack_base.bin` first, then loads `weights/stcmtrack_ltcp.bin`.
- `train_stcmtrack.sh` uses `weights/stcmtrack_base.bin` as the initialization checkpoint.

## Weight Files

| File | Required by | Purpose | SHA256 | Notes |
|---|---|---|---|---|
| [`weights/stcmtrack_base.bin`](https://github.com/li-jiachen/STCMTrack/releases/download/v0.2.0/stcmtrack_base.bin) | `test_stcmtrack.sh`, `train_stcmtrack.sh` | Base initialization checkpoint | `71eb8f20cbdec8bf2dff5177806923eca0fa08b694c2b40cf5399f2d9ad2d2d0` | This checkpoint was trained by the STCMTrack authors and is publicly available in GitHub Release `v0.2.0`. |
| [`weights/stcmtrack_ltcp.bin`](https://github.com/li-jiachen/STCMTrack/releases/download/v0.2.0/stcmtrack_ltcp.bin) | `test_stcmtrack.sh` | LTCP incremental compatibility checkpoint | `ed4e063bcac94e6fd98de14abb1416965f4a26ad92f482602234807228b291b7` | This file only contains incremental parameters and is publicly available in GitHub Release `v0.2.0`. |

## How to Place the Weights

Base checkpoint download:

```bash
mkdir -p weights
wget -O weights/stcmtrack_base.bin \
  https://github.com/li-jiachen/STCMTrack/releases/download/v0.2.0/stcmtrack_base.bin
sha256sum weights/stcmtrack_base.bin
```

LTCP checkpoint:

```bash
mkdir -p weights
wget -O weights/stcmtrack_ltcp.bin \
  https://github.com/li-jiachen/STCMTrack/releases/download/v0.2.0/stcmtrack_ltcp.bin
sha256sum weights/stcmtrack_ltcp.bin
```

A symlink is also acceptable when the checkpoint files are stored outside the repository:

```bash
ln -s /path/to/stcmtrack_base.bin weights/stcmtrack_base.bin
ln -s /path/to/stcmtrack_ltcp.bin weights/stcmtrack_ltcp.bin
```

## Environment Variable Overrides

`test_stcmtrack.sh` supports these overrides:

- `BASE_WEIGHT=/path/to/stcmtrack_base.bin`
- `LTCP_WEIGHT=/path/to/stcmtrack_ltcp.bin`
- `DEVICE_IDS=0`
- `EVAL_SCOPE=short` or `EVAL_SCOPE=full`

Example:

```bash
BASE_WEIGHT=/path/to/stcmtrack_base.bin LTCP_WEIGHT=/path/to/stcmtrack_ltcp.bin DEVICE_IDS=0 ./test_stcmtrack.sh
```

`train_stcmtrack.sh` supports these overrides:

- `BASE_WEIGHT=/path/to/stcmtrack_base.bin`
- `DEVICE_IDS=0`
- `OUTPUT_DIR=/path/to/output`

Example:

```bash
BASE_WEIGHT=/path/to/stcmtrack_base.bin DEVICE_IDS=0 OUTPUT_DIR=/path/to/train_output ./train_stcmtrack.sh
```

## Missing Keys

The base checkpoint and the LTCP incremental checkpoint are loaded sequentially.

The LTCP compatibility file only contains a small set of incremental parameters, so loading it by itself will naturally produce many `missing keys` messages. This is expected.

During sanity checks, the important condition is that `unexpected keys` should remain 0. Do not change the model structure or checkpoint format just to remove `missing keys` warnings.

## Do Not Commit Weights

Weight files are ignored by Git. Do not commit `.bin`, `.pth`, `.pt`, `.ckpt`, or `.safetensors` files.

Use GitHub Releases or external download locations for checkpoint distribution.
