# [Under Review] STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking

Official implementation of “STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking”.

This repository provides the released inference and evaluation pipeline, together with a training entry point and checkpoint placement instructions.

## News

- Inference, evaluation, and training entry points are available.
- The base checkpoint is available through GitHub Releases. The LTCP incremental checkpoint is not yet publicly available; see `weights/README.md` for its expected filename and checksum.
- Checkpoint files are not committed to Git. Please see `weights/README.md` for placement instructions.

### ⭐ Introduction

Infrared anti-UAV tracking is difficult because UAV targets are tiny, low-contrast, weak-textured, and easily affected by fast motion, occlusion, out-of-view, and dynamic background clutter.

STCMTrack is built on a DINOv2 / ViT-B style local tracking backbone. It introduces confidence-guided spatio-temporal context modeling.

It contains two main modules:

- LTCP: Local-enhanced Temporal Context Propagation
- CTR: Confidence-Triggered Re-localization

LTCP models local region-level temporal consistency between current search tokens and historical search-token memory. The current verified memory size is `2`.

CTR uses prediction confidence to trigger re-localization. The paper describes motion center correction and residual-guided target correction, while the released implementation currently provides the verified foreground-correction path based on MOG2. The residual-guided branch described in the paper is not presented as a fully verified release in this repository.

### ⭐ Strong Performance

Paper-reported results on the Anti-UAV410 test set.

#### Comparison with Local Trackers

| Method | AUC (%) | P (%) | PNorm (%) |
|---|---:|---:|---:|
| OSTrack | 53.7 | 73.9 | 70.9 |
| TDAT | 58.1 | 79.8 | 83.0 |
| AiATrack | 58.4 | 82.3 | 78.9 |
| MixFormerV2 | 58.5 | 80.7 | 76.8 |
| DropTrack | 59.0 | 82.3 | 77.8 |
| MixFormer | 59.1 | 81.0 | 77.4 |
| HipTrack | 59.6 | 81.8 | 78.0 |
| FocusTrack | 62.0 | 86.3 | 82.0 |
| UAUTrack | 64.2 | 85.0 | 82.9 |
| MACTrack | 66.5 | 88.9 | 84.1 |
| **STCMTrack (ours)** | **69.2** | **88.9** | **88.4** |

#### Comparison with Global Trackers

| Method | AUC (%) | P (%) |
|---|---:|---:|
| QRDT | 38.9 | 57.4 |
| StrongSiamTracker | 66.7 | - |
| SiamDT | 66.8 | 90.0 |
| FSTC-DiMP | 67.7 | 91.3 |
| MCATrack | 67.8 | 92.5 |
| **STCMTrack (ours)** | **69.2** | 88.9 |

#### Ablation Study

| LTCP | CTR | AUC (%) | P (%) | PNorm (%) |
|---|---|---:|---:|---:|
| ✗ | ✗ | 67.0 | 85.6 | 85.3 |
| ✓ | ✗ | 67.5 | 86.5 | 86.0 |
| ✗ | ✓ | 68.4 | 87.9 | 87.4 |
| ✓ | ✓ | **69.2** | **88.9** | **88.4** |

## Quick Start

### Data Preparation

Put the Anti-UAV410 dataset outside or alongside the project directory. The default portable layout is:

```text
${PROJECT_ROOT}/../antiuav410/
├── train/
├── val/
└── test/
```

Edit `consts.yaml` if your dataset is placed elsewhere:

```yaml
ANTIUAV410_PATH: '../antiuav410/test'
ANTIUAV410_TRAIN_PATH: '../antiuav410/train'
ANTIUAV410_VAL_PATH: '../antiuav410/val'
```

`ANTIUAV_GT_DIR` in `test_stcmtrack.sh` only controls the external metric script. Tracker-side dataset loading still depends on `consts.yaml`.

### Install the environment

Our current verified environment uses Python 3.11.2, PyTorch 2.3.1, torchvision 0.18.1, and CUDA 12.1.

```bash
conda create -n spmtrack python=3.11 -y
conda activate spmtrack
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt
```

The provided scripts default to `CONDA_ENV=spmtrack`. You can override it:

```bash
CONDA_ENV=my_tracking_env DEVICE_IDS=0 ./test_stcmtrack.sh
```

### Model Weights

Checkpoint binaries are not committed to Git.

Required files:

```text
weights/stcmtrack_base.bin
weights/stcmtrack_ltcp.bin
```

See `weights/README.md` for SHA256 values and placement instructions.

```bash
wget -O weights/stcmtrack_base.bin \
  https://github.com/li-jiachen/STCMTrack/releases/download/v0.1.0/stcmtrack_base.bin

LTCP incremental checkpoint is not yet publicly available. After obtaining it, place the file at `weights/stcmtrack_ltcp.bin`.
```

### Evaluation

For a quick sanity check:

```bash
DEVICE_IDS=0 ./test_stcmtrack.sh
```

Default behavior is `EVAL_SCOPE=short`.

For full Anti-UAV410 test evaluation:

```bash
EVAL_SCOPE=full DEVICE_IDS=0 ./test_stcmtrack.sh
```

### Training

```bash
DEVICE_IDS=0 ./train_stcmtrack.sh
```

- Uses `weights/stcmtrack_base.bin` by default.
- Outputs to `output/stcmtrack_train`.
- Training launcher is provided as a reference entry point. Full reproduction of every paper table is not currently claimed.

```bash
BASE_WEIGHT=/path/to/stcmtrack_base.bin \
LTCP_WEIGHT=/path/to/stcmtrack_ltcp.bin \
OUTPUT_DIR=/path/to/train_output \
DEVICE_IDS=0 \
./train_stcmtrack.sh
```

## Evaluation Metrics

- AUC: success AUC over IoU thresholds from 0 to 1.
- P: center precision at 20 pixels.
- PNorm: normalized precision at 0.5.
- The released evaluator computes sequence-level macro average.

```bash
python3 tools/evaluate_antiuav_iou_p20.py <results.zip> \
  --gt-dir ../antiuav410/test \
  --sequence-csv docs/stcmtrack_eval_sequence.csv
```

## Checkpoint Loading Notes

- The base checkpoint and LTCP compatibility checkpoint are loaded sequentially.
- The LTCP compatibility checkpoint only contains incremental parameters.
- Many `missing keys` messages are expected when loading the LTCP checkpoint.
- `unexpected keys` should remain 0.

## Citing STCMTrack

```bibtex
@misc{stcmtrack2026,
  title  = {STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking},
  author = {Li, Jiachen and Yang, Tao and Zhou, Kun and Zhang, Jingyi},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## Acknowledgement

This project is built upon the tracking framework and implementation practices of related visual tracking repositories, including SPMTrack. We thank the authors for their open-source contributions.

## Contact

For questions, please create an issue in this repository.

## License

This project is released under the license specified in `LICENSE`.
