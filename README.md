# STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking

**Status:** manuscript under review.

Official implementation of STCMTrack, including inference, evaluation, the component ablation, two-stage training and the Anti-UAV410 checkpoints.

## Overview

Infrared anti-UAV tracking is difficult because UAVs are often tiny and low-contrast, while background clutter and frame-to-frame errors can cause tracking drift. STCMTrack is a confidence-guided spatio-temporal context modeling framework with two complementary modules:

- **Local-enhanced Temporal Context Propagation (LTCP)** uses global inter-frame correlation and local temporal consistency to selectively propagate reliable historical features.
- **Confidence-Triggered Re-localization (CTR)** compensates inter-frame motion to stabilize the search region (Motion Center Correction, MCC) and triggers residual-guided re-localization when the prediction confidence is low (Residual-Guided Target Correction, RGTC).

### LTCP

LTCP enhances the current search tokens $`\mathbf{X}_t`$ with the historical search-token memory $`\mathcal{M}_{t-1}`$ of the $`k_t \le m`$ most recent frames ($`m = 2`$) and the target state token $`\mathbf{q}_t`$. Softmax weights over frame-level similarities aggregate the memory into a temporal context $`\mathbf{C}_t`$ and give a global confidence $`c^{g}_t`$. The local consistency at each token position (Eq. 1), mapped to $`[0, 1]`$, gives the local confidence $`c^{l}_{t,i}`$, and $`c^{\mathrm{temp}}_{t,i} = c^{g}_t \, c^{l}_{t,i}`$ together with $`\mathbf{q}_t`$ determines a bounded gate (Eq. 2) that fuses $`\mathbf{X}_t`$ with $`\mathbf{C}_t`$ (Eq. 3):

```math
p_{t,i} = \max_{k=1,\ldots,k_t} \big\langle \mathrm{Norm}(\mathbf{X}_{t,i}),\ \mathrm{Norm}(\mathcal{M}_{t-1,k,i}) \big\rangle \qquad (1)
```

```math
g_{t,i} = \sigma\big(\mathbf{W}_g [c^{\mathrm{temp}}_{t,i}; \mathbf{q}_t] + \mathbf{b}_g\big)\, g_{\max} \qquad (2)
```

```math
\widetilde{\mathbf{X}}_{t,i} = (1 - g_{t,i})\, \mathbf{X}_{t,i} + g_{t,i}\, \mathbf{C}_{t,i} \qquad (3)
```

### CTR

CTR estimates one homography $`\mathcal{T}_t`$ per frame from ORB keypoints of adjacent frames (Hamming matching and RANSAC) and reuses it in both branches. MCC maps the previous target center with $`\mathcal{T}_t`$ and, keeping the previous target size, re-centers the search region before inference. When the target-center confidence $`s_t`$ is below $`\tau = 0.40`$, RGTC aligns the previous frame with $`\mathcal{T}_t`$, thresholds the absolute residual adaptively within the valid aligned region (Eqs. 4–6), combines the residual mask with the MOG2 foreground mask ($`M_t = M_t^{\mathrm{mog2}} \cup M_t^{\mathrm{res}}`$), and selects the candidate consistent with the motion-corrected center as the corrected box:

```math
R_t(\mathbf{p}) = \big| I_t(\mathbf{p}) - \widetilde{I}_{t-1}(\mathbf{p}) \big| \qquad (4)
```

```math
T_{\mathrm{res}} = \mathrm{median}(R_v) + \alpha \cdot \mathrm{MAD}(R_v) \qquad (5)
```

```math
M_t^{\mathrm{res}}(\mathbf{p}) = \begin{cases} 1, & R_t(\mathbf{p}) > T_{\mathrm{res}} \\ 0, & \text{otherwise} \end{cases} \qquad (6)
```

MCC and RGTC have no trainable parameters and are applied only at inference.

### Code

- LTCP: `trackit/models/methods/STCMTrack/modules/ltcp.py`
- CTR (MCC and RGTC): `trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/one_stream/ctr.py`
- Tracking pipeline: `trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/one_stream/__init__.py`
- Metrics: `tools/evaluate_antiuav_iou_p20.py`

## Results

Numbers reported in the paper (%).

### Comparison with local trackers on Anti-UAV410 and Anti-UAV

Best results are in **bold**; second-best results are <ins>underlined</ins>. † Our baseline.

| Method | Anti-UAV410 AUC | P@20 | P<sub>n</sub> | Anti-UAV AUC | P@20 | P<sub>n</sub> |
|---|---:|---:|---:|---:|---:|---:|
| OSTrack | 53.7 | 73.9 | 70.9 | 59.2 | 79.4 | 77.5 |
| ROMTrack | 54.7 | 74.5 | 71.7 | 59.4 | 78.9 | 77.1 |
| ZoomTrack | 58.4 | 81.2 | 77.4 | 63.5 | 86.0 | 83.4 |
| DropTrack | 59.0 | 82.3 | 77.8 | 64.2 | 85.8 | 83.2 |
| FocusTrack | 62.8 | <ins>86.3</ins> | 82.8 | 67.7 | <ins>90.9</ins> | 88.4 |
| UAUTrack | 64.2 | 85.0 | 82.9 | <ins>68.8</ins> | 89.7 | <ins>89.0</ins> |
| SPMTrack† | <ins>67.0</ins> | 85.6 | <ins>85.3</ins> | 68.3 | 87.1 | 86.9 |
| **STCMTrack (ours)** | **69.2** | **88.9** | **88.4** | **70.7** | **91.3** | **90.1** |

### Comparison with global trackers on the Anti-UAV410 test set

Best results are in **bold**; second-best results are <ins>underlined</ins>; –: not available.

| Method | AUC | P@20 |
|---|---:|---:|
| QRDT | 38.9 | 57.4 |
| StrongSiamTracker | 66.7 | – |
| SiamDT | 66.8 | 90.0 |
| FSTC-DiMP | 67.7 | <ins>91.3</ins> |
| MCATrack | <ins>67.8</ins> | **92.5** |
| **STCMTrack (ours)** | **69.2** | 88.9 |

### Component ablation on the Anti-UAV410 test set

MCC and RGTC are the two branches of CTR. Best results are in **bold**. The last column is the `VARIANT` value that runs the row with `test_stcmtrack.sh`.

| # | LTCP | MCC | RGTC | AUC | P@20 | P<sub>n</sub> | `VARIANT` |
|---|:---:|:---:|:---:|---:|---:|---:|---|
| 1 | × | × | × | 67.0 | 85.6 | 85.3 | `baseline` |
| 2 | ✓ | × | × | 67.8 | 86.5 | 86.1 | `ltcp` |
| 3 | × | ✓ | × | 67.4 | 85.9 | 85.8 | `mcc` |
| 4 | × | × | ✓ | 68.2 | 87.1 | 86.9 | `rgtc` |
| 5 | ✓ | ✓ | × | 68.0 | 86.8 | 86.7 | `ltcp_mcc` |
| 6 | ✓ | × | ✓ | 68.7 | 88.1 | 87.7 | `ltcp_rgtc` |
| 7 | × | ✓ | ✓ | 68.4 | 87.5 | 87.6 | `mcc_rgtc` |
| 8 | ✓ | ✓ | ✓ | **69.2** | **88.9** | **88.4** | `full` |

## Installation

Tested with Python 3.11, PyTorch 2.3.1, torchvision 0.18.1 and CUDA 12.1 on an NVIDIA RTX 4090.

```bash
conda create -n stcmtrack python=3.11 -y
conda activate stcmtrack
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt
```

The DINOv2 ViT-B/14 weights are downloaded automatically on first use. The scripts activate the conda environment `stcmtrack` by default; override it with `CONDA_ENV=<name>`.

## Data Preparation

**Anti-UAV410** (official split: 200 training, 90 validation and 120 test sequences). Default layout (edit `consts.yaml` to use other locations):

```text
${PROJECT_ROOT}/../antiuav410/
├── train/
├── val/
└── test/
    └── <sequence>/000001.jpg ... IR_label.json
```

`tools/prepare_antiuav410.py --source-root <raw> --output-root ../antiuav410` links or copies an existing Anti-UAV410 copy into this layout.

**Anti-UAV** (infrared videos only; `antiuav300` in the scripts) is converted to the same layout:

```bash
python tools/prepare_antiuav.py --source-root /path/to/Anti-UAV --output-root ../antiuav300_ir
```

## Model Weights

Checkpoints are distributed through the GitHub Release `v1.0.0` (SHA256 values in [weights/README.md](weights/README.md)):

```bash
wget -O weights/stcmtrack_base.bin \
  https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_base.bin
wget -O weights/stcmtrack_ltcp.bin \
  https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_ltcp.bin
```

- `stcmtrack_base.bin`: tracking network trained on Anti-UAV410 (the baseline, row 1 of the component ablation).
- `stcmtrack_ltcp.bin`: LTCP parameters, loaded after the base checkpoint.

## Evaluation

```bash
# quick sanity check (first 4 sequences, 200 frames each)
DEVICE_IDS=0 ./test_stcmtrack.sh

# full Anti-UAV410 test set, complete model
EVAL_SCOPE=full DEVICE_IDS=0 ./test_stcmtrack.sh

# a row of the component ablation, e.g. row 4 (RGTC only)
EVAL_SCOPE=full VARIANT=rgtc DEVICE_IDS=0 ./test_stcmtrack.sh

# Anti-UAV test set
EVAL_SCOPE=full DATASET=antiuav300 DEVICE_IDS=0 ./test_stcmtrack.sh
```

At the end of each run, `tools/evaluate_antiuav_iou_p20.py` prints the metrics used in the paper:

- **AUC**: area under the success curve obtained by sweeping the IoU threshold over [0, 1];
- **P@20**: fraction of frames whose center error is within 20 pixels;
- **P<sub>n</sub>**: fraction of frames whose center error, normalized by the ground-truth box size, is within 0.5.

## Training

On each training set, the tracking network is trained for 80 epochs, followed by 20 epochs of LTCP training with all other parameters frozen:

```bash
# Stage 1: tracking network, 80 epochs
TRAIN_STAGE=1 DEVICE_IDS=0 ./train_stcmtrack.sh

# Stage 2: LTCP, 20 epochs with all other parameters frozen, initialized from stage 1
TRAIN_STAGE=2 BASE_WEIGHT=output/stcmtrack_train_antiuav410_stage1/<run_id>/checkpoint/epoch_79/model.bin \
  DEVICE_IDS=0 ./train_stcmtrack.sh
```

Add `DATASET=antiuav300` to train on Anti-UAV. The backbone is a DINOv2-pretrained ViT-B/14; the template and search inputs are 196 × 196 and 378 × 378. Center prediction and box regression use equally weighted binary cross-entropy and GIoU losses, optimized with AdamW (initial learning rate 1e-4, weight decay 0.1) and a cosine scheduler. The LTCP memory size is m = 2 and the CTR correction threshold is τ = 0.40.

## Citation

```bibtex
@misc{stcmtrack2026,
  title  = {STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking},
  author = {Li, Jiachen and Yang, Tao and Zhou, Kun and Zhang, Jingyi},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## Acknowledgement

We thank the authors of DINOv2, SPMTrack, Anti-UAV410 and Anti-UAV for making their models, code and datasets publicly available.

## License

This project is released under the Apache License 2.0 (see `LICENSE`).
