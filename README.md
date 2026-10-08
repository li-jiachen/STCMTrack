# STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking

Official PyTorch implementation of STCMTrack.

**Jiachen Li<sup>1</sup>, Tao Yang<sup>1,\*</sup>, Kun Zhou<sup>1,\*</sup>, Jingyi Zhang<sup>2</sup>**

<sup>1</sup> School of Computer Science, China West Normal University, Nanchong, China<br>
<sup>2</sup> Chongqing Vocational Institute of Engineering, Chongqing, China<br>
<sup>\*</sup> Corresponding authors

This repository contains the tracker, the two-stage training code, the evaluation code for Anti-UAV410 and Anti-UAV, the eight configurations of the component ablation, and the SPMTrack baseline.

## Overview

Infrared anti-UAV tracking is difficult because UAVs are often tiny and low-contrast, while background clutter and frame-to-frame errors can cause tracking drift. Global temporal modeling can introduce background interference, and search regions based on previous predictions are prone to error accumulation. STCMTrack is a confidence-guided spatio-temporal context modeling framework with two complementary modules:

- **Local-enhanced Temporal Context Propagation (LTCP)** uses global inter-frame correlation and local temporal consistency to selectively propagate reliable historical features.
- **Confidence-Triggered Re-localization (CTR)** compensates inter-frame motion to stabilize the search region (Motion Center Correction, MCC) and triggers residual-guided re-localization when the prediction confidence is low (Residual-Guided Target Correction, RGTC).

## Method

Section, equation and table numbers refer to the paper.

### Framework (Sec. 2.1)

For each frame $`I_t`$, the tracker runs the following steps:

1. A search region $`S_t`$ is generated from $`I_t`$ and the preceding target states, and MCC re-centers it to obtain $`S_t^{\mathrm{corr}}`$.
2. A single Transformer (DINOv2 ViT-B/14) jointly encodes this crop and the first-frame template $`z`$, producing the search tokens $`\mathbf{X}_t`$ and a target state token $`\mathbf{q}_t`$ aggregated from the joint features.
3. LTCP combines these tokens with the historical search-token memory $`\mathcal{M}_{t-1}`$ to produce the enhanced features $`\widetilde{\mathbf{X}}_t`$.
4. The prediction head outputs a center response map and the box regression, yielding $`B_t^{\mathrm{tracker}}`$ and the confidence $`s_t \in [0, 1]`$ measured at the predicted center.
5. The final box is $`B_t = B_t^{\mathrm{tracker}}`$ when $`s_t \ge \tau`$; otherwise RGTC produces $`B_t = B_t^{\mathrm{corr}}`$ from residuals and motion priors.
6. The final box updates the target state and the next search region, and $`\mathbf{X}_t`$ is stored in the historical memory.

### LTCP (Sec. 2.2)

LTCP enhances the current search tokens $`\mathbf{X}_t \in \mathbb{R}^{N \times D}`$ with the memory $`\mathcal{M}_{t-1} \in \mathbb{R}^{k_t \times N \times D}`$ of the $`k_t \le m`$ most recent frames and the target state token $`\mathbf{q}_t \in \mathbb{R}^{1 \times D}`$. The memory is updated after each forward pass.

**Frame-level correlation.** The search tokens of each frame are mean-pooled. Normalized dot-product similarities between the current and the historical pooled features are turned into frame weights with a softmax, and the weights aggregate the historical tokens into the context $`\mathbf{C}_t \in \mathbb{R}^{N \times D}`$. The frame-level similarity, mapped to $`[0, 1]`$, gives the global confidence $`c^{g}_t`$.

**Local consistency.** Since global similarity can overlook tiny target regions, LTCP measures the consistency at each token position:

```math
p_{t,i} = \max_{k=1,\ldots,k_t} \big\langle \mathrm{Norm}(\mathbf{X}_{t,i}),\ \mathrm{Norm}(\mathcal{M}_{t-1,k,i}) \big\rangle \qquad (1)
```

where $`k`$ indexes the historical frames and $`\mathrm{Norm}`$ normalizes the features, so the dot product is a cosine similarity. Mapping $`p_{t,i}`$ from $`[-1, 1]`$ to $`[0, 1]`$ gives the local confidence $`c^{l}_{t,i}`$, and $`c^{\mathrm{temp}}_{t,i} = c^{g}_t\, c^{l}_{t,i}`$.

**Gate and fusion.** The broadcast target state token and the temporal confidence jointly determine a bounded gate, which fuses the current tokens with the context:

```math
g_{t,i} = \sigma\big(\mathbf{W}_g [c^{\mathrm{temp}}_{t,i}; \mathbf{q}_t] + \mathbf{b}_g\big)\, g_{\max} \qquad (2)
```

```math
\widetilde{\mathbf{X}}_{t,i} = (1 - g_{t,i})\, \mathbf{X}_{t,i} + g_{t,i}\, \mathbf{C}_{t,i} \qquad (3)
```

where $`[\cdot\,;\cdot]`$ denotes concatenation, $`\sigma`$ is the sigmoid function, $`\mathbf{W}_g, \mathbf{b}_g`$ are learnable parameters and $`g_{\max}`$ bounds the context injection. The fused tokens keep the original dimensions and enter the prediction head.

### CTR (Sec. 2.3)

CTR estimates one homography $`\mathcal{T}_t`$ per frame and reuses it to stabilize the search crop and to compute the residuals.

**Motion Center Correction (MCC), before inference.** ORB keypoints of the adjacent frames are matched with the Hamming distance, and RANSAC estimates $`\mathcal{T}_t`$ with a reprojection inlier threshold of 2.0 pixels. The previous target center $`\mathbf{o}_{t-1} = (x_{t-1}, y_{t-1})`$ is mapped to $`\hat{\mathbf{o}}_t = \mathcal{T}_t(\mathbf{o}_{t-1})`$. Retaining the previous target size gives $`B_t^{\mathrm{geo}} = (\hat{x}_t, \hat{y}_t, w_{t-1}, h_{t-1})`$, which defines the reference center and scale of the corrected search crop $`S_t^{\mathrm{corr}}`$.

**Residual-Guided Target Correction (RGTC), when the confidence is low.** If $`s_t \ge \tau`$ with $`\tau = 0.40`$, the prediction is accepted directly. Otherwise, the previous grayscale frame is aligned to the current frame with the same homography, giving $`\widetilde{I}_{t-1}`$, and the absolute residual is thresholded adaptively within the valid aligned region:

```math
R_t(\mathbf{p}) = \big| I_t(\mathbf{p}) - \widetilde{I}_{t-1}(\mathbf{p}) \big| \qquad (4)
```

```math
T_{\mathrm{res}} = \mathrm{median}(R_v) + \alpha \cdot \mathrm{MAD}(R_v) \qquad (5)
```

```math
M_t^{\mathrm{res}}(\mathbf{p}) = \begin{cases} 1, & R_t(\mathbf{p}) > T_{\mathrm{res}} \\ 0, & \text{otherwise} \end{cases} \qquad (6)
```

where $`R_v`$ contains the residual values in that region, $`\mathrm{MAD}(\cdot)`$ is the median absolute deviation and $`\alpha`$ is a fixed scale coefficient. The residual mask is combined with the MOG2 foreground mask, $`M_t = M_t^{\mathrm{mog2}} \cup M_t^{\mathrm{res}}`$. External contours of $`M_t`$ are converted into bounding rectangles to form the candidate set, area constraints based on the previous target scale remove isolated noise and large background regions, and among the remaining candidates the box consistent with the motion-corrected center is selected as $`B_t^{\mathrm{corr}}`$.

MCC and RGTC have no trainable parameters and are applied only at inference. RGTC runs after the current-frame inference, so it does not alter the LTCP features already computed for that frame; its effect is carried forward through the subsequent search regions and memory updates.

### Code map

- Model: `trackit/models/methods/STCMTrack/`
  - `STCMTrack.py`: joint encoder, target state token and prediction heads
  - `STCMTrack_inference.py`: frame-by-frame inference with the per-sequence LTCP memory
  - `modules/ltcp.py`: LTCP, Eqs. (1)-(3)
- Tracking pipeline: `trackit/runner/evaluation/distributed/tracker_evaluator/`
  - `default/pipelines/one_stream/__init__.py`: tracking loop
  - `default/pipelines/one_stream/ctr.py`: CTR (MCC and RGTC), Eqs. (4)-(6)
  - `components/post_process/box_with_score_map.py`: predicted box and confidence $`s_t`$
- Losses: `trackit/criteria/methods/box_with_score_map/__init__.py`
- Metrics: `trackit/core/evaluation/antiuav.py`, `tools/evaluate_antiuav_iou_p20.py`
- Configuration: `config/STCMTrack/` (`run.yaml`, `dinov2/config.yaml`, `_mixin/*.yaml`)
- SPMTrack baseline: `trackit/models/methods/SPMTrack/`, `config/SPMTrack/`

## Results

All numbers are in %.

### Comparison with local trackers on Anti-UAV410 and Anti-UAV (Table 1)

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

### Component ablation on the Anti-UAV410 test set (Table 2)

MCC and RGTC are the two branches of CTR. Row 1 corresponds to the SPMTrack baseline in Table 1; row 8 is the complete STCMTrack model. The last column is the `VARIANT` value that runs the row with `test_stcmtrack.sh`.

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

### Comparison with global trackers on the Anti-UAV410 test set (Table 3)

Best results are in **bold**; second-best results are <ins>underlined</ins>; –: not available.

| Method | AUC | P@20 |
|---|---:|---:|
| QRDT | 38.9 | 57.4 |
| StrongSiamTracker | 66.7 | – |
| SiamDT | 66.8 | 90.0 |
| FSTC-DiMP | 67.7 | <ins>91.3</ins> |
| MCATrack | <ins>67.8</ins> | **92.5** |
| **STCMTrack (ours)** | **69.2** | 88.9 |

## Installation

Linux, Python 3.11 or 3.12, PyTorch 2.3.1 and torchvision 0.18.1. All experiments run on an NVIDIA RTX 4090 GPU.

```bash
conda create -n stcmtrack python=3.11 -y
conda activate stcmtrack
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt
```

The DINOv2 ViT-B/14 weights are downloaded through Torch Hub the first time the backbone is built; on an offline machine, prepare the Torch Hub cache in advance. The training and evaluation scripts require a CUDA GPU. They activate the conda environment `stcmtrack` by default; use `CONDA_ENV` and `CONDA_SH` to select another environment.

## Data preparation

Set the six dataset paths in `consts.yaml` (relative paths are resolved from the repository root; the defaults are `../antiuav410/` and `../antiuav300_ir/`). Both benchmarks use the same layout:

```text
<root>/{train,val,test}/<sequence>/000001.jpg, 000002.jpg, ...
<root>/{train,val,test}/<sequence>/IR_label.json
```

`IR_label.json` has the form `{"exist": [...], "gt_rect": [[x, y, w, h], ...]}`. `exist` and `gt_rect` have one entry per frame, and `gt_rect` uses the top-left corner, width and height.

**Anti-UAV410** (official split: 200 training, 90 validation and 120 test sequences; the loader checks these counts):

```bash
python tools/prepare_antiuav410.py --source-root /path/to/Anti-UAV410 \
  --output-root ../antiuav410 --write-manifest
```

**Anti-UAV** (infrared videos; named `antiuav300` in the scripts):

```bash
python tools/prepare_antiuav.py --source-root /path/to/Anti-UAV --output-root ../antiuav300_ir
```

Data preparation and loading fail explicitly on missing frames, annotation length mismatches and invalid coordinates; nothing is truncated silently. The first frame of every sequence must contain the target, because it provides the template. The dataset cache in `trackit/datasets/cache/` is keyed by the absolute split path, the sequence set and the annotation contents; delete it manually if images are replaced without changing the annotations.

## Model weights

The trained weights are available on the [Releases](https://github.com/li-jiachen/STCMTrack/releases) page; they are not stored in Git. Download them into `weights/`. `test_stcmtrack.sh` reads the following files by default:

| Setting | Files in `weights/` |
|---|---|
| STCMTrack on Anti-UAV410 (default) | `stcmtrack_base.safetensors`, `stcmtrack_ltcp.safetensors` |
| STCMTrack on Anti-UAV (`DATASET=antiuav300`) | `stcmtrack_antiuav300_base.safetensors`, `stcmtrack_antiuav300_ltcp.safetensors` |
| SPMTrack baseline on Anti-UAV410 (`VARIANT=baseline`) | `spmtrack_baseline.safetensors` |
| SPMTrack baseline on Anti-UAV (`VARIANT=baseline DATASET=antiuav300`) | `spmtrack_antiuav300_baseline.safetensors` |

A base file contains the tracking network and an LTCP file contains the LTCP gate. The base file is loaded first and the LTCP file second; variants without LTCP use the base file only. `test_stcmtrack.sh` checks before the evaluation that the two files were exported from the same training snapshot. Weights obtained with the training commands below use the same names, and other locations can be passed with `BASE_WEIGHT` and `LTCP_WEIGHT`. See [weights/README.md](weights/README.md).

## Evaluation

```bash
# STCMTrack on the Anti-UAV410 test set
DEVICE_IDS=0 ./test_stcmtrack.sh

# STCMTrack on the Anti-UAV test set
DATASET=antiuav300 DEVICE_IDS=0 ./test_stcmtrack.sh

# One row of the component ablation (here: Table 2, row 4)
VARIANT=rgtc DEVICE_IDS=0 ./test_stcmtrack.sh

# Quick run: the first 4 sequences, at most 200 frames each
EVAL_SCOPE=short DEVICE_IDS=0 ./test_stcmtrack.sh
```

The default is the full test set. `EVAL_SCOPE=short` is a quick run for checking the setup; its report is marked as partial.

| `VARIANT` | Model | LTCP | MCC | RGTC | Table 2 row |
|---|---|:---:|:---:|:---:|:---:|
| `baseline` (alias `spmtrack`) | SPMTrack | × | × | × | 1 |
| `ltcp` | STCMTrack | ✓ | × | × | 2 |
| `mcc` | STCMTrack | × | ✓ | × | 3 |
| `rgtc` | STCMTrack | × | × | ✓ | 4 |
| `ltcp_mcc` | STCMTrack | ✓ | ✓ | × | 5 |
| `ltcp_rgtc` | STCMTrack | ✓ | × | ✓ | 6 |
| `mcc_rgtc` | STCMTrack | × | ✓ | ✓ | 7 |
| `full` (default) | STCMTrack | ✓ | ✓ | ✓ | 8 |
| `stcm_base` | STCMTrack | × | × | × | – |

`stcm_base` is the STCMTrack network with all three components switched off; it is provided for convenience and is not a row of Table 2. With MCC switched off, the search crop is not re-centered, while RGTC still estimates the homography for frame alignment and the motion-corrected reference center. `python tools/check_variant_mapping.py` prints this mapping as derived from the script, the configuration files and the model builders.

Paths can be set with `BASE_WEIGHT`, `LTCP_WEIGHT`, `ANTIUAV_GT_DIR` and `OUTPUT_DIR`. `ANTIUAV_GT_DIR` applies to both the tracking data source and the metric computation. Each run writes to `output/stcmtrack_test_<dataset>_<variant>_<scope>/<time>/`: the predictions (`results.zip`, in the `eval/epoch_0/` sub-directory of the run), a per-sequence CSV file (`*_sequence.csv`) and a JSON report (`*_metrics.json`) with the protocol, the sequence coverage and the SHA-256 of the predictions and the ground truth. A full evaluation requires exactly matching sequence names and frame counts; missing, duplicated or malformed entries are errors.

The metrics of an exported result file can be recomputed separately:

```bash
python tools/evaluate_antiuav_iou_p20.py /path/to/results.zip \
  --gt-dir /path/to/test --expected-sequences 120 \
  --sequence-csv sequence_metrics.csv --report-json metrics.json
```

### Metrics (Sec. 3.2)

- **AUC**: area under the success curve obtained by sweeping the IoU threshold over [0, 1].
- **P@20**: fraction of frames whose center error is below 20 pixels.
- **P<sub>n</sub>**: fraction of frames whose center error normalized by the ground-truth box diagonal is below 0.5.

Frames in which the target is absent (`exist = 0`) are excluded, the initialization frame is included, and the reported numbers are unweighted means over the sequences. All evaluation entry points share one implementation (`trackit/core/evaluation/antiuav.py`). The scripts print the three metrics as fractions in [0, 1]: as `success_score`, `precision_score` and `norm_precision_score` during the run, and as `AUC`, `P@20` and `NP@0.5` in the final report.

## Training

On each training set, the tracking network is trained for 80 epochs, followed by 20 epochs of LTCP training with all other parameters frozen (Sec. 3.1):

```bash
# Stage 1: tracking network, 80 epochs
TRAIN_STAGE=1 DEVICE_IDS=0 ./train_stcmtrack.sh

# Stage 2: LTCP, 20 epochs, all other parameters frozen; starts from the stage-1 checkpoint
TRAIN_STAGE=2 DEVICE_IDS=0 BASE_WEIGHT=<stage1_run_dir>/checkpoint/epoch_79/model.bin \
  ./train_stcmtrack.sh
```

Each training run creates `output/stcmtrack_train_<dataset>_stage<n>/<run_id>/`, where `<run_id>` is composed of the configuration name and the start time. This directory contains the log (`train_stdout.log`) and a second directory `<run_id>/` with the checkpoints (`checkpoint/epoch_<n>/model.bin`). `<stage1_run_dir>` stands for that inner directory of the stage-1 run; the run prints it as `output directory: ...` when it starts.

Add `DATASET=antiuav300` to both commands to train on Anti-UAV. Each benchmark is trained separately; do not mix the base and LTCP weights of the two datasets. Training snapshots are named `model.bin`, are stored as Safetensors and contain the full model in both stages.

Export the weight pair from the stage-2 snapshot (`<stage2_run_dir>` is the inner run directory of the stage-2 run), so that both files come from the same training result:

```bash
python tools/export_stcmtrack_weights.py <stage2_run_dir>/checkpoint/epoch_19/model.bin \
  --base-output weights/stcmtrack_base.safetensors \
  --ltcp-output weights/stcmtrack_ltcp.safetensors
```

| Setting | Value |
|---|---|
| Backbone | DINOv2-pretrained ViT-B/14 (12 blocks, D = 768) |
| Inputs | template 196 × 196, search region 378 × 378 |
| Stage 1 | 80 epochs; the tracking network is trained with the DINOv2 weights frozen and adapted by TMoE (r = 64, 4 experts); TMoE adapters, target-state query, token-type embeddings and prediction heads are optimized |
| Stage 2 | 20 epochs; only the LTCP gate is optimized, all other parameters are frozen |
| Training sample | the first-frame template and three consecutive search frames, so that LTCP sees 0, 1 and 2 memory frames |
| Loss | binary cross-entropy for center prediction and GIoU for box regression, equally weighted, averaged over the three search frames |
| Optimizer | AdamW, initial learning rate 1e-4, weight decay 0.1, cosine schedule |
| LTCP memory size | m = 2 |
| CTR correction threshold | τ = 0.40 |

All remaining settings (batch size, augmentation, LTCP gate bound, homography and foreground-mask parameters, fallback rules) are listed in [docs/IMPLEMENTATION_DETAILS.md](docs/IMPLEMENTATION_DETAILS.md).

## SPMTrack baseline

The baseline (SPMTrack† in Table 1, row 1 of Table 2) is built from the official SPMTrack implementation. It is trained on the same training sets with the stage-1 budget (80 epochs) and evaluated with the same metrics:

```bash
DEVICE_IDS=0 ./train_spmtrack.sh                     # 80 epochs on Anti-UAV410
cp <run_dir>/checkpoint/epoch_79/model.bin weights/spmtrack_baseline.safetensors
VARIANT=baseline DEVICE_IDS=0 ./test_stcmtrack.sh    # row 1 of Table 2
```

`<run_dir>` is the output directory printed by the training run (`output/spmtrack_train_antiuav410/<run_id>/<run_id>`). The provenance, structure and checkpoint rules of the baseline are described in [docs/SPMTRACK_BASELINE.md](docs/SPMTRACK_BASELINE.md).

## Tests

```bash
python -m unittest discover -s tests -v
```

The unit and integration tests use a small backbone and synthetic images and run on the CPU in a few seconds. They cover Eqs. (1)-(6), the equivalence of the three-frame training forward and frame-by-frame inference, parameter freezing in stage 2, checkpoint export and loading, the preprocessing and loss, a 70-frame tracking run through the evaluation pipeline, and the metric definitions. Three further checks are available as scripts:

```bash
python tools/validate_full_model.py                     # forward pass of the full-size model
python tools/evaluate_antiuav_iou_p20.py --self-test    # metric definitions
python tools/check_variant_mapping.py --check           # components of every VARIANT
```

## Citation

```bibtex
@misc{li2026stcmtrack,
  title  = {STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling
            for Robust Anti-UAV Tracking},
  author = {Li, Jiachen and Yang, Tao and Zhou, Kun and Zhang, Jingyi},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Acknowledgement

This code base is built on [SPMTrack](https://github.com/WenRuiCai/SPMTrack). We thank the authors of DINOv2, SPMTrack, Anti-UAV410 and Anti-UAV for making their models, code and datasets publicly available.

## License

This project is released under the Apache License 2.0 (see `LICENSE`). The files under `trackit/models/methods/SPMTrack/` and the related pipeline and configuration files are derived from [WenRuiCai/SPMTrack](https://github.com/WenRuiCai/SPMTrack) (Apache-2.0) at commit `c581fe27231f3e16c38578e47daddadfaf6ffd7d`; their origin and the changes made to them are recorded in `trackit/models/methods/SPMTrack/UPSTREAM.json`.
