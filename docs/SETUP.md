# Setup

## Code and environment

```bash
git clone --branch main https://github.com/li-jiachen/STCMTrack.git
cd STCMTrack
conda create -n stcmtrack python=3.10 -y
conda activate stcmtrack
```

Use Python 3.10, [PyTorch 2.4.1](https://pytorch.org/get-started/previous-versions/) and torchvision 0.19.1 (CUDA 12.1).

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
chmod +x boot.sh train_stcmtrack.sh test_stcmtrack.sh
```

Use the **main** branch with the `stcmtrack` Conda environment; the official DINOv2 ViT-B/14 pretrained backbone is downloaded automatically on first use.

## Data

Obtain the benchmark data separately and keep the official splits. Anti-UAV410 uses 200 training, 90 validation and 120 test sequences. The expected layout is:

```text
antiuav410/
  train/<sequence>/000001.jpg, 000002.jpg, ..., IR_label.json
  val/<sequence>/000001.jpg, 000002.jpg, ..., IR_label.json
  test/<sequence>/000001.jpg, 000002.jpg, ..., IR_label.json
```

`IR_label.json` contains matching `gt_rect` and `exist` lists; boxes are top-left `x, y, width, height`. If the extracted Anti-UAV410 data already have this layout, point `consts.yaml` to those splits directly. Otherwise, prepare a new output directory from a root containing `train`, `val` and `test`:

```bash
python tools/prepare_antiuav410.py --source-root /data/Anti-UAV410 \
  --output-root ../antiuav410 --mode symlink --write-manifest
```

Use `--mode copy` if symbolic links are unsuitable. The output directory must be new. This tool checks frame/annotation correspondence and organizes existing frames; it does not create or redistribute the official split.

For the separate Anti-UAV infrared-video benchmark (`DATASET=antiuav300` in the scripts), convert its official splits:

```bash
python tools/prepare_antiuav.py --source-root /data/Anti-UAV \
  --output-root ../antiuav300_ir --gt-format xywh
```

Each source sequence must contain `infrared.json` or `IR_label.json`, plus `infrared.mp4`/`IR.mp4` or extracted frames. Use `--gt-format xyxy` only if the source boxes are corner coordinates.

Set these entries in the repository's `consts.yaml` to your prepared split directories; absolute paths are recommended:

```yaml
ANTIUAV410_TRAIN_PATH: '/data/antiuav410/train'
ANTIUAV410_VAL_PATH: '/data/antiuav410/val'
ANTIUAV410_PATH: '/data/antiuav410/test'
ANTIUAV300_TRAIN_PATH: '/data/antiuav300_ir/train'
ANTIUAV300_VAL_PATH: '/data/antiuav300_ir/val'
ANTIUAV300_TEST_PATH: '/data/antiuav300_ir/test'
```

Only the benchmark you run needs to be present. Evaluation reads the test directory from these constants; `ANTIUAV_GT_DIR` can explicitly override it for both prediction loading and metrics.

## Training and evaluation

Use [train_stcmtrack.sh](../train_stcmtrack.sh) for two-stage STCMTrack training and [test_stcmtrack.sh](../test_stcmtrack.sh) for evaluation; the script headers list the commands and supported options. New exports use `weights/retrained/` and explicit `BASE_WEIGHT`/`LTCP_WEIGHT` paths. The released weights are for Anti-UAV410; Anti-UAV requires separately trained checkpoints.
