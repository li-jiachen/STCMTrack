# Installation and Usage

## Environment

```bash
git clone https://github.com/li-jiachen/STCMTrack.git
cd STCMTrack
conda create -n stcmtrack python=3.10 -y
conda activate stcmtrack
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
chmod +x boot.sh train_stcmtrack.sh test_stcmtrack.sh
```

The pretrained DINOv2 backbone weights are downloaded automatically on the first run.

## Datasets

Use the official dataset splits. Anti-UAV410 contains 200 training, 90 validation, and 120 test sequences.

```text
antiuav410/
  train/<sequence>/
  val/<sequence>/
  test/<sequence>/
```

Each sequence contains image frames and `IR_label.json`. The annotation fields are `gt_rect` and `exist`, and bounding boxes use the `x, y, width, height` format.

If your data already follows this structure, use it directly. Otherwise, run the conversion command for the corresponding dataset. The output directory must be a new directory:

```bash
python tools/prepare_antiuav410.py --source-root /data/Anti-UAV410 \
  --output-root ../antiuav410 --mode symlink --write-manifest

python tools/prepare_antiuav.py --source-root /data/Anti-UAV \
  --output-root ../antiuav300_ir --gt-format xywh
```

To copy files instead of creating symbolic links, replace `--mode symlink` with `--mode copy`. If the Anti-UAV annotations use corner coordinates, replace `--gt-format xywh` with `--gt-format xyxy`.

Set the paths for the datasets you use in `consts.yaml`:

```yaml
ANTIUAV410_TRAIN_PATH: '/data/antiuav410/train'
ANTIUAV410_VAL_PATH: '/data/antiuav410/val'
ANTIUAV410_PATH: '/data/antiuav410/test'
ANTIUAV300_TRAIN_PATH: '/data/antiuav300_ir/train'
ANTIUAV300_VAL_PATH: '/data/antiuav300_ir/val'
ANTIUAV300_TEST_PATH: '/data/antiuav300_ir/test'
```

The test dataset path can also be specified with `ANTIUAV_GT_DIR`.

## Training and Evaluation

```bash
TRAIN_STAGE=1 ./train_stcmtrack.sh
TRAIN_STAGE=2 BASE_WEIGHT=/path/to/model.bin ./train_stcmtrack.sh
./test_stcmtrack.sh
```

For the second stage, set `BASE_WEIGHT` to the checkpoint saved in the first stage. To use Anti-UAV, prefix the commands with `DATASET=antiuav300`.

The test entry point runs the complete STCMTrack model with LTCP and CTR enabled. To evaluate custom weights, specify both `BASE_WEIGHT` and `LTCP_WEIGHT`.

```bash
BASE_WEIGHT=/path/to/stcmtrack/base.bin \
  LTCP_WEIGHT=/path/to/stcmtrack/ltcp.bin ./test_stcmtrack.sh
```
