# Running

Install the environment and prepare the official data splits as described in [Setup](SETUP.md). These scripts require Bash, Conda and a working CUDA/PyTorch installation. Read [Implementation notes](IMPLEMENTATION_NOTES.md) before using released or independently supplied checkpoints, including the released base-weight validation limitation.

## STCMTrack training

Stage 1 trains the tracking network for 80 epochs using the DINOv2-pretrained ViT-B/14 backbone:

```bash
TRAIN_STAGE=1 DEVICE_IDS=0 ./train_stcmtrack.sh
```

Stage 2 initializes from a stage-1 checkpoint and trains LTCP for 20 epochs with all other parameters frozen:

```bash
TRAIN_STAGE=2 BASE_WEIGHT=/path/to/stage1/checkpoint/epoch_79/model.bin DEVICE_IDS=0 ./train_stcmtrack.sh
```

Stage 2 validates the supplied base before CUDA setup and data loading. It saves full model snapshots. Export the base and LTCP pair explicitly from a stage-2 snapshot using the export tool's command-line options:

```bash
python tools/export_stcmtrack_weights.py /path/to/stage2/model.bin \
  --base-output weights/retrained/stcmtrack_base.bin \
  --ltcp-output weights/retrained/stcmtrack_ltcp.bin
```

Outputs must be new paths. A stage-2 export includes its frozen base so the exported pair comes from the same snapshot. Use explicit paths for evaluation.

Set `DATASET=antiuav300` to train on the separate Anti-UAV infrared benchmark. Train the datasets separately and use the corresponding checkpoints. Arguments following `./train_stcmtrack.sh` are passed to `boot.sh`.

## Independent SPMTrack training

```bash
DEVICE_IDS=0 ./train_spmtrack.sh
DATASET=antiuav300 DEVICE_IDS=0 ./train_spmtrack.sh
```

SPMTrack uses a single training stage, three templates, two search frames and query propagation between the search frames. The shared public budget is 80 epochs, global batch 4 and 2048 training samples per epoch, rather than the upstream budget. The resulting trainable-parameter checkpoint includes `_spmtrack_port_version`. Arguments following `./train_spmtrack.sh` are passed to `boot.sh`.

## Evaluation

```bash
BASE_WEIGHT=/path/to/valid/stcmtrack_base.bin LTCP_WEIGHT=/path/to/matching/stcmtrack_ltcp.bin DEVICE_IDS=0 ./test_stcmtrack.sh
VARIANT=baseline BASE_WEIGHT=/path/to/spmtrack.safetensors DEVICE_IDS=0 ./test_stcmtrack.sh
VARIANT=mcc_rgtc BASE_WEIGHT=/path/to/valid/stcmtrack_base.bin DEVICE_IDS=0 ./test_stcmtrack.sh
DATASET=antiuav300 BASE_WEIGHT=/path/to/antiuav300/base.bin LTCP_WEIGHT=/path/to/antiuav300/ltcp.bin DEVICE_IDS=0 ./test_stcmtrack.sh
EVAL_SCOPE=short BASE_WEIGHT=/path/to/valid/stcmtrack_base.bin LTCP_WEIGHT=/path/to/matching/stcmtrack_ltcp.bin DEVICE_IDS=0 ./test_stcmtrack.sh
```

`EVAL_SCOPE=short` selects four sequences with at most 200 frames each. It is a partial evaluation. Full Anti-UAV410 evaluation expects 120 sequences. The script evaluates the current run's result archive and writes sequence CSV and metric JSON reports; it does not select an older run's output.

| `VARIANT` | Table 2 row | Model | Enabled components |
|---|---:|---|---|
| `baseline` | 1 | Independent SPMTrack | None of LTCP, MCC, RGTC |
| `ltcp` | 2 | STCMTrack | LTCP |
| `mcc` | 3 | STCMTrack | MCC |
| `rgtc` | 4 | STCMTrack | RGTC |
| `ltcp_mcc` | 5 | STCMTrack | LTCP, MCC |
| `ltcp_rgtc` | 6 | STCMTrack | LTCP, RGTC |
| `mcc_rgtc` | 7 | STCMTrack | MCC, RGTC |
| `full` | 8 | STCMTrack | LTCP, MCC, RGTC |

Rows 2–8 share the STCMTrack base and non-component settings. Row 1 requires independent SPMTrack weights and retains its separate structure and inference path. Both model families use the current zero-Hann public default.

STCMTrack checkpoints are rejected for `VARIANT=baseline`. Only set `ALLOW_UNMARKED_SPMTRACK_WEIGHTS=1` when the supplied unmarked checkpoint was trained with the official SPMTrack code at commit `c581fe27231f3e16c38578e47daddadfaf6ffd7d`; this is an explicit declaration of origin, not a format-conversion switch.

## Environment options

| Variable | Applies to | Default / meaning |
|---|---|---|
| `CONDA_SH` | All three wrappers | Optional explicit path to Conda's `etc/profile.d/conda.sh`; otherwise discovered from `CONDA_EXE`, Conda on `PATH`, or standard home-directory installations. |
| `CONDA_ENV` | All three wrappers | `stcmtrack`. |
| `DEVICE_IDS` | All three wrappers | `0`; comma-separated numeric GPU IDs. The wrappers check the first requested device before launch. |
| `DATASET` | All three wrappers | `antiuav410`; alternatively `antiuav300`. |
| `OUTPUT_DIR` | All three wrappers | STCMTrack training: `output/stcmtrack_train_<dataset>_stage<stage>`; SPMTrack training: `output/spmtrack_train_<dataset>`; evaluation root: `output/stcmtrack_test_<dataset>_<variant>_<scope>`. |
| `TRAIN_STAGE` | STCMTrack training | Required: `1` or `2`. |
| `BASE_WEIGHT` | STCMTrack stage 2 | Required path to the stage-1 checkpoint. |
| `VARIANT` | Evaluation | `full`; one of the eight variants above. |
| `EVAL_SCOPE` | Evaluation | `full` or `short`. |
| `BASE_WEIGHT` | Evaluation | Optional override of the dataset/model-specific path listed below. Use a valid checkpoint. |
| `LTCP_WEIGHT` | Evaluation | Optional override; required when the selected variant enables LTCP. |
| `ANTIUAV_GT_DIR` | Evaluation | Optional test-directory override used for both tracking data and metrics; otherwise read from `ANTIUAV410_PATH` or `ANTIUAV300_TEST_PATH` in `consts.yaml`. |
| `REPORT_TAG` | Evaluation | `stcmtrack_<dataset>_<variant>_<scope>`; used in report filenames. |
| `ALLOW_UNMARKED_SPMTRACK_WEIGHTS` | Baseline evaluation | `0`; `1` explicitly declares an unmarked checkpoint's official SPMTrack origin. |

Default evaluation checkpoint paths are:

| Dataset | STCMTrack base / LTCP | Independent SPMTrack |
|---|---|---|
| `antiuav410` | `weights/stcmtrack_base.bin` / `weights/stcmtrack_ltcp.bin` | `weights/spmtrack_baseline.safetensors` |
| `antiuav300` | `weights/stcmtrack_antiuav300_base.bin` / `weights/stcmtrack_antiuav300_ltcp.bin` | `weights/spmtrack_antiuav300_baseline.safetensors` |

The published files cover Anti-UAV410 only. Anti-UAV300 requires separately trained checkpoints. Evaluation creates a unique run directory under its output root.

## Direct launcher and static checks

The wrappers launch `boot.sh`, disable W&B logging and `torch.compile`, and select the appropriate training or evaluation mixins. To inspect the launcher's supported options:

```bash
./boot.sh STCMTrack dinov2 --help
python tools/check_variant_mapping.py --check
```

The static mapping check verifies configuration relationships and selected explicit paper settings. It is not a benchmark-score reproduction test. For the distinction between code checks and experimental validation, see [Implementation notes](IMPLEMENTATION_NOTES.md).

These commands and environment options are derived from `train_stcmtrack.sh`, `train_spmtrack.sh`, `test_stcmtrack.sh` and `boot.sh` at commit `8f874ad68a9c03a2704aa9c5380ddaeaf803ef68`.
