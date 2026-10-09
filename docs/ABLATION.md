# Table 2 ablations

The implementation follows the paper's method and Table 2 component combinations. Row 1 uses the independent SPMTrack baseline; rows 2–8 use STCMTrack with the indicated components. Settings omitted from the paper use the fixed defaults in [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md). The published scores have not been reproduced with this implementation.

Complete [setup and data preparation](SETUP.md) before running these commands.

| Row | `VARIANT` | LTCP | MCC | RGTC | Model |
|---|---|---|---|---|---|
| 1 | `baseline` | off | off | off | SPMTrack |
| 2 | `ltcp` | on | off | off | STCMTrack |
| 3 | `mcc` | off | on | off | STCMTrack |
| 4 | `rgtc` | off | off | on | STCMTrack |
| 5 | `ltcp_mcc` | on | on | off | STCMTrack |
| 6 | `ltcp_rgtc` | on | off | on | STCMTrack |
| 7 | `mcc_rgtc` | off | on | on | STCMTrack |
| 8 | `full` | on | on | on | STCMTrack |

## Training and weights

Train the SPMTrack baseline separately with `./train_spmtrack.sh`. Its checkpoint is required for row 1 and is not included in the release. See [SPMTRACK_BASELINE.md](SPMTRACK_BASELINE.md).

For STCMTrack, train the tracking network for 80 epochs, then train only LTCP for 20 epochs with all other parameters frozen. MCC and RGTC have no trainable parameters and are used only during evaluation.

```bash
TRAIN_STAGE=1 DEVICE_IDS=0 ./train_stcmtrack.sh
TRAIN_STAGE=2 BASE_WEIGHT=/path/to/stage1/checkpoint/epoch_79/model.bin \
  DEVICE_IDS=0 ./train_stcmtrack.sh
python tools/export_stcmtrack_weights.py /path/to/stage2/checkpoint/epoch_19/model.bin \
  --base-output weights/retrained/stcmtrack_base.bin \
  --ltcp-output weights/retrained/stcmtrack_ltcp.bin
```

Rows 2–8 share the STCMTrack base file. Rows with LTCP additionally load the same gate file; the other rows skip it. Newly exported files record their source snapshot. The loader recognizes the published incremental `.bin` format, but currently rejects the released base because its learned query values overflow FP32 LayerNorm. A valid checkpoint is required; see [weights/README.md](../weights/README.md).

SPMTrack and STCMTrack share the paper's data splits, input sizes, 80-epoch base training budget, AdamW (learning rate 1e-4, weight decay 0.1), cosine schedule and equally weighted BCE/GIoU losses. STCMTrack adds the frozen-base LTCP training stage. Binary center targets, averaging search-frame losses, no warm-up and the batch/sample counts are fixed public defaults not specified in the paper. Each benchmark requires its own trained weights.

## Evaluation

Evaluate the independently trained SPMTrack checkpoint and the newly exported STCMTrack pair:

```bash
VARIANT=baseline BASE_WEIGHT=/path/to/spmtrack/checkpoint/epoch_79/model.bin \
  DEVICE_IDS=0 ./test_stcmtrack.sh
for variant in ltcp mcc rgtc ltcp_mcc ltcp_rgtc mcc_rgtc full; do
  VARIANT="$variant" DEVICE_IDS=0 \
    BASE_WEIGHT="$PWD/weights/retrained/stcmtrack_base.bin" \
    LTCP_WEIGHT="$PWD/weights/retrained/stcmtrack_ltcp.bin" \
    ./test_stcmtrack.sh
done
```

The paper's Table 2 uses the complete Anti-UAV410 test split. For the separate Anti-UAV benchmark, set `DATASET=antiuav300` for training and evaluation and supply that benchmark's own trained checkpoints; see [weight paths](../weights/README.md). Export refuses existing outputs, so choose a fresh directory for each trained pair. Validate the variant mapping and shared settings with:

```bash
python tools/check_variant_mapping.py --check
python tools/check_variant_mapping.py --dataset antiuav300 --check
```

## Baseline interpretation

The paper calls row 1 SPMTrack and states that the remaining experimental settings are shared. Row 1 retains the independent SPMTrack model: three templates, propagated query state, query-based reweighting before the prediction heads and online template updates. Its training inputs contain three templates and two interval-sampled search frames. STCMTrack uses the first-frame template, three causal search frames for training, a query encoded from the current joint features and LTCP features passed directly to the heads. These input and temporal computations belong to the respective models; their differences alone do not contradict the paper's shared-settings statement.

The public configurations share the ViT backbone and MLP head architectures, data splits, crop settings, joint augmentation, optimizer, schedule, loss conventions, training budget and metric implementation. Both disable Hann-window post-processing and use the same evaluation crop minimum size. These are common public settings, not evidence of the undocumented settings used in the original experiments. Train valid checkpoints and evaluate all eight variants to obtain results for this implementation. Historical configuration and result files are still needed to verify the published Table 2 scores.
