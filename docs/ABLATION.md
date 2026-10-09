# Table 2 ablations

The implementation follows the paper's method and Table 2 component combinations. Row 1 uses the independent SPMTrack baseline; rows 2–8 use STCMTrack with the indicated components. Settings omitted from the paper use the fixed defaults in [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md). The published scores have not been reproduced with this implementation.

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
  --base-output weights/stcmtrack_base.bin \
  --ltcp-output weights/stcmtrack_ltcp.bin
```

Rows 2–8 share the STCMTrack base file. Rows with LTCP additionally load the same gate file; the other rows skip it. Newly exported files record their source snapshot. The loader recognizes the published incremental `.bin` format, but currently rejects the released base because its learned query values overflow FP32 LayerNorm. A valid checkpoint is required; see [weights/README.md](../weights/README.md).

SPMTrack and STCMTrack share the paper's data splits, input sizes, 80-epoch base training budget, AdamW (learning rate 1e-4, weight decay 0.1), cosine schedule and equally weighted BCE/GIoU losses. STCMTrack adds the frozen-base LTCP training stage. Binary center targets, averaging search-frame losses, no warm-up and the batch/sample counts are fixed public defaults not specified in the paper. Each benchmark requires its own trained weights.

## Evaluation

Place valid SPMTrack and STCMTrack checkpoints at the default paths described in [weights/README.md](../weights/README.md), then run:

```bash
for variant in baseline ltcp mcc rgtc ltcp_mcc ltcp_rgtc mcc_rgtc full; do
  VARIANT="$variant" DEVICE_IDS=0 ./test_stcmtrack.sh
done
```

The paper's Table 2 uses the complete Anti-UAV410 test split. For the separate Anti-UAV benchmark, set `DATASET=antiuav300` for training and evaluation. Validate the variant mapping and shared settings with:

```bash
python tools/check_variant_mapping.py --check
python tools/check_variant_mapping.py --dataset antiuav300 --check
```

## Baseline interpretation

The paper calls row 1 SPMTrack and states that the remaining experimental settings are shared. The independent SPMTrack model retains three templates, propagated query state and query-based reweighting before the prediction heads, together with two-search-frame interval sampling, per-image augmentation, online templates and Hann-window post-processing. STCMTrack uses the first-frame template, a query encoded from the current joint features and LTCP features passed directly to the heads. The ViT backbone and MLP head architectures are shared, but these surrounding computations differ.

Keeping the independent SPMTrack baseline and the paper-defined STCMTrack method therefore leaves differences beyond the three component switches between row 1 and rows 2–8. The scripts preserve that baseline identity and align the common training settings; they do not make row 1 a strictly controlled three-switch comparison. Train valid checkpoints and evaluate all eight variants to obtain new results for this implementation. Those measurements must be reported separately from the paper's published scores.
