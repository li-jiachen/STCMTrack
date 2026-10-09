# Component ablations

The eight combinations of LTCP, MCC and RGTC use the same STCMTrack tracking network. The evaluation variants correspond to the component columns of Table 2:

| Row | `VARIANT` | LTCP | MCC | RGTC | Model |
|---|---|---|---|---|---|
| 1 | `baseline` (alias `stcm_base`) | off | off | off | STCMTrack |
| 2 | `ltcp` | on | off | off | STCMTrack |
| 3 | `mcc` | off | on | off | STCMTrack |
| 4 | `rgtc` | off | off | on | STCMTrack |
| 5 | `ltcp_mcc` | on | on | off | STCMTrack |
| 6 | `ltcp_rgtc` | on | off | on | STCMTrack |
| 7 | `mcc_rgtc` | off | on | on | STCMTrack |
| 8 | `full` | on | on | on | STCMTrack |

`VARIANT=spmtrack` selects the independent SPMTrack comparison for Table 1. It uses its own model, training configuration and weights; it is not a ninth component ablation.

## Shared training and weights

For each dataset, train the tracking network once for 80 epochs, then train only the LTCP gate for 20 epochs from that stage-1 checkpoint. MCC and RGTC have no trainable parameters and are enabled only during evaluation.

```bash
TRAIN_STAGE=1 DEVICE_IDS=0 ./train_stcmtrack.sh
TRAIN_STAGE=2 BASE_WEIGHT=/path/to/stage1/checkpoint/epoch_79/model.bin \
  DEVICE_IDS=0 ./train_stcmtrack.sh
python tools/export_stcmtrack_weights.py /path/to/stage2/checkpoint/epoch_19/model.bin \
  --base-output weights/stcmtrack_base.bin \
  --ltcp-output weights/stcmtrack_ltcp.bin
```

Use the same exported base file for all eight variants. Stage 2 freezes the base network, so it adds the LTCP gate without further training the backbone, adapters, queries or heads. Every variant with LTCP uses the same exported gate file; the other variants skip that file. Both files must come from the same stage-2 snapshot, as checked by `tools/check_stcmtrack_weights.py`. The older files in the v1.0.0 release do not meet the current loader's requirements; see [weights/README.md](../weights/README.md).

The variants share one first-frame template, three chronological search frames per training sample, binary center targets, the mean loss across those frames, AdamW and the cosine schedule. Evaluation shares the same STCMTrack post-processing, data split and metric implementation. The switches change only LTCP fusion, MCC search recentering and RGTC low-confidence correction. Search crops and subsequent states can consequently differ, as intended by these components.

## Evaluation

```bash
for variant in baseline ltcp mcc rgtc ltcp_mcc ltcp_rgtc mcc_rgtc full; do
  VARIANT="$variant" DEVICE_IDS=0 ./test_stcmtrack.sh
done
```

For Anti-UAV, set `DATASET=antiuav300` for both training stages and every evaluation. That dataset requires its own weights and must not reuse the Anti-UAV410 pair.

Validate the model, component mapping and shared settings for each dataset:

```bash
python tools/check_variant_mapping.py --check
python tools/check_variant_mapping.py --dataset antiuav300 --check
```

## Conflict with the paper and reported scores

The paper identifies Table 2 row 1 as the SPMTrack baseline, while also stating that the backbone, prediction head and remaining experimental settings are shared across all eight configurations. The independent SPMTrack implementation differs from STCMTrack in more than these three switches: it uses three templates, a propagated query, query-based head-input reweighting, different training samples and targets, summed frame losses, warm-up and parameter-specific weight decay, and Hann-window post-processing.

Consequently, keeping that independent SPMTrack path as row 1 would not isolate LTCP, MCC and RGTC. The mapping above enforces a shared STCMTrack foundation for the eight component combinations. It does not establish that the new row 1 reproduces the paper's SPMTrack scores, or that the eight configurations reproduce the published numbers. Resolving that requires the original experiment configurations and a new evaluation; no paper results have been changed or relabeled here.
