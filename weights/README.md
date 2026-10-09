# Model weights

The [v1.0.0 release](https://github.com/li-jiachen/STCMTrack/releases/tag/v1.0.0) contains the two original **Anti-UAV410** files. Place them in this directory:

| File | Content |
|---|---|
| `stcmtrack_base.bin` | Tracking adapters, queries, token embeddings and prediction heads |
| `stcmtrack_ltcp.bin` | LTCP gate |

Both files use Safetensors despite the `.bin` extension. The base is an incremental checkpoint: the frozen backbone comes from the official pretrained DINOv2 ViT-B/14 weights, which the model builder downloads on first use. The loader recognizes the original checkpoint layout and checks file identity, required parameters, shapes, TMoE scaling and query numerical range before loading. A randomly initialized backbone is refused.

**Current release limitation:** the original base contains `track_query` and `query_embed` with maximum absolute values of approximately 4.03e28 and 9.85e28. Their combined variance exceeds the FP32 range. With the official pretrained backbone, a CPU FP32 test produced NaNs in the first LayerNorm and attention layer. Evaluation now rejects this checkpoint with a specific diagnostic. The original training code and a valid training checkpoint are needed to determine why these values were saved; the stored files have not been changed or repaired.

```bash
python tools/check_stcmtrack_weights.py \
  --base weights/stcmtrack_base.bin --ltcp weights/stcmtrack_ltcp.bin
DEVICE_IDS=0 ./test_stcmtrack.sh
```

The check currently reports query overflow for the released base. The evaluation command requires a checkpoint that passes validation.

The default `VARIANT=full` loads a validated base first and the LTCP gate second, with MCC and RGTC enabled. Table 2 rows 2-8 share the STCMTrack base; rows using LTCP also load the gate. `BASE_WEIGHT` and `LTCP_WEIGHT` accept other file locations. See [the eight configurations](../docs/ABLATION.md).

The original files have no training-snapshot metadata. Their hashes verify their identity as the published pair; they do not prove the original training configuration or reproduce the paper's scores.

## New training checkpoints

Checkpoints produced by the current code contain the complete model, including the frozen backbone. Export a base and gate from the same stage-2 checkpoint:

```bash
python tools/export_stcmtrack_weights.py /path/to/stage2/checkpoint/epoch_19/model.bin \
  --base-output weights/stcmtrack_base.bin \
  --ltcp-output weights/stcmtrack_ltcp.bin
```

Export checks the query numerical range before writing any output. These exports carry source-snapshot metadata and an output hash manifest. Evaluation checks that the base and gate originate from the same snapshot. Keep each pair together; mixing original release files with new exports is refused. Omit `--ltcp-output` for a stage-1 checkpoint.

The release provides Anti-UAV410 weights only. `DATASET=antiuav300` requires separately trained files, named `stcmtrack_antiuav300_base.bin` and `stcmtrack_antiuav300_ltcp.bin` by default.

## SPMTrack baseline

Table 2 row 1, `VARIANT=baseline`, uses the independent SPMTrack model and its own checkpoint. Its weights are not included in this release. Copy the checkpoint produced by `train_spmtrack.sh` to `spmtrack_baseline.safetensors` (or `spmtrack_antiuav300_baseline.safetensors` for Anti-UAV).

```bash
VARIANT=baseline DEVICE_IDS=0 ./test_stcmtrack.sh
```

STCMTrack weights cannot be used as the SPMTrack baseline. Official unmarked SPMTrack checkpoints require `ALLOW_UNMARKED_SPMTRACK_WEIGHTS=1`; see [SPMTrack baseline](../docs/SPMTRACK_BASELINE.md).
