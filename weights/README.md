# Model weights

Checkpoints are released on the [Releases](https://github.com/li-jiachen/STCMTrack/releases) page and are not stored in Git. Place them in this directory under the following names, which are the defaults of `test_stcmtrack.sh`:

| File | Content |
|---|---|
| `stcmtrack_paper_base.safetensors` | STCMTrack tracking network trained on Anti-UAV410 |
| `stcmtrack_paper_ltcp.safetensors` | LTCP gate trained on Anti-UAV410 |
| `stcmtrack_antiuav300_paper_base.safetensors` | STCMTrack tracking network trained on Anti-UAV |
| `stcmtrack_antiuav300_paper_ltcp.safetensors` | LTCP gate trained on Anti-UAV |
| `spmtrack_baseline.safetensors` | SPMTrack baseline trained on Anti-UAV410 |
| `spmtrack_antiuav300_baseline.safetensors` | SPMTrack baseline trained on Anti-UAV |

Other locations can be passed with `BASE_WEIGHT=...` and `LTCP_WEIGHT=...`.

## STCMTrack

The base file contains the complete tracking network (including the frozen DINOv2 backbone), and the LTCP file contains the gate parameters of LTCP. The base file is always loaded first; variants with LTCP load the LTCP file second.

Both files are exported from the same stage-2 training snapshot:

```bash
python tools/export_paper_weights.py /path/to/stage2/checkpoint/epoch_19/model.bin \
  --base-output weights/stcmtrack_paper_base.safetensors \
  --ltcp-output weights/stcmtrack_paper_ltcp.safetensors
```

The export records the SHA-256 of the source snapshot in both files and writes `stcmtrack_paper_base.manifest.json` with the SHA-256 of the two outputs. Omit `--ltcp-output` to export a stage-1 snapshot. `test_stcmtrack.sh` checks before every evaluation that the base and LTCP files come from the same snapshot; the check can also be run directly:

```bash
python tools/check_paper_weights.py \
  --base weights/stcmtrack_paper_base.safetensors \
  --ltcp weights/stcmtrack_paper_ltcp.safetensors
```

Each benchmark has its own pair of files. Do not combine the base file of one dataset with the LTCP file of the other.

## SPMTrack baseline

`VARIANT=baseline` uses its own weight file. It is the `model.bin` written by `./train_spmtrack.sh`, copied to `weights/spmtrack_baseline.safetensors`; it contains the trainable parameters and the marker `_spmtrack_port_version`.

```bash
python tools/check_spmtrack_weights.py --weights weights/spmtrack_baseline.safetensors
```

STCMTrack weight files are refused by the baseline. A file without the marker, such as a checkpoint trained with the official SPMTrack code, is accepted only with `ALLOW_UNMARKED_SPMTRACK_WEIGHTS=1`. See [docs/SPMTRACK_BASELINE.md](../docs/SPMTRACK_BASELINE.md).
