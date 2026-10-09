# Model weights

The [v1.0.0 release](https://github.com/li-jiachen/STCMTrack/releases/tag/v1.0.0) provides two Anti-UAV410 weight files; they are not stored in Git. `test_stcmtrack.sh` uses these filenames by default when the files are placed in this directory:

| File | Content |
|---|---|
| `stcmtrack_base.bin` | STCMTrack tracking network, Anti-UAV410 |
| `stcmtrack_ltcp.bin` | LTCP gate, Anti-UAV410 |

The `.bin` extension is a filename convention: these files use the Safetensors format, not PyTorch pickle.

**Current release compatibility:** the two v1.0.0 files use an older checkpoint layout. Their scaling fields are named `expert_alpha` / `use_rsexpert`, while the current loader expects `_expert_alpha` / `_use_rsexpert`. They also lack the required source metadata, and the base file does not include the frozen backbone. Changing their filenames does not make them compatible. This does not establish that the original weights are corrupt or unusable with their original model and matching DINOv2 backbone. Evaluation with the current code requires a complete checkpoint exported as described below.

Other locations can be passed with `BASE_WEIGHT=...` and `LTCP_WEIGHT=...`.

## STCMTrack

For checkpoints exported with the current code, the base file contains the complete tracking network (including the frozen DINOv2 backbone), and the LTCP file contains the gate parameters of LTCP. The base file is always loaded first; variants with LTCP load the LTCP file second.

All eight component ablations use the same base file. `VARIANT=baseline` (alias `stcm_base`) disables LTCP, MCC and RGTC and loads only this STCMTrack base file. The LTCP variants additionally load the shared gate file. See [docs/ABLATION.md](../docs/ABLATION.md).

Both files are exported from the same stage-2 training snapshot:

```bash
python tools/export_stcmtrack_weights.py /path/to/stage2/checkpoint/epoch_19/model.bin \
  --base-output weights/stcmtrack_base.bin \
  --ltcp-output weights/stcmtrack_ltcp.bin
```

The export records the SHA-256 of the source snapshot in both files and writes `stcmtrack_base.manifest.json` with the SHA-256 of the two outputs. Omit `--ltcp-output` to export a stage-1 snapshot. `test_stcmtrack.sh` checks before every evaluation that the base and LTCP files come from the same snapshot; the check can also be run directly:

```bash
python tools/check_stcmtrack_weights.py \
  --base weights/stcmtrack_base.bin \
  --ltcp weights/stcmtrack_ltcp.bin
```

Each benchmark requires its own pair of files. For separately trained Anti-UAV weights, `DATASET=antiuav300` defaults to `stcmtrack_antiuav300_base.bin` and `stcmtrack_antiuav300_ltcp.bin`; these are not included in the release. Do not combine the base file of one dataset with the LTCP file of the other.

## Independent SPMTrack comparison

`VARIANT=spmtrack` uses its own weight file for the separate Table 1 comparison. It is the `model.bin` written by `./train_spmtrack.sh`, copied to `weights/spmtrack_baseline.safetensors` (`weights/spmtrack_antiuav300_baseline.safetensors` for Anti-UAV); it contains the trainable parameters and the marker `_spmtrack_port_version`. These SPMTrack weights are not included in the release.

```bash
python tools/check_spmtrack_weights.py --weights weights/spmtrack_baseline.safetensors
```

STCMTrack weight files are refused by the SPMTrack comparison. A file without the marker, such as a checkpoint trained with the official SPMTrack code, is accepted only with `ALLOW_UNMARKED_SPMTRACK_WEIGHTS=1`. See [docs/SPMTRACK_BASELINE.md](../docs/SPMTRACK_BASELINE.md).
