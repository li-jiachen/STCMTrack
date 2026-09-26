# Model Weights

Checkpoints are distributed through the GitHub Release `v1.0.0` and are not committed to Git.

| File | Content | SHA256 |
|---|---|---|
| [`stcmtrack_base.bin`](https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_base.bin) | Tracking network trained on Anti-UAV410 (the baseline, row 1 of the component ablation) | `71eb8f20cbdec8bf2dff5177806923eca0fa08b694c2b40cf5399f2d9ad2d2d0` |
| [`stcmtrack_ltcp.bin`](https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_ltcp.bin) | LTCP parameters | `ed4e063bcac94e6fd98de14abb1416965f4a26ad92f482602234807228b291b7` |

```bash
mkdir -p weights
wget -O weights/stcmtrack_base.bin https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_base.bin
wget -O weights/stcmtrack_ltcp.bin https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_ltcp.bin
sha256sum weights/stcmtrack_base.bin weights/stcmtrack_ltcp.bin
```

`test_stcmtrack.sh` loads the base checkpoint first and then, for variants with LTCP, the LTCP checkpoint. `missing keys` messages are expected when loading them (the DINOv2 weights are loaded separately and the LTCP file holds only the LTCP parameters); `unexpected keys` should be 0. Other locations can be set with `BASE_WEIGHT=...` and `LTCP_WEIGHT=...`.

For Anti-UAV (`DATASET=antiuav300`), the scripts use `weights/stcmtrack_antiuav300_base.bin` and `weights/stcmtrack_antiuav300_ltcp.bin`.
