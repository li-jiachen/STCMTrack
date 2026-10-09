# STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking

**Jiachen Li, Tao Yang, Kun Zhou, Jingyi Zhang**

Official PyTorch implementation of STCMTrack for infrared anti-UAV tracking. Please refer to the paper for method details and experimental results.

See [training and evaluation](docs/ABLATION.md) for the two-stage training procedure and Table 2 configurations.

## Model weights

The released weights for **Anti-UAV410** are available in [v1.0.0](https://github.com/li-jiachen/STCMTrack/releases/tag/v1.0.0). Download both files and place them in `weights/`.

| File | Description | Download |
|---|---|---|
| `stcmtrack_base.bin` | Base model weights | [Download](https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_base.bin) |
| `stcmtrack_ltcp.bin` | LTCP module weights | [Download](https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_ltcp.bin) |

The released base currently fails query numerical validation. See the [weight loading notes](weights/README.md) for the diagnosis and checkpoint requirements.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{li2026stcmtrack,
  title  = {STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking},
  author = {Li, Jiachen and Yang, Tao and Zhou, Kun and Zhang, Jingyi},
  year   = {2026}
}
```

