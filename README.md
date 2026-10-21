# STCMTrack

**Jiachen Li, Tao Yang, Kun Zhou, Jingyi Zhang**

Official PyTorch implementation of STCMTrack for infrared anti-UAV tracking. See the paper for the method and experimental results, and the [setup guide](docs/SETUP.md) for installation and usage.

## Model Weights

Download the [base weights](https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_base.bin) and [LTCP weights](https://github.com/li-jiachen/STCMTrack/releases/download/v1.0.0/stcmtrack_ltcp.bin), and place them in `weights/`.

These weights are for Anti-UAV410. Evaluation on Anti-UAV requires weights trained on that dataset.

## Citation

```bibtex
@misc{li2026stcmtrack,
  title  = {STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking},
  author = {Li, Jiachen and Yang, Tao and Zhou, Kun and Zhang, Jingyi},
  year   = {2026}
}
```

## Licenses

See the [component licenses and notices](docs/THIRD_PARTY_NOTICES.md) for source references and license terms.

## Contact

For questions, please contact Jiachen Li at [jiachenli@stu.cwnu.edu.cn](mailto:jiachenli@stu.cwnu.edu.cn).
