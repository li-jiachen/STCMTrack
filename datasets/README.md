# Datasets

This directory is used to store external datasets required by **STCMTrack**.

The original dataset files are **not included** in this repository due to their large size and possible redistribution restrictions. Please download the datasets from their official sources and organize them according to the directory structure described below.

## Anti-UAV410

**Anti-UAV410** is a thermal infrared benchmark for single-object UAV tracking in realistic anti-UAV scenarios. It is designed for evaluating drone tracking under long-range imaging, tiny target appearance, fast motion, occlusion, out-of-view cases, thermal crossover, scale variation, and dynamic background clutter.

### Official Resources

- Paper: *Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild*
- Official repository: https://github.com/HwangBo94/Anti-UAV410
- Dataset download: please refer to the official repository for the latest Google Drive and Baidu Disk links.
- Baidu Disk access code: `a410`

## Dataset Preparation

Please download the Anti-UAV410 dataset from the official source and place it under the `datasets/` directory as follows:

```text
STCMTrack/
└── datasets/
    └── AntiUAV410/
        ├── train/
        ├── val/
        └── test/
```

Please keep the original sequence names and annotation files unchanged unless the dataloader explicitly requires a different format.

If the official attribute annotations are used, keep their structure consistent with the official benchmark toolkit:

```text
annos/
├── train/
│   └── att/
├── val/
│   └── att/
└── test/
    └── att/
```

Depending on the implementation, the `annos/` directory can either be kept in the project root or placed under `datasets/AntiUAV410/`. Make sure the dataset path is correctly configured before training or evaluation.

## Dataset Split

The official Anti-UAV410 benchmark is split into three subsets:

```text
train: 200 sequences
val:    90 sequences
test:  120 sequences
```

For fair comparison with existing methods, please use the official split and avoid mixing training, validation, and test sequences.

## Annotation Format

Anti-UAV410 provides frame-level bounding box annotations for UAV targets.

The target bounding box is represented as:

```text
[x, y, width, height]
```

where:

- `x` and `y` denote the top-left corner of the bounding box;
- `width` and `height` denote the size of the target bounding box;
- an empty annotation indicates that the target is not visible in the current frame.

## Attributes

Anti-UAV410 provides sequence-level challenge attributes for comprehensive evaluation. The main attributes include:

| Attribute | Full Name | Description |
|---|---|---|
| `OV` | Out-of-View | The target leaves the camera view. |
| `OC` | Occlusion | The target is partially or heavily occluded. |
| `TC` | Thermal Crossover | The target has similar temperature to surrounding background regions. |
| `FM` | Fast Motion | The target center moves rapidly between adjacent frames. |
| `SV` | Scale Variation | The target scale changes significantly during tracking. |
| `DBC` | Dynamic Background Clutter | The background around the target changes dynamically. |

The dataset also contains different target scale categories, including tiny, small, medium, and normal-size UAV targets.

## Recommended Configuration

For this project, the dataset root can be configured as:

```yaml
DATA:
  DATASET: AntiUAV410
  ROOT: ./datasets/AntiUAV410
```

If your local dataset is stored elsewhere, please update the corresponding configuration file or create a symbolic link to avoid hard-coding absolute paths.

## Git Tracking Policy

Large dataset files should not be committed to this repository.

A recommended `.gitignore` configuration is:

```gitignore
# Ignore dataset files
datasets/*

# Keep dataset documentation
!datasets/README.md
!datasets/.gitkeep
```

This allows the repository to keep the dataset instructions while excluding the actual dataset files.

## Citation

If you use Anti-UAV410 in your research, please cite the original paper:

```bibtex
@article{huang2024antiuav410,
  title={Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild},
  author={Huang, Bo and Li, Jianan and Chen, Junjie and Wang, Gang and Zhao, Jian and Xu, Tingfa},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={46},
  number={5},
  pages={2852--2865},
  year={2024},
  doi={10.1109/TPAMI.2023.3335338}
}
```

## Notes

- This repository does not redistribute Anti-UAV410.
- Please follow the license and usage terms provided by the original dataset authors.
- For reproducible evaluation, use the official train/validation/test split.
- Do not rename sequence folders unless the dataloader is modified accordingly.
- Before running training or testing scripts, make sure the dataset path in the configuration file matches your local environment.
