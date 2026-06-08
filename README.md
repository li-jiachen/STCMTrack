# STCMTrack

This repository contains the official implementation of **STCMTrack**, a single-object tracking framework for infrared anti-UAV scenarios.

The project focuses on robust UAV tracking under challenging thermal infrared conditions, including tiny target appearance, fast motion, occlusion, background clutter, scale variation, and target re-localization.

## Updates

This repository is currently under active organization. The following materials will be gradually released and documented:

- Source code
- Training commands
- Evaluation commands
- Debugging and testing commands
- Environment configuration
- Experimental results and visualization examples

The pre-trained checkpoint for **STCMTrack-Base** has been released.

## Project Structure

The repository is organized as follows:

```text
STCMTrack/
├── assets/        # Project figures and visualization assets
├── checkpoints/   # Training checkpoints and intermediate saved models
├── configs/       # Model and experiment configuration files
├── datasets/      # Dataset preparation instructions
├── docs/          # Supplementary documentation
├── pretrained/    # Pre-trained weights downloaded from GitHub Releases
├── results/       # Tracking results, evaluation outputs, and visualizations
├── scripts/       # Helper scripts for running experiments
├── stcmtrack/     # Core implementation of STCMTrack
├── tools/         # Entry scripts for training, testing, and evaluation
└── README.md
```

Please note that large files, such as datasets and model weights, are not directly stored in this repository. Download links and usage instructions are provided instead.

## Dataset

The datasets used in this project are not included in the repository due to their large size and possible redistribution restrictions.

Please refer to:

```text
datasets/README.md
```

for dataset download links, preparation instructions, and the recommended directory structure.

## Pretrained Models

| Model | Checkpoint | Description |
|---|---|---|
| STCMTrack-Base | [Download](https://github.com/li-jiachen/STCMTrack/releases/download/v0.1.0/stcmtrack_base.bin) | Pretrained checkpoint for STCMTrack-Base |

The pre-trained checkpoint is released through GitHub Releases.

Please download `stcmtrack_base.bin` from the link above and place it under the following path:

```text
pretrained/stcmtrack_base.bin
```

For example:

```bash
mkdir -p pretrained
# then place stcmtrack_base.bin into the pretrained/ directory
```

After preparation, the expected file structure should be:

```text
STCMTrack/
└── pretrained/
    └── stcmtrack_base.bin
```

## Installation

The running environment and dependency installation instructions will be updated soon.

A complete environment configuration file will be provided for reproducibility.

## Training

Training commands will be added after the source code and configuration files are fully organized.

## Evaluation

Evaluation commands and benchmark testing instructions will be provided soon.

## Debugging and Testing

Basic debugging commands and testing examples will be added to help users verify the environment and reproduce the experimental results.

## Contact

If you have any questions about this repository, please feel free to contact:

```text
jiachenli@stu.cwnu.edu.cn
```

## Acknowledgement

We sincerely thank the authors of the public datasets and open-source tracking frameworks that support this research.

More details will be updated soon.
