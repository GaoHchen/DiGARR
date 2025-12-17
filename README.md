<div align='center'>

<h2>Disentangled Generation and Aggregation for Robust Radiance Fields</h2>

<b>ECCV 2024</b>

Shihe Shen*, Huachen Gao*, Wangze Xu, Rui Peng, Luyang Tang, Kaiqiang Xiong, Jianbo Jiao, Ronggang Wang†

Peking University, Peng Cheng Laboratory, University of Birmingham <br>
<sup>*</sup> Equal Contribution, <sup>†</sup> Corresponding Author

</div>

## Overview

DiGARR is a novel neural rendering framework for robust radiance fields that implements disentangled generation and aggregation methods.

## 🚧 Development Status

**Note**: This code is currently under organization and not ready to run directly. Please wait for code organization to complete or contact the authors for the full version.

### Installation

1. Clone the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

The project supports LLFF dataset. Please download the corresponding data and place it in the correct location.

## Training

### Using Training Scripts

The project provides a convenient training script `scripts/train/static_train_hybrid.sh` for LLFF dataset training:

```bash
# Syntax: ./scripts/train/static_train_hybrid.sh <GPU_ID> <CONFIG_NAME>
# Example: train fern scene
./scripts/train/static_train_hybrid.sh 0 fern

# Train flower scene
./scripts/train/static_train_hybrid.sh 0 flower

# Train horns scene
./scripts/train/static_train_hybrid.sh 0 horns
```

#### Parameter Description

- `GPU_ID`: CUDA device ID (e.g., 0, 1, 2...)
- `CONFIG_NAME`: Configuration file name corresponding to LLFF dataset scene name

#### Supported LLFF Scenes

- `fern` - Fern scene
- `flower` - Flower scene
- `fortress` - Fortress scene
- `horns` - Horns scene
- `leaves` - Leaves scene
- `orchids` - Orchids scene
- `room` - Room scene
- `trex` - T-Rex scene


## TODO List

### 🔄 In Progress
- [x] Code init
- [ ] Code organization and refactoring
- [ ] NeRF Blender dataset/training
- [ ] Dependency file organization (requirements.txt)
- [ ] Pre-trained model preparation

## Citation
```
@InProceedings{digarr,
author="Shen, Shihe and Gao, Huachen and Xu, Wangze and Peng, Rui and Tang, Luyang and Xiong, Kaiqiang and Jiao, Jianbo and Wang, Ronggang",
title="Disentangled Generation and Aggregation for Robust Radiance Fields",
booktitle="Computer Vision -- ECCV 2024",
year="2025",
}
```
