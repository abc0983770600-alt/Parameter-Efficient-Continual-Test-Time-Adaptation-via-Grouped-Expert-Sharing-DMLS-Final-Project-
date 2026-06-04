# Parameter-Efficient Continual Test-Time Adaptation via Grouped Expert Sharing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fRl4J_73y_9BMEC4gs2Dyd-6dlUCdIcf?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Authors:** Mei-Wen Chen*
> **Institution:** National Taiwan University of Science and Technology, Taipei, Taiwan.

Official PyTorch implementation of:

**"Parameter-Efficient Continual Test-Time Adaptation via Grouped Expert Sharing"**

---

# 📖 Abstract

Continual Test-Time Adaptation (CTTA) requires efficiently adapting to continuously changing unknown domains while retaining previously learned knowledge. However, despite recent progress in CTTA, deploying models that achieve an optimal forgetting-adaptation trade-off with high efficiency remains challenging. Furthermore, current CTTA scenarios typically assume disjoint domain shifts, whereas real-world domains change seamlessly.

To address these challenges, we propose the **Intra-Stage Grouped Expert Framework**, which consists of two core components:

1. **Mixture of Experts (MoE)** modules that selectively capture domain-adaptive knowledge via multiple domain routers.
2. **Intra-Stage Grouped Expert Sharing**, which leverages the identical feature dimensions and high semantic homogeneity of cascaded adjacent layers within the same stage.

Our design preserves the low-memory advantage of EcoTTA's frozen backbone while introducing adaptive multi-expert routing. Experimental results demonstrate that the proposed architecture achieves competitive segmentation performance with substantially fewer trainable parameters.

**Key Result:**

* Global mIoU: **60.47%**
* Trainable Parameters: **0.69M**
* Parameter Reduction: **74.3%**

---

# 🚀 Quick Start (Google Colab)

For the easiest setup, we provide a Google Colab notebook with all dependencies, datasets, and execution scripts pre-configured.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fRl4J_73y_9BMEC4gs2Dyd-6dlUCdIcf?usp=sharing)

---

# 🖥️ Local Setup

## 1. Environment

We follow the `mmsegmentation` codebase provided by the CoTTA authors.

> 📣 **Note:** Our SegFormer source model requires `mmcv==1.2.0`.

### Option A: Using the provided environment file (Recommended)

```bash
conda env update --name ctta_env --file environment.yml
conda activate ctta_env
```

### Option B: Manual Installation

Tested on:

* Python 3.8
* PyTorch 1.7.0
* CUDA 11.0
* MMCV 1.2.0

```bash
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 \
-f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.2.0 \
-f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

For issues related to lower MMCV versions, please refer to the corresponding discussions in the CoTTA repository.

---

## 2. Dataset Preparation

Download the ACDC dataset and construct the continual adaptation sequence:

```text
Fog → Night → Rain → Snow
```

using the training split.

Update the dataset root path in:

```text
configs/_base_/datasets/acdc_1024x1024_repeat_origin.py
```

```python
dataset_type = 'ACDCDataset'
data_root = '/path/to/your/ACDC_dataset'
```

### Supported Datasets

Configuration files are also provided for:

* ACDC
* BDD100K
* KITTI
* GTA5

located in:

```text
mmseg/datasets/
```

> Ensure that the segmentation labels follow the Cityscapes format.

---

## 3. Pre-trained Models

Download the following pre-trained SegFormer checkpoints and place them under:

```text
./pretrained/
```

### SegFormer Weights

```text
segformer.b5.1024x1024.city.160k.pth
mit_b5.pth
```

Directory structure:

```text
pretrained/
├── segformer.b5.1024x1024.city.160k.pth
└── mit_b5.pth
```

---

# 🏃‍♂️ Usage

Run continual test-time adaptation with:

```bash
python test.py \
    --config configs/your_config_file.py \
    --checkpoint pretrained/segformer.b5.1024x1024.city.160k.pth
```

Replace:

```text
configs/your_config_file.py
```

with your experiment configuration.

---

# 📊 Main Results

Our method significantly reduces trainable parameters while maintaining competitive performance.

| Method                     | Fog      | Night    | Rain     | Snow     | Mean mIoU | Trainable Params |
| -------------------------- | -------- | -------- | -------- | -------- | --------- | ---------------- |
| EcoTTA (CVPR'23)           | 68.5     | 35.8     | 62.1     | 57.4     | 55.8      | 3.46M            |
| Unoptimized MoE            | 71.8     | 48.0     | 66.3     | 62.0     | 61.9      | 2.70M            |
| **Grouped Sharing (Ours)** | **71.6** | **47.8** | **66.1** | **61.7** | **60.47** | **0.69M**        |

### Parameter Efficiency

| Method                 | Trainable Parameters | Reduction |
| ---------------------- | -------------------- | --------- |
| EcoTTA                 | 3.46M                | -         |
| Grouped Sharing (Ours) | 0.69M                | 74.3% ↓   |

---

# 📌 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{chen2024groupedsharing,
  title={Parameter-Efficient Continual Test-Time Adaptation via Grouped Expert Sharing},
  author={Chen, Mei-Wen},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
