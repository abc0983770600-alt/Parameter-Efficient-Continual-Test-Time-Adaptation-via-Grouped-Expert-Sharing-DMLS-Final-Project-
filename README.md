# Parameter-Efficient Continual Test-Time Adaptation via Grouped Expert Sharing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fRl4J_73y_9BMEC4gs2Dyd-6dlUCdIcf?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
> **Authors:** Mei-Wen Chen*
> 
> **Institution:** National Taiwan University of Science and Technology, Taipei, Taiwan.

This is the official PyTorch implementation of the paper **"Parameter-Efficient Continual Test-Time Adaptation via Grouped Expert Sharing"**.

## 📖 Abstract 
Continual Test-Time Adaptation (CTTA) requires efficiently adapting to continuously changing unknown domains while retaining previously learned knowledge. However, despite recent progress in CTTA, deploying models that achieve an optimal forgetting-adaptation trade-off with high efficiency remains challenging. Furthermore, current CTTA scenarios typically assume disjoint domain shifts, whereas real-world domains change seamlessly. To address these challenges, this paper proposes the Intra-Stage Grouped Expert framework, which consists of two core components: (i) a Mixture of Experts (MoE) module that selectively captures domain-adaptive knowledge via multiple domain routers; and (ii) Intra-Stage Grouped Expert Sharing, which leverages the identical feature dimensions and high semantic homogeneity of cascaded adjacent layers within a single stage. This design not only effectively preserves the low-memory advantage of EcoTTA's frozen backbone and introduces the adaptability of multi-expert routing, but also significantly reduces the trainable parameter scale. Experimental results demonstrate that our architecture maintains a competitive global mIoU of 60.47%, achieving favorable parameter cost-effectiveness and structural robustness.

---

## 🚀 Quick Start (Google Colab)
For the easiest setup, we provide a Google Colab notebook with all dependencies, datasets, and execution scripts pre-configured.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fRl4J_73y_9BMEC4gs2Dyd-6dlUCdIcf?usp=sharing)

---

## 🖥️ Local Setup

### 1. Environment
We follow the `mmsegmentation` code base provided by the [CoTTA](https://github.com/qinenergy/cotta) authors. 
*📣 Note: Our source model (Segformer) requires a specific version of mmcv (`mmcv==1.2.0`).*

**Option A: Using provided `.yml` (Recommended)** ```shell
conda env update --name ctta_env --file environment.yml
conda activate ctta_env
