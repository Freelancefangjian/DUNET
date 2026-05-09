# DUNET
DUNET: Deep Unfolding Network Enhanced by Transformer Priors for Unregistered Hyperspectral and Multispectral Image Fusion

https://img.shields.io/badge/License-MIT-blue.svg](LICENSE)
https://img.shields.io/badge/PyTorch-2.0+-red.svg](https://pytorch.org/)

This repository is the official PyTorch implementation of the IEEE TGRS 2024 paper: "Deep Unfolding Network Enhanced by Transformer Priors for Unregistered Hyperspectral and Multispectral Image Fusion".

📖 Overview

Fusing a Low-Resolution Hyperspectral Image (LR-HSI) with a High-Resolution Multispectral Image (HR-MSI) to generate a High-Resolution HSI (HR-HSI) is a crucial yet ill-posed problem. Most existing methods critically depend on the precise registration of the input images, which is often unavailable in real-world satellite remote sensing scenarios due to differing acquisition conditions.

This paper proposes a novel Deep Unfolding Network Enhanced by Transformer Priors (DUNET) to tackle the simultaneous registration and fusion of unregistered HSI and MSI data. Unlike conventional "black-box" deep fusion methods, DUNET incorporates the deep unfolding methodology, which leverages mutual information and deep priors to inform a more accurate, degradation model-aware fusion process. The network integrates Hybrid Attention Transformers (HATs) and Spatial-Frequency modules to fully exploit the spatial-spectral information, resulting in superior fusion performance without requiring pre-registration.

✨ Key Features

•   Simultaneous Registration & Fusion: Solves the core challenge of fusing unregistered HSI and MSI images in an end-to-end manner, eliminating the need for a separate, error-prone registration step.

•   Interpretable Deep Unfolding: Translates a model-based optimization algorithm (solved via ADMM) into a deep network, enhancing interpretability and incorporating physical degradation models.

•   Advanced Transformer Priors: Employs Hybrid Attention Transformers (HATs), combining Window-based and Spectral Self-Attention, to effectively capture both local details and global spectral dependencies.

•   Spatial-Spectral Feature Mining: Introduces a Spatial-Frequency Block (SFB) to explicitly model information in both the spatial and frequency domains, leading to richer feature representations.

•   State-of-the-Art Performance: Extensive experiments on three benchmark datasets (ICVL, Chikusei, Houston) under simulated misalignment (2° and 5° rotation) demonstrate that DUNET outperforms existing mainstream methods by a significant margin.

🎯 Method

DUNET unfolds the iterative steps of solving a model-based optimization problem (Eq. 8-9 in the paper). Each iteration stage in the network (with N=5 stages) explicitly solves sub-problems for the target HR-HSI (Z), aligned LR-HSI (X_hat), and an auxiliary variable (V) using learned proximal operators.

Core Components (Refer to Figure 1 in the paper):
1.  Mutual Information-based Initialization: Provides an initial alignment estimate for the unregistered LR-HSI.
2.  Deep Unfolding Framework: Iterative network stages that update the variables Z, X_hat, V, and Lagrange multipliers based on the unfolded ADMM steps.
3.  Hybrid Attention Transformer (HAT) Proximal Operator: Serves as a deep prior within the unfolding process. It uses a cascade of HAT blocks (with L=3 blocks) to refine features, combining Windowed MSA for local context and Spectral MSA for cross-channel attention.
4.  Spatial-Frequency Block (SFB): Enhances the output of HAT by processing features through parallel spatial (convolutional) and frequency (FFT-based) branches, capturing complementary information.

📊 Performance Summary

Quantitative results (average) on the ICVL, Chikusei, and Houston datasets with a scaling factor of 4 and simulated misalignment. DUNET achieves consistent improvements.

ICVL Dataset (2° rotation) - Selected Metrics:

| Method | MPSNR | SAM | ERGAS  | MSSIM |
|--------|-------|-----|-------|-------|
| CNMF | 29.97 | 2.83 | 5.22 | 0.907 |
| HySure | 29.22 | 7.17 | 5.71 | 0.838 |
| u²-MDN | 43.15 | 6.44 | 1.44 | 0.943 |
| MSST | 45.38 | 1.87 | 0.67 | 0.994 | 
| MoE-PNP | 34.93 | 5.20 | 2.65 | 0.976 |
| DUNET (Ours) | 48.81 | 1.56 | 0.56 | 0.995 |

Note: DUNET achieves improvements of 3.4 dB, 5.1 dB, and 8.2 dB in PSNR over the latest methods on the ICVL, Chikusei, and Houston datasets, respectively. Please refer to Tables I-III in the paper for complete results across all datasets, rotation angles, and evaluation metrics (RMSE, UIQI).

🚀 Quick Start

1. Environment Setup

The code is implemented in PyTorch. The experiments in the paper were conducted on an NVIDIA 2080Ti GPU.
# Create and activate a conda environment (recommended)
conda create -n dunet python=3.8
conda activate dunet

# Install PyTorch (Please choose the correct version for your CUDA from https://pytorch.org/)
# Example for CUDA 11.8:
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy matplotlib tqdm scikit-image


2. Installation & Data Preparation

1.  Clone the repository
    git clone https://github.com/your_username/DUNET.git
    cd DUNET
    
2.  Prepare Datasets
    ◦   Download the benchmark datasets:

        ▪   http://icvl.cs.bgu.ac.il/hyperspectral/

        ▪   http://naotoyokoya.com/Download.html

        ▪   https://hyperspectral.ee.uh.edu/?page_id=1075

    ◦   Organize the data. We provide a script to preprocess the data and generate training/validation patches following the Wald's protocol as described in the paper (Section III-G and [28]).
    # Modify the paths inside the script first
    python data/prepare_data.py
    

3. Training

Train the DUNET model on the ICVL dataset (default: 5 iteration stages, 3 HAT blocks, rotation angle 2°).
python train.py --dataset ICVL --data_path ./data/ICVL/train --save_dir ./checkpoints --epochs 1000 --lr 1e-4

Key Training Details (from paper Section III-G & IV):
•   Loss Function: Combines L1 loss for HR-HSI, Fourier (FFT) loss (τ1=0.1), and LR-HSI loss (τ2=0.5). See Eq. (32).

•   Optimizer: Adam (β1=0.9, β2=0.999).

•   Batch Size: 32.

•   Training Samples: Generated via Wald's protocol from high-res ground truth.

4. Test & Evaluation

Evaluate a trained model on the test set. The script will compute MPSNR, SAM, ERGAS, UIQI, MSSIM, and RMSE.
python test.py --dataset ICVL --data_path ./data/ICVL/test --model_path ./checkpoints/best_model.pth


5. Predict on Your Data

Use a trained model to fuse your own unregistered LR-HSI and HR-MSI pair.
python predict.py --lrhsi_path ./your_data/lr.hdr --hrmsi_path ./your_data/msi.tif --model_path ./checkpoints/best_model.pth --output_path ./fused_result.hdr


📁 Project Structure


DUNET/
├── data/                   # Data loading and preprocessing utilities
│   ├── prepare_data.py    # Script to prepare datasets (Wald's protocol)
│   └── datasets.py        # PyTorch Dataset classes
├── models/                # Network architecture
│   ├── dunet.py          # Main DUNET model definition
│   ├── hat.py            # Hybrid Attention Transformer (HAT) block
│   ├── sf_block.py       # Spatial-Frequency Block (SFB)
│   └── proximal.py       # Deep proximal operator (HAT + SFB)
├── utils/
│   ├── mi_registration.py # Mutual Information initialization module
│   ├── metrics.py         # Evaluation metrics (MPSNR, SAM, ERGAS, etc.)
│   └── logger.py          # Training logger
├── configs/               # Configuration files for different datasets
├── checkpoints/           # Directory for saving trained models
├── results/               # Directory for saving test outputs
├── train.py              # Main training script
├── test.py               # Main testing and evaluation script
├── predict.py            # Inference script for new data
├── requirements.txt      # Python dependencies
└── README.md             # This file


📜 Citation

If you find this code or our paper useful for your research, please cite:

```bibtex
@article{fang2024dunet,
  title={Deep Unfolding Network Enhanced by Transformer Priors for Unregistered Hyperspectral and Multispectral Image Fusion},
  author={Fang, Jian and Yang, Jingxiang and Khader, Abdolraheem and Xiao, Liang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--16},
  year={2024},
  publisher={IEEE},
  doi={10.1109/TGRS.2024.3460186}
}
```

📄 License

This project is open-sourced under the MIT License. See the LICENSE file for details.

⁉️ Contact

For any questions or discussions regarding the paper or code, feel free to open an issue or contact:
•   Jian Fang: fangjian@njust.edu.cn

•   Jingxiang Yang: yang123jx@njust.edu.cn

Acknowledgments: This work was supported by the National Natural Science Foundation of China (Grants 61871226, 61571230, 62471235, 62001226), the Jiangsu Provincial Social Developing Project (Grant BE2018727), and the Natural Science Foundation of Jiangsu Province (Grant BK20200465).
