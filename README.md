# Oil Spill Detection in SAR Imagery

An end-to-end deep learning pipeline for automated detection and semantic segmentation of marine oil spills using dual-polarization Synthetic Aperture Radar (SAR) imagery.

---

## Overview

Synthetic Aperture Radar (SAR) sensors provide cloud-penetrating, day-and-night observation capabilities critical for ocean surveillance. Marine oil spills dampen surface capillary waves, resulting in low backscatter areas that appear dark in SAR imagery. However, natural meteorological and oceanographic phenomena—such as biogenic slicks, low-wind zones, and internal waves—create dark patches known as "lookalikes" that lead to false positives.

This project implements a complete pipeline that processes dual-channel Sentinel-1 SAR data (VV and VH polarizations), extracts balanced image patches, and trains a custom U-Net segmentation network with validity-masked loss functions to accurately segment oil spills while suppressing lookalikes.

---

## Key Features

- **Dual-Polarization Processing**: Utilizes both VV (vertical transmit/vertical receive) and VH (vertical transmit/horizontal receive) channels to exploit polarization sensitivity differences between crude oil and sea surface roughness.
- **Tiling and Sampling Engine**: Slides 512x512 windows with 128-pixel overlap across large-scale satellite scenes, balancing positive oil patches against clean ocean and lookalike samples to mitigate extreme class imbalance.
- **Validity-Aware Masking**: Accounts for swath borders and sensor nodata regions by generating pixel-level validity masks, preventing invalid backscatter values from contaminating training losses.
- **Custom Loss Formulations**: Implements masked Binary Cross-Entropy and masked Dice loss functions that restrict gradient computation strictly to valid oceanic pixels.
- **Mixed-Precision Training**: Integrated with PyTorch Automatic Mixed Precision (AMP) and learning rate scheduling (`ReduceLROnPlateau`) for efficient convergence and memory management.

---

## Repository Structure

```text
Oil-Spill-Detection/
├── configs/
│   └── config.yaml                 # Central configuration for paths, bands, and hyperparameters
├── models/
│   ├── best_unet.pth               # Model checkpoint with best validation metric (created upon training)
│   ├── latest_checkpoint.pth       # Resume checkpoint for interrupted training runs
│   └── training_history.csv        # Epoch-by-epoch metrics log
├── notebooks/
│   ├── 1_dataset_exploration.ipynb # Exploratory data analysis for raw SAR scenes
│   └── 02_model_evaluation.ipynb   # Qualitative and quantitative model assessment
├── src/
│   ├── data/
│   │   ├── check_valid_pixel_distribution.py # Valid pixel ratio diagnostics
│   │   ├── check_zero_nodata.py              # Zero and nodata value scanner
│   │   ├── dataset.py                        # PyTorch Dataset and DataLoader implementations
│   │   ├── dataset_analysis.py               # Statistical distribution of dataset classes
│   │   ├── dataset_inspector.py              # Single-patch inspection utility
│   │   ├── data_visualization.py             # Visualizer for SAR backscatter and masks
│   │   ├── preprocess.py                     # Sliding-window patch extraction and manifest generation
│   │   └── verify_preprocessed.py            # Patch integrity and format validation
│   ├── evaluation/
│   │   └── evaluate.py                       # Evaluation scripts for test sets
│   ├── inference/
│   │   └── predict.py                        # Full-scene sliding-window inference and reconstruction
│   ├── models/
│   │   └── unet.py                           # 4-level U-Net architecture in PyTorch
│   └── training/
│       ├── batch_benchmark.py                # DataLoader throughput and batch transfer benchmarking
│       ├── losses.py                         # Masked BCE, Masked Dice, and BCEDiceLoss implementations
│       └── train.py                          # Training loop with AMP, metrics logging, and checkpointing
├── .gitignore
├── README.md
└── requirements.txt
```

---

## System Architecture

### Model Architecture (`src/models/unet.py`)

The network is a custom U-Net variant tailored for multi-channel SAR inputs:
- **Input Layer**: 2 channels (VV and VH backscatter in dB, min-max normalized to `[0, 1]`).
- **Encoder**: 4 levels of Double Convolution blocks (3x3 Conv -> BatchNorm -> ReLU) with feature dimensions `[64, 128, 256, 512]`, downsampled via 2x2 Max Pooling.
- **Bottleneck**: Double Convolution block expanding to 1024 feature dimensions.
- **Decoder**: 4 levels of Up-convolutions (ConvTranspose2d) concatenating corresponding encoder features via skip connections, followed by Double Convolution blocks.
- **Output Layer**: 1x1 Convolution yielding raw logits for binary pixel classification (oil spill vs. background).

### Masked Loss Formulation (`src/training/losses.py`)

Satellite swaths frequently contain non-data border regions or masked land areas. Standard loss functions would bias the model on these artifacts. The training pipeline uses masked loss formulations:

$$\mathcal{L}_{\text{BCE, masked}} = \frac{\sum (l_i \cdot m_{\text{valid}, i})}{\sum m_{\text{valid}, i} + \epsilon}$$

$$\mathcal{L}_{\text{Dice, masked}} = 1 - \frac{2 \sum (p_i \cdot y_i \cdot m_{\text{valid}, i}) + \epsilon}{\sum (p_i \cdot m_{\text{valid}, i}) + \sum (y_i \cdot m_{\text{valid}, i}) + \epsilon}$$

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BCE, masked}} + \mathcal{L}_{\text{Dice, masked}}$$

---

## Configuration

Dataset directories, preprocessing parameters, and training settings are configured in `configs/config.yaml`:

```yaml
dataset:
  oil:
    images: "path/to/oil_images"
    masks: "path/to/oil_masks"
  no_oil:
    images: "path/to/no_oil_images"
    masks: "path/to/no_oil_masks"
  lookalike:
    images: "path/to/lookalike_images"
    masks: "path/to/lookalike_masks"

preprocessing:
  channels: [VV, VH]
  patch_size: 512
  overlap: 128
  normalize: true
  normalization:
    vv_min: -44.086868
    vv_max: -12.757253
    vh_min: -35.326511
    vh_max: -6.280492
  invalid_value: 0
  patch_sampling:
    min_oil_pixels: 1
    max_background_patches_per_image: 5
    max_no_oil_patches_per_image: 5
    max_lookalike_patches_per_image: 5
    keep_all_oil_patches: true

training:
  batch_size: 4
  epochs: 50
  learning_rate: 0.0001
  loss:
    type: "bce_dice"
  checkpoint:
    save_best: true
    path: "models/best_unet.pth"

device:
  type: "auto"
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for training)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Serion89/Oil-Spill-Detection.git
   cd Oil-Spill-Detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install PyTorch with CUDA support matching your system drivers:
   ```bash
   # Example for CUDA 12.1:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

---

## Usage Workflow

### 1. Data Preprocessing

Configure image paths in `configs/config.yaml`, then run the patch extraction and normalization routine:

```bash
python src/data/preprocess.py
```

This generates:
- Tiled `.npy` image and mask patches under `data/patches/`.
- A unified metadata split catalog in `data/manifests/patch_manifest.csv`.

To verify the integrity of the generated dataset:
```bash
python src/data/verify_preprocessed.py
```

### 2. Model Training

Launch model training with automatic checkpointing and metric tracking:

```bash
python src/training/train.py
```

- Training resumes automatically if `models/latest_checkpoint.pth` is present.
- The best performing model weights based on validation loss are saved to `models/best_unet.pth`.
- Training progress is recorded to `models/training_history.csv`.

### 3. DataLoader Performance Benchmarking

Benchmark I/O and batch transfer throughput across worker configurations:

```bash
python src/training/batch_benchmark.py
```

---

## Training Metrics & Progress

Performance snapshot across validation checkpoints (logged in `models/training_history.csv`):

| Epoch | Train Loss | Val Loss | Dice Score | IoU | Precision | Recall | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.4241 | 0.4110 | 0.6033 | 0.4320 | 0.4700 | 0.8422 | 1.0e-4 |
| 5 | 0.2055 | 0.2487 | 0.5853 | 0.4137 | 0.4653 | 0.7887 | 1.0e-4 |
| 10 | 0.1718 | 0.2075 | 0.6327 | 0.4628 | 0.5279 | 0.7895 | 5.0e-5 |
| 13 | 0.1686 | 0.2164 | 0.6280 | 0.4578 | 0.5342 | 0.7620 | 2.5e-5 |
| 15 | 0.1607 | 0.2369 | 0.5674 | 0.3960 | 0.4249 | 0.8535 | 2.5e-5 |

Best recorded validation loss achieved at Epoch 10: **0.2075**, yielding a **0.6327 Dice score** and **0.7895 Recall**.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
