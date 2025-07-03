# ENTRep 2025 Track 2 - Vector Field Training

This repository contains the training code for ENTRep 2025 Track 2, implementing a Vector Field model with Flow Matching for entity recognition using CLIP embeddings.

## Overview

The model uses:
- **CLIP ViT-B/32** for image embedding extraction
- **Vector Field Model** with Flow Matching for learning time-dependent transformations
- **Triplet Loss** with MultiSimilarityLoss for training
- **Gaussian Fourier Projection** for time encoding
- **Euler Integration** with RK4 method for flow simulation

## Files

- `train.py` - Main training script
- `models.py` - Vector Field and Gaussian Fourier Projection model definitions
- `utils.py` - Data processing and utility functions
- `requirements.txt` - Python dependencies

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
```bash
# The training script will attempt to download the dataset automatically
# Or manually download from: https://drive.google.com/uc?id=1I56vd3aWsy_nkY6zdXk4faIM6NSS5mer
```

## Usage

### Training

Run the training script:
```bash
python train.py
```

The script will:
1. Load and embed images using CLIP
2. Create class mappings and parent-child relationships
3. Generate triplets for training
4. Train the Vector Field model
5. Save the trained model as `vf_model.pth`

### Model Architecture

The Vector Field model consists of:
- **Input Normalization**: LayerNorm on input embeddings
- **Time Encoding**: Gaussian Fourier Projection for time `t`
- **Multi-Head MLPs**: 4 independent MLP heads for processing
- **Residual Connections**: Learnable residual scaling for stability

### Training Parameters

- **Embedding Dimension**: 512 (CLIP ViT-B/32)
- **Learning Rate**: 1e-4 with AdamW optimizer
- **Batch Size**: 32
- **Integration Steps**: 10 (Euler with RK4)
- **Early Stopping**: 40 epochs patience
- **Warmup**: 20 epochs

## Dataset

The dataset should be organized in CIFAR-10 style format:
```
data_cifar10_style_public/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### Classes

The model recognizes 7 classes:
- `ear-left` (3)
- `ear-right` (2) 
- `nose-left` (1)
- `nose-right` (0)
- `throat` (6)
- `vc-closed` (4)
- `vc-open` (5)

## Model Output

The training produces:
- `best_vf.pt` - Best model checkpoint during training
- `vf_model.pth` - Final trained model

## Key Features

1. **Hard Negative Mining**: Uses predefined negative class relationships
2. **Parent-Child Relationships**: Finds most similar images within classes
3. **Flow Matching**: Time-dependent vector field learning
4. **Robust Training**: Early stopping, gradient clipping, LR scheduling

## Dependencies

- PyTorch >= 1.13.0
- CLIP (OpenAI)
- pytorch-metric-learning
- scikit-learn
- Pillow
- tqdm
- matplotlib (optional, for visualization)

## Citation

This implementation is based on the ENTRep 2025 Track 2 challenge requirements and uses Flow Matching techniques for embedding transformation learning.