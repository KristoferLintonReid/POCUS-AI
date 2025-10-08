# POCUS-AI

Advanced AI models for Point-of-Care Ultrasound (POCUS) imaging analysis, with focus on radiomics and deep learning techniques.

## 📋 Overview

POCUS-AI is an open-source platform for developing and deploying artificial intelligence solutions for Point-of-Care Ultrasound (POCUS) imaging. The project focuses on:

- **Radiomics analysis**: Extracting quantitative features from medical images for diagnostic and prognostic modeling
- **Deep learning models**: Implementing state-of-the-art neural networks for segmentation, classification, and detection
- **Self-supervised learning**: Leveraging unlabeled ultrasound data to improve model performance
- **Clinical validation**: Tools for model evaluation in clinical scenarios
- **Deployment frameworks**: Solutions for deploying models in resource-constrained environments

## 🔍 Key Features

- Comprehensive radiomics feature extraction using PyRadiomics
- Pre-trained deep learning models for common POCUS tasks
- Self-supervised learning implementations for representation learning
- Interpretability and explainability tools
- Optimized inference for resource-constrained devices
- Extensive documentation and educational notebooks

## 🏗️ Repository Structure

```
POCUS-AI/
├── data/                  # Dataset manifests and processing scripts (no raw data)
├── docs/                  # Documentation and design specifications
│   ├── images/           # Images for documentation
│   ├── tutorials/        # Step-by-step guides
│   └── api/              # API documentation
├── notebooks/            # Jupyter notebooks for analysis and demonstrations
│   ├── introduction.ipynb         # Introduction to the project
│   ├── radiomics_extraction.ipynb # Detailed radiomics workflow
│   ├── model_training.ipynb       # Model training examples
│   └── clinical_validation.ipynb  # Validation protocols
├── src/
│   ├── pocus_ai/         # Main package directory
│   │   ├── models/       # Deep learning model implementations
│   │   │   ├── segmentation/    # Segmentation models
│   │   │   ├── classification/  # Classification models  
│   │   │   └── detection/       # Detection models
│   │   ├── radiomics/    # Radiomics feature extraction and analysis
│   │   │   ├── extractors/      # Feature extractors
│   │   │   ├── preprocessors/   # Image preprocessing
│   │   │   └── analyzers/       # Statistical analysis
│   │   ├── ssl/          # Self-supervised learning implementations
│   │   │   ├── contrastive/     # Contrastive learning methods
│   │   │   └── generative/      # Generative methods
│   │   └── utils/        # Utility functions
│   └── scripts/          # Utility scripts and tools
├── experiments/          # Training configurations and experiment results
└── tests/                # Unit tests and integration tests
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Method 1: Using Conda (recommended)

```bash
# Clone the repository
git clone https://github.com/KristoferLintonReid/POCUS-AI.git
cd POCUS-AI

# Create and activate the conda environment
conda env create -f environment.yml
conda activate pocus-ai

# Install the package in development mode
pip install -e .
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/KristoferLintonReid/POCUS-AI.git
cd POCUS-AI

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 📊 Radiomics with PyRadiomics

POCUS-AI integrates PyRadiomics for comprehensive feature extraction from ultrasound images. The framework enables:

1. **First-order statistics**: Intensity-based features (mean, variance, skewness, etc.)
2. **Shape features**: Morphological characteristics of regions of interest
3. **Texture features**: GLCM, GLRLM, GLSZM, NGTDM, and GLDM matrices
4. **Wavelet features**: Multi-resolution analysis of images
5. **Custom features**: Domain-specific features for POCUS imaging

### Example Usage:

```python
from pocus_ai.radiomics import extract_features

# Load image and segmentation mask
image = load_image('patient001.dcm')
mask = load_mask('patient001_mask.nii.gz')

# Extract radiomics features
features = extract_features(image, mask)

# Analyze features
print(features)
```

## 🧠 Deep Learning Models

The repository includes several pre-trained models:

1. **Segmentation Models**:
   - U-Net variants for anatomical segmentation
   - DeepLabV3+ for detailed boundary delineation
   - Transformer-based segmentation models

2. **Classification Models**:
   - Disease classification networks
   - Abnormality detection models
   - Transfer learning implementations

3. **Detection Models**:
   - Object detection for anatomical landmarks
   - Lesion detection frameworks

### Example Usage:

```python
from pocus_ai.models import UNetSegmentation

# Initialize model
model = UNetSegmentation(pretrained=True)

# Load image
image = load_image('patient001.dcm')

# Perform segmentation
mask = model.predict(image)

# Save or visualize results
save_mask(mask, 'patient001_segmentation.nii.gz')
```

## 🚀 Getting Started

See the [Introduction Notebook](notebooks/introduction.ipynb) for a walkthrough of the core functionality.

For a complete guide to radiomics feature extraction, see the [Radiomics Extraction Notebook](notebooks/radiomics_extraction.ipynb).

## 📚 Documentation

Full documentation is available in the `docs/` directory.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


