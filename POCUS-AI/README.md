# POCUS-AI

Deep learning model development for handheld and transvaginal ultrasound imaging analysis, with a focus on radiomics and self-supervised learning methods.

## Project Overview

POCUS-AI is a research project aimed at developing advanced deep learning models for point-of-care ultrasound (POCUS) and transvaginal ultrasound imaging. The project implements state-of-the-art techniques in:

- Radiomics feature extraction and analysis
- Self-supervised learning methods
- Deep learning model development
- Medical image processing and analysis

## Repository Structure

```
POCUS-AI/
├── data/                  # Dataset manifests and processed data (no raw data)
├── docs/                  # Documentation and design specifications
├── notebooks/            # Jupyter notebooks for analysis and demonstrations
├── src/
│   ├── pocus_ai/        # Main package directory
│   │   ├── models/      # Deep learning model implementations
│   │   ├── radiomics/   # Radiomics feature extraction and analysis
│   │   └── ssl/         # Self-supervised learning implementations
│   └── scripts/         # Utility scripts and tools
├── experiments/          # Training configurations and experiment results
└── tests/               # Unit tests and integration tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/POCUS-AI.git
cd POCUS-AI
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate pocus-ai
```

## Usage

Detailed usage instructions and examples can be found in the `notebooks/` directory and the documentation in `docs/`.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and feedback, please open an issue in the GitHub repository.