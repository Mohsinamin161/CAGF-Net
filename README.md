# CAGF-Net: Cross-Attention and Gene Fusion Network for Early Parkinson's Disease Prediction

This repository contains the official implementation of **CAGF-Net**, a multi-modal deep learning framework that integrates 3D MRI and SNP data to classify early Parkinson's disease stages (HC, Prodromal, PD).

## Project Structure
- `data/`: Data loading and preprocessing modules.
- `models/`: Neural network architectures (MedicalNet, SNP Transformer, fusion models).
- `training/`: Training loops and utilities.
- `explainability/`: SHAP and Grad-CAM analysis.
- `scripts/`: Executable pipelines.
- `config/`: Configuration files.

## Setup
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your data (see instructions in `data/`).
4. Run the scripts in order (see `scripts/README.md`).

## License
MIT
