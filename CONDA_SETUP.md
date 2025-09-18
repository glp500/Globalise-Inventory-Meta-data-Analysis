# Conda Environment Setup Guide

## Quick Start

### 1. Choose Your Environment

#### CUDA GPU Systems (Linux/Windows with NVIDIA GPU)
```bash
conda env create -f environment.yml
conda activate voc-classifier
```

#### CPU-Only Systems
```bash
conda env create -f environment-cpu.yml
conda activate voc-classifier-cpu
```

#### macOS Systems (Intel/Apple Silicon)
```bash
conda env create -f environment-mac.yml
conda activate voc-classifier-mac
```

### 2. Test Your Setup
```bash
python test_conda_setup.py
python test_gpu.py
```

### 3. Run Classifier
```bash
python classify.py /path/to/images --model qwen2-vl-2b
```

## Environment Files Explained

### `environment.yml` - CUDA GPU Support
- PyTorch with CUDA 11.8 support
- Optimized for NVIDIA GPUs
- Best performance for large models

### `environment-cpu.yml` - CPU-Only
- PyTorch CPU-only version
- Smaller download size
- Works on any system without GPU

### `environment-mac.yml` - macOS Optimized
- PyTorch with Metal Performance Shaders (MPS)
- Optimized for Apple Silicon/Intel Macs
- GPU acceleration via Metal

## Troubleshooting

### Environment Creation Fails
```bash
# Try with explicit conda-forge channel
conda env create -f environment-mac.yml -c conda-forge

# Or create manually
conda create -n voc-classifier python=3.11
conda activate voc-classifier
conda install pytorch torchvision pillow -c pytorch
pip install transformers>=4.35.0
```

### Package Conflicts
```bash
# Remove and recreate environment
conda env remove -n voc-classifier
conda env create -f environment.yml
```

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, use CPU environment instead
conda env create -f environment-cpu.yml
```

## Manual Installation

If environment files don't work, install manually:

```bash
# Create environment
conda create -n voc-classifier python=3.11

# Activate
conda activate voc-classifier

# Install PyTorch (choose based on your system)
# CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# CPU only
conda install pytorch torchvision cpuonly -c pytorch

# macOS
conda install pytorch torchvision -c pytorch

# Install other dependencies
conda install pillow -c conda-forge
pip install transformers>=4.35.0
```

## SSH + VSCode Workflow

1. **Copy environment file to remote server**
2. **Create environment on remote server**:
   ```bash
   conda env create -f environment.yml
   ```
3. **Connect via SSH in VSCode**
4. **Select conda environment** in VSCode Python interpreter
5. **Run classifier**:
   ```bash
   conda activate voc-classifier
   python classify.py /path/to/images
   ```