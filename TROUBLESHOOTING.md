# Troubleshooting Guide

## Qwen2.5-VL Model Issues

### Error: "Unrecognized configuration class Qwen2_5_VLConfig"

This error occurs when your `transformers` library version doesn't support Qwen2.5-VL models.

#### Solution 1: Update transformers (Recommended)
```bash
# In your conda environment
conda activate voc-classifier
pip install --upgrade transformers>=4.45.0

# Or update conda environment
conda env update -f environment.yml --prune
```

#### Solution 2: Use older Qwen2-VL models instead
```bash
# Instead of: Qwen/Qwen2.5-VL-32B-Instruct
# Use:
python classify.py /path/to/images --model qwen2-vl-7b
python classify.py /path/to/images --model Qwen/Qwen2-VL-7B-Instruct
```

#### Solution 3: Recreate environment with latest versions
```bash
# Remove old environment
conda env remove -n voc-classifier

# Create new with latest versions
conda env create -f environment.yml
conda activate voc-classifier

# Test the model
python classify.py --list-models
```

## Server Usage Commands

Once your transformers library is updated, you can use the newer models:

```bash
# Activate environment
conda activate voc-classifier

# Use Qwen2.5-VL shortcuts
python classify.py /path/to/images --model qwen2.5-vl-3b   # Smallest, fastest
python classify.py /path/to/images --model qwen2.5-vl-7b   # Balanced
python classify.py /path/to/images --model qwen2.5-vl-32b  # Most accurate

# Or use full model names
python classify.py /path/to/images --model "Qwen/Qwen2.5-VL-32B-Instruct"

# With GPU specification
CUDA_VISIBLE_DEVICES=0 python classify.py /path/to/images --model qwen2.5-vl-32b
```

## Other Common Issues

### CUDA Out of Memory
```bash
# Use smaller model
python classify.py /path/to/images --model qwen2.5-vl-3b

# Or specify CPU
python classify.py /path/to/images --model qwen2.5-vl-7b --device cpu
```

### Model Download Issues
```bash
# Set cache directory
export HF_HOME=/path/to/large/storage
export TRANSFORMERS_CACHE=/path/to/large/storage

# Or use offline mode if model already downloaded
export TRANSFORMERS_OFFLINE=1
```

### Import Errors
```bash
# Check environment
conda list transformers
python -c "import transformers; print(transformers.__version__)"

# Should show >= 4.45.0 for Qwen2.5-VL support
```