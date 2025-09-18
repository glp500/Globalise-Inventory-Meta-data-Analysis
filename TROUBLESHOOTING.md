# Troubleshooting Guide

## ⚠️ Server Environment Issues (Most Common)

### Error: "requires accelerate" + "Unrecognized configuration class"

Your server environment is missing required packages. **This is the exact error you encountered.**

#### 🚀 Quick Fix (Run on your server):
```bash
# Activate your conda environment
conda activate voc-classifier

# Update all required packages
pip install --upgrade transformers>=4.45.0 accelerate>=0.25.0

# Verify versions
python -c "import transformers, accelerate; print(f'transformers: {transformers.__version__}'); print(f'accelerate: {accelerate.__version__}')"

# Should show: transformers: 4.45.0+ and accelerate: 0.25.0+
```

#### 🔄 Alternative: Recreate Environment (More reliable)
```bash
# Remove old environment
conda env remove -n voc-classifier

# Create fresh environment with updated packages
conda env create -f environment.yml
conda activate voc-classifier

# Test setup
python classify.py --list-models
```

## Qwen2.5-VL Model Issues

### Error: "Unrecognized configuration class Qwen2_5_VLConfig"

This occurs when `transformers` library version doesn't support Qwen2.5-VL models.

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

### Error: "requires accelerate"

The `accelerate` library is needed for GPU memory optimization.

#### Solution:
```bash
conda activate voc-classifier
pip install accelerate>=0.25.0
```

## ✅ Server Usage Commands (After fixing environment)

Once your environment is updated, these commands will work:

```bash
# Activate environment
conda activate voc-classifier

# Your original command should now work:
CUDA_VISIBLE_DEVICES=0 python classify.py /home/gavinl/Globalise-Inventory-Meta-data-Analysis/data/input --model Qwen/Qwen2.5-VL-32B-Instruct

# Or use shortcuts:
CUDA_VISIBLE_DEVICES=0 python classify.py /path/to/images --model qwen2.5-vl-32b   # Same model, shorter name
CUDA_VISIBLE_DEVICES=0 python classify.py /path/to/images --model qwen2.5-vl-7b    # Faster, less memory
CUDA_VISIBLE_DEVICES=0 python classify.py /path/to/images --model qwen2.5-vl-3b    # Fastest

# Test that everything is working:
python classify.py --list-models
```

## 🎯 Expected Output After Fix

After running the environment update, you should see:
```bash
INFO: CUDA GPU detected: NVIDIA A40
INFO: Using device: cuda
INFO: Loading Qwen/Qwen2.5-VL-32B-Instruct (family: qwen2.5-vl)
INFO: Using accelerate for automatic device mapping
INFO: Loading with dtype: torch.float16
INFO: Model loaded successfully
```

Instead of the error messages you saw before.

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