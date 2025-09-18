#!/bin/bash
# Server Environment Fix Script
# Run this on your server to fix the Qwen2.5-VL model issues

echo "🔧 Fixing VOC Classifier Server Environment..."
echo ""

# Check if conda environment exists
if conda env list | grep -q "voc-classifier"; then
    echo "✅ Found voc-classifier environment"
    echo "📦 Updating packages..."

    # Activate environment and update packages
    conda activate voc-classifier
    pip install --upgrade transformers>=4.45.0 accelerate>=0.25.0

    echo ""
    echo "🧪 Verifying installation..."
    python -c "
import transformers, accelerate
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ accelerate: {accelerate.__version__}')
"

    echo ""
    echo "🎯 Testing model shortcuts..."
    python classify.py --list-models

    echo ""
    echo "🚀 Environment is ready! You can now run:"
    echo "CUDA_VISIBLE_DEVICES=0 python classify.py /path/to/images --model qwen2.5-vl-32b"

else
    echo "❌ voc-classifier environment not found"
    echo "📦 Creating new environment..."

    # Create new environment
    conda env create -f environment.yml
    conda activate voc-classifier

    echo ""
    echo "✅ New environment created and ready!"

fi