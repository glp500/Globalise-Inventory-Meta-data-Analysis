# Simple VOC Archive Page Classifier

A streamlined tool for classifying scanned pages from VOC (Dutch East India Company) archives with support for multiple vision-language models.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python classify.py <input_directory> [--model MODEL] [--output OUTPUT]
```

### Quick Start Examples

```bash
# Use default model (Qwen2-VL-7B)
python classify.py /path/to/scanned/pages

# Test different model sizes
python classify.py /path/to/images --model qwen2-vl-2b
python classify.py /path/to/images --model llava-1.6-7b

# Use custom HuggingFace model
python classify.py /path/to/images --model "microsoft/kosmos-2-patch14-224"

# List available model shortcuts
python classify.py --list-models
```

## Available Models

### Built-in Shortcuts
- `qwen2-vl-2b` - Qwen/Qwen2-VL-2B-Instruct (fastest)
- `qwen2-vl-7b` - Qwen/Qwen2-VL-7B-Instruct (default, balanced)
- `qwen2-vl-72b` - Qwen/Qwen2-VL-72B-Instruct (most accurate)
- `llava-1.6-7b` - llava-hf/llava-v1.6-vicuna-7b-hf
- `llava-1.6-13b` - llava-hf/llava-v1.6-vicuna-13b-hf
- `instructblip-7b` - Salesforce/instructblip-vicuna-7b
- `instructblip-13b` - Salesforce/instructblip-vicuna-13b

### Custom Models
Use any HuggingFace vision-language model by providing its full name:
```bash
python classify.py /path/to/images --model "Qwen/Qwen2-VL-2B-Instruct"
python classify.py /path/to/images --model "microsoft/kosmos-2-patch14-224"
```

## Categories

The classifier recognizes these historical document types:
- `single_column` - Text in single column layout
- `two_column` - Text in two columns
- `table_full` - Page dominated by tables
- `table_partial` - Mixed table and text
- `marginalia` - Text with margin notes
- `two_page_spread` - Two pages in one image
- `extended_foldout` - Wide unfolding pages
- `illustration` - Drawings, maps, diagrams
- `title_page` - Title or cover pages
- `blank` - Empty pages
- `seal_signature` - Official seals/signatures
- `mixed_layout` - Complex mixed layouts
- `damaged_partial` - Damaged pages
- `index_list` - Indexes and lists

## Output

Results are saved as JSON with this format:
```json
[
  {
    "image_path": "/path/to/image.jpg",
    "category": "single_column",
    "raw_response": "single_column",
    "status": "success",
    "model": "Qwen/Qwen2-VL-7B-Instruct"
  }
]
```

## Model Selection Guide

**For Speed**: Use `qwen2-vl-2b` - Fastest inference, good for quick testing
**For Balance**: Use `qwen2-vl-7b` (default) - Good accuracy vs speed tradeoff
**For Accuracy**: Use `qwen2-vl-72b` - Highest accuracy, requires more GPU memory
**For Comparison**: Try `llava-1.6-7b` or `instructblip-7b` - Alternative architectures

## Requirements

- Python 3.8+
- GPU acceleration: CUDA, Apple Metal (MPS), or CPU fallback
- RAM: 4-8GB for 2B models, 8-16GB for 7B models, 32GB+ for 72B models

## GPU Support

The classifier automatically detects and uses the best available acceleration:

### NVIDIA CUDA
```bash
# Verify CUDA is available
python test_gpu.py

# Run with CUDA acceleration
python classify.py /path/to/images --model qwen2-vl-7b
```

### Apple Metal (MPS)
```bash
# Verify Metal is available
python test_gpu.py

# Run with Metal acceleration (macOS)
python classify.py /path/to/images --model qwen2-vl-7b
```

### CPU Fallback
Automatically used if no GPU acceleration is available.

## SSH + VSCode Usage

Perfect for remote development through SSH connections:

1. **Connect via SSH in VSCode**: Open remote folder in VSCode
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Test GPU**: `python test_gpu.py`
4. **Run classifier**: `python classify.py /path/to/images --model qwen2-vl-2b`

The program will automatically use CUDA if available on the remote server, or fallback gracefully to CPU.