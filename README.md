# VOC Archive Page Classifier

A professional-grade tool for analyzing scanned pages from VOC (Dutch East India Company) archives using advanced vision-language models. Features **multi-dimensional classification** with detailed explanations for each analysis decision.

## ✨ Key Features

- **Multi-Question Analysis**: 10 comprehensive classification dimensions with explanations
- **Historical Expertise**: VOC-specific prompts for 17th-19th century documents
- **Multiple Model Support**: Qwen2-VL, Qwen2.5-VL, LLaVA, InstructBLIP architectures
- **Professional Output**: Research-quality analysis suitable for digital humanities
- **Robust Processing**: Smart device detection, error recovery, and batch processing

## Installation

### Option 1: Conda Environment (Recommended)

#### For CUDA GPU Systems
```bash
# Create and activate conda environment with CUDA support
conda env create -f environment.yml
conda activate voc-classifier
```

#### For CPU-Only Systems
```bash
# Create and activate CPU-only environment
conda env create -f environment-cpu.yml
conda activate voc-classifier-cpu
```

#### For macOS (Apple Silicon/Intel)
```bash
# Create and activate macOS environment with Metal support
conda env create -f environment-mac.yml
conda activate voc-classifier-mac
```

### Option 2: Pip Installation
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

**Qwen2.5-VL Series** (Latest, Recommended):
- `qwen2.5-vl-3b` - Qwen/Qwen2.5-VL-3B-Instruct (fastest, 6GB VRAM)
- `qwen2.5-vl-7b` - Qwen/Qwen2.5-VL-7B-Instruct (balanced, 14GB VRAM)
- `qwen2.5-vl-32b` - Qwen/Qwen2.5-VL-32B-Instruct (highest quality, 42GB VRAM)

**Qwen2-VL Series** (Stable):
- `qwen2-vl-2b` - Qwen/Qwen2-VL-2B-Instruct (lightweight)
- `qwen2-vl-7b` - Qwen/Qwen2-VL-7B-Instruct (default fallback)
- `qwen2-vl-72b` - Qwen/Qwen2-VL-72B-Instruct (very large)

**Alternative Architectures**:
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

## Analysis Dimensions

Each page receives comprehensive analysis across **10 key dimensions**:

### 1. Content Presence
- `blank` - Mostly empty page
- `non_blank` - Contains meaningful content

### 2. Page Format
- `standard_page` - Normal single page
- `extended_foldout` - Wide unfolding page
- `two_page_spread` - Two pages side by side

### 3. Text Layout
- `single_column` - Single column text
- `two_column` - Two column layout
- `mixed_layout` - Complex arrangement

### 4. Tabular Content
- `table_full` - Page dominated by tables
- `table_partial` - Mixed table and text
- `table_none` - No tabular content

### 5. Marginalia
- `marginalia` - Contains margin notes
- `no_marginalia` - No margin annotations

### 6. Visual Elements
- `illustration` - Contains drawings/diagrams
- `no_illustration` - Text-only content

### 7. Document Type
- `title_page` - Title/cover page
- `non_title_page` - Regular content page

### 8. Authentication Marks
- `seal_signature` - Contains seals/signatures
- `no_signatures` - No authentication marks

### 9. Physical Condition
- `damaged_partial` - Shows damage/deterioration
- `no_damage` - Good condition

### 10. Content Structure
- `index_list` - Index/catalog format
- `no_index_list` - Regular text format

## Enhanced Output Format

Results include comprehensive analysis with explanations for each dimension:

```json
[
  {
    "image_path": "/path/to/image.jpg",
    "summary": "This VOC archive page displays handwritten Dutch administrative text in a formal two-column layout typical of 17th-century company records. The writing appears to be in secretary hand script with iron gall ink that has aged to a brown color.",
    "classifications": {
      "empty_status": {
        "answer": "non_blank",
        "explanation": "Page contains substantial handwritten text across both columns with clear administrative content."
      },
      "text_layout": {
        "answer": "two_column",
        "explanation": "Text is clearly organized into two distinct columns with visible vertical margin, consistent with formal administrative layout."
      },
      "marginalia": {
        "answer": "marginalia",
        "explanation": "Several marginal annotations visible on the left side, appearing to be contemporary additions or cross-references."
      }
    },
    "raw_response": "Complete model response with all questions and answers...",
    "status": "success",
    "model": "Qwen/Qwen2.5-VL-7B-Instruct"
  }
]
```

### Sample Analysis Output

Each classification includes:
- **Answer**: The determined category
- **Explanation**: Detailed reasoning for the decision
- **Summary**: Overall visual description of the document
- **Raw Response**: Complete model output for verification

## Model Selection Guide

### Recommended Models

**🚀 For Speed & Efficiency**: `qwen2.5-vl-3b`
- Fastest inference (2-3 seconds per page)
- 6GB VRAM requirement
- Excellent for large-scale processing

**⚖️ For Balanced Performance**: `qwen2.5-vl-7b` *(Recommended)*
- Best accuracy-speed balance
- 14GB VRAM requirement
- Professional-quality analysis

**🎯 For Maximum Accuracy**: `qwen2.5-vl-32b`
- Highest analysis quality
- 42GB VRAM requirement (A40/A100 class GPUs)
- Research-grade results

### Alternative Testing
**For Architecture Comparison**:
- `llava-1.6-7b` - Different training approach
- `instructblip-7b` - Alternative vision-language architecture

### GPU Memory Requirements
- **3B models**: 6-8GB VRAM
- **7B models**: 14-16GB VRAM
- **32B models**: 42-44GB VRAM
- **CPU fallback**: 8-32GB system RAM (slower)

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

### With Conda (Recommended)
```bash
# 1. Connect via SSH in VSCode
# 2. Create conda environment
conda env create -f environment.yml
conda activate voc-classifier

# 3. Test GPU
python test_gpu.py

# 4. Run classifier
python classify.py /path/to/images --model qwen2-vl-2b
```

### With Pip
```bash
# 1. Connect via SSH in VSCode
# 2. Install dependencies
pip install -r requirements.txt

# 3. Test GPU
python test_gpu.py

# 4. Run classifier
python classify.py /path/to/images --model qwen2-vl-2b
```

The program will automatically use CUDA if available on the remote server, or fallback gracefully to CPU.

## Environment Management

### Conda Commands
```bash
# List environments
conda env list

# Activate environment
conda activate voc-classifier

# Deactivate environment
conda deactivate

# Remove environment
conda env remove -n voc-classifier

# Update environment
conda env update -f environment.yml --prune
```

## 🎓 Research Applications

### Digital Humanities
- **Archival Cataloging**: Systematic classification of large document collections
- **Historical Analysis**: Quantitative study of document layouts and features
- **Manuscript Studies**: Detailed analysis of handwriting styles and conditions
- **Cultural Heritage**: Digital preservation with comprehensive metadata

### Professional Features
- **Multi-Dimensional Analysis**: 10 classification dimensions with explanations
- **Historical Context**: VOC-specific expertise (1602-1800 period)
- **Robust Processing**: Handles damaged, aged, and poor-quality documents
- **Batch Processing**: Efficient analysis of large archives
- **Research Output**: JSON format suitable for database integration

### Example Use Cases
```bash
# Large archive processing
python classify.py /path/to/voc_archive --model qwen2.5-vl-7b --output complete_analysis.json

# Quick survey of document types
python classify.py /path/to/samples --model qwen2.5-vl-3b

# High-accuracy research analysis
python classify.py /path/to/important_docs --model qwen2.5-vl-32b --output research_data.json
```

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)**: Development guide and technical details
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Common issues and solutions
- **[PROMPT_IMPROVEMENTS.md](PROMPT_IMPROVEMENTS.md)**: Prompt engineering details

## 🔧 Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions to common issues:

- Model loading failures
- GPU memory problems
- Classification accuracy issues
- Environment setup problems

**Quick Health Check**:
```bash
# Test GPU and environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify model loading
python classify.py --list-models

# Test single image
python classify.py path/to/single/image
```