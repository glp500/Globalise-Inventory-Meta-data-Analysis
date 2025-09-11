# VOC Archive Page Classifier

A computer vision system for classifying scanned pages from VOC (Dutch East India Company) archives using Qwen2-VL models for visual understanding of 17th-19th century documents.

## Features

- **Automated Page Classification**: Classify documents into 14 categories including single/two-column text, tables, illustrations, title pages, and more
- **Two-Page Spread Detection**: Automatically detect and handle documents spanning two pages
- **Foldout Page Detection**: Identify extended/folded pages with multiple panels
- **Batch Processing**: Process entire directories of images efficiently
- **Detailed Reporting**: Generate comprehensive reports with visualizations and statistics
- **Resume Functionality**: Resume interrupted processing from checkpoints
- **Flexible Output**: Organize classified files into category directories with customizable options

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the Qwen2-VL model available (will be downloaded automatically on first use)

## Quick Start

1. **Basic Classification**:
```bash
python src/main.py --input data/input --output data/output
```

2. **With Custom Configuration**:
```bash
python src/main.py --input data/input --output data/output --config src/config/settings.yaml
```

3. **With Two-Page Detection**:
```bash
python src/main.py --input data/input --output data/output --detect-spreads --split-pages
```

## Usage

### Command Line Options

```bash
python src/main.py [OPTIONS]

Required Arguments:
  --input, -i          Input directory containing images
  --output, -o         Output directory for classified images

Optional Arguments:
  --config, -c         Configuration file path (default: src/config/settings.yaml)
  --detect-spreads     Enable two-page spread detection
  --split-pages        Split two-page spreads into separate files
  --confidence-threshold FLOAT  Minimum confidence for classification (default: 0.5)
  --resume FILE        Resume from checkpoint file
  --checkpoint-interval INT  Save checkpoint every N files (default: 100)
  --report-only FILE   Generate report from existing results file
  --report-format {csv,json,html}  Report output format (default: csv)
  --log-level {DEBUG,INFO,WARNING,ERROR}  Logging level (default: INFO)
  --log-file FILE      Log file path
```

### Configuration

Edit `src/config/settings.yaml` to customize:

- **Model Settings**: Device (CPU/GPU), temperature, token limits
- **Detection Thresholds**: Aspect ratios for two-page and foldout detection
- **Classification**: Confidence thresholds, batch size, save intervals
- **Output Options**: Directory organization, report generation, file operations

### Classification Categories

The system classifies documents into these categories:

- `single_column` - Text arranged in a single column
- `two_column` - Text arranged in two columns  
- `table_full` - Full page table with rows and columns
- `table_partial` - Partial table mixed with text
- `marginalia` - Page with extensive margin notes
- `two_page_spread` - Two pages displayed side by side
- `extended_foldout` - Extended/folded page with multiple panels
- `illustration` - Page primarily containing drawings, maps, or diagrams
- `title_page` - Title or cover page
- `blank` - Mostly empty page
- `seal_signature` - Page with official seals or signatures
- `mixed_layout` - Mixed content types with no clear primary layout
- `damaged_partial` - Damaged page with missing or unclear content
- `index_list` - Index, list, or catalog page

## Output

The system generates:

1. **Organized Directories**: Files sorted into category subdirectories
2. **Classification Reports**: Detailed CSV/JSON reports with results and statistics
3. **Visual Reports**: HTML reports with charts and analysis
4. **Processing Logs**: Detailed logs of all operations
5. **Checkpoints**: Resume points for large batch processing

## Examples

### Process a Test Directory
```bash
python src/main.py --input tests/test_samples --output results/test_run --log-level DEBUG
```

### Resume Interrupted Processing
```bash
python src/main.py --resume logs/checkpoint_20240115.pkl
```

### Generate Report Only
```bash
python src/main.py --report-only results/classification_report.json --output results --report-format html
```

### High-Confidence Processing
```bash
python src/main.py --input data/input --output data/output --confidence-threshold 0.8 --detect-spreads
```

## Performance

Target performance metrics:
- Processing speed: 2-5 seconds per page (with GPU)
- Detection accuracy: >90% for two-page spreads  
- Classification consistency: >85% confidence on clear pages
- Memory usage: <8GB for batch of 100 images

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in config or use CPU
2. **Model Loading Issues**: Ensure sufficient disk space and internet connection
3. **Poor Classification**: Adjust confidence thresholds or check image quality
4. **File Permission Errors**: Ensure write permissions for output directory

### Debug Mode
```bash
python src/main.py --input data/input --output data/output --log-level DEBUG --log-file debug.log
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ disk space for models and processing

## Data Privacy

- All processing is done locally
- No data is sent to external services
- Original files are preserved (unless move option is selected)
- VOC archive images remain secure and private

## License

This project is designed for historical research and digital humanities applications. Ensure compliance with your institution's policies for handling historical documents.

## Support

For issues and questions:
1. Check the debug logs with `--log-level DEBUG`
2. Review the troubleshooting section
3. Consult the detailed documentation in `CLAUDE.md`