"""
Main entry point for VOC page classifier.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.classifier import VOCPageClassifier
from utils.file_handler import FileHandler
from utils.report_generator import ReportGenerator


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path or not Path(config_path).exists():
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading config {config_path}: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'model': {
            'name': 'Qwen2-VL-7B-Instruct',
            'device': 'cuda',
            'temperature': 0.1,
            'max_tokens': 50
        },
        'detection': {
            'two_page_threshold': 1.8,
            'foldout_threshold': 3.0,
            'center_fold_width': 50,
            'edge_detection_sensitivity': 0.7
        },
        'classification': {
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.5,
                'low': 0.3
            },
            'batch_size': 10,
            'save_interval': 100
        },
        'output': {
            'create_subdirs': True,
            'generate_report': True,
            'copy_files': False,
            'report_format': 'csv'
        }
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='VOC Archive Page Classifier')
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', required=True, 
                       help='Input directory containing images')
    parser.add_argument('--output', '-o', required=True, 
                       help='Output directory for classified images')
    parser.add_argument('--config', '-c', 
                       help='Configuration file path')
    
    # Processing options
    parser.add_argument('--detect-spreads', action='store_true',
                       help='Enable two-page spread detection')
    parser.add_argument('--split-pages', action='store_true',
                       help='Split two-page spreads into separate files')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Minimum confidence threshold for classification')
    
    # Resume/checkpoint options
    parser.add_argument('--resume', 
                       help='Resume from checkpoint file')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Save checkpoint every N files')
    
    # Report options
    parser.add_argument('--report-only', 
                       help='Generate report from existing results file')
    parser.add_argument('--report-format', choices=['csv', 'json', 'html'], 
                       default='csv', help='Report output format')
    
    # Logging options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', 
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Handle report-only mode
        if args.report_only:
            generate_report_only(args.report_only, args.output, args.report_format)
            return
        
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.confidence_threshold != 0.5:
            config.setdefault('classification', {})['confidence_threshold'] = args.confidence_threshold
        
        # Initialize components
        classifier = VOCPageClassifier(args.config)
        file_handler = FileHandler(args.output)
        report_generator = ReportGenerator(Path(args.output) / "reports")
        
        # Check if resuming from checkpoint
        start_results = []
        if args.resume:
            checkpoint_data = file_handler.load_checkpoint(args.resume)
            start_results = checkpoint_data.get('results', [])
            logger.info(f"Resuming from checkpoint with {len(start_results)} existing results")
        
        # Create category directories
        if config.get('output', {}).get('create_subdirs', True):
            file_handler.create_category_directories(classifier.CATEGORIES)
        
        logger.info(f"Starting classification of images in: {args.input}")
        logger.info(f"Output directory: {args.output}")
        
        # Process images
        results = classifier.process_directory(args.input)
        
        # Combine with resumed results
        all_results = start_results + results
        
        # Filter by confidence threshold
        filtered_results = [
            r for r in all_results 
            if r.confidence >= args.confidence_threshold
        ]
        
        logger.info(f"Classification complete. Processed {len(all_results)} files")
        logger.info(f"Files meeting confidence threshold: {len(filtered_results)}")
        
        # Organize files if requested
        if config.get('output', {}).get('create_subdirs', True):
            copy_files = config.get('output', {}).get('copy_files', False)
            file_handler.organize_classified_files(filtered_results, copy_files)
        
        # Generate reports
        if config.get('output', {}).get('generate_report', True):
            # Save basic report
            report_format = config.get('output', {}).get('report_format', args.report_format)
            file_handler.save_classification_report(all_results, report_format)
            
            # Generate detailed report
            report_generator.generate_detailed_report(all_results)
            
            # Create visualizations
            report_generator.create_visualization_report(all_results)
            
            # Export for analysis
            report_generator.export_for_analysis(all_results, 'excel')
        
        # Clean up temporary files
        file_handler.cleanup_temp_files()
        
        logger.info("Processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


def generate_report_only(results_file: str, output_dir: str, format: str):
    """Generate report from existing results file."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load existing results
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if isinstance(results, dict) and 'results' in results:
            results = results['results']
        
        # Generate reports
        report_generator = ReportGenerator(Path(output_dir) / "reports")
        
        if format == 'html':
            report_path = report_generator.generate_detailed_report(results)
        else:
            file_handler = FileHandler(output_dir)
            report_path = file_handler.save_classification_report(results, format)
        
        # Also create visualization
        report_generator.create_visualization_report(results)
        
        logger.info(f"Report generated: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()