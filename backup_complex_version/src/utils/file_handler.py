"""
File and directory management utilities for VOC document processing.
"""

import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import json
import csv
from datetime import datetime


logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file operations for the VOC classifier."""
    
    def __init__(self, base_output_dir: str):
        """
        Initialize file handler.
        
        Args:
            base_output_dir: Base directory for output files
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different categories
        self.category_dirs = {}
    
    def create_category_directories(self, categories: List[str]):
        """
        Create subdirectories for each classification category.
        
        Args:
            categories: List of category names
        """
        for category in categories:
            category_dir = self.base_output_dir / category
            category_dir.mkdir(exist_ok=True)
            self.category_dirs[category] = category_dir
            logger.info(f"Created directory: {category_dir}")
    
    def organize_classified_files(self, results: List, copy_files: bool = False):
        """
        Organize files into category directories based on classification results.
        
        Args:
            results: List of classification results
            copy_files: If True, copy files; if False, move files
        """
        operation = "Copying" if copy_files else "Moving"
        logger.info(f"{operation} {len(results)} files to category directories")
        
        for result in results:
            try:
                # Handle both dict and ClassificationResult objects
                if hasattr(result, 'image_path'):
                    source_path = Path(result.image_path)
                    category = result.category
                else:
                    source_path = Path(result['image_path'])
                    category = result['category']
                
                if category not in self.category_dirs:
                    logger.warning(f"No directory for category: {category}")
                    continue
                
                destination_dir = self.category_dirs[category]
                destination_path = destination_dir / source_path.name
                
                # Handle filename conflicts
                counter = 1
                while destination_path.exists():
                    stem = source_path.stem
                    suffix = source_path.suffix
                    destination_path = destination_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # Copy or move file
                if copy_files:
                    shutil.copy2(source_path, destination_path)
                else:
                    shutil.move(str(source_path), str(destination_path))
                
                logger.debug(f"{operation[:-3]}d {source_path.name} to {category}")
                
            except Exception as e:
                image_path = result.image_path if hasattr(result, 'image_path') else result.get('image_path', 'unknown')
                logger.error(f"Error organizing file {image_path}: {e}")
    
    def save_classification_report(self, results: List, format: str = 'csv'):
        """
        Save classification results to a report file.
        
        Args:
            results: List of classification results
            format: Output format ('csv' or 'json')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'csv':
            report_path = self.base_output_dir / f"classification_report_{timestamp}.csv"
            self._save_csv_report(results, report_path)
        elif format.lower() == 'json':
            report_path = self.base_output_dir / f"classification_report_{timestamp}.json"
            self._save_json_report(results, report_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved classification report to: {report_path}")
        return report_path
    
    def _save_csv_report(self, results: List[Dict[str, Any]], report_path: Path):
        """Save results as CSV file."""
        if not results:
            return
        
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'filename', 'category', 'confidence', 'page_type', 
                'timestamp', 'error', 'width', 'height', 'aspect_ratio'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Handle both dict and ClassificationResult objects
                if hasattr(result, 'image_path'):
                    row = {
                        'filename': Path(result.image_path).name,
                        'category': getattr(result, 'category', 'unknown'),
                        'confidence': getattr(result, 'confidence', 0.0),
                        'page_type': getattr(result, 'page_type', 'unknown'),
                        'timestamp': getattr(result, 'timestamp', ''),
                        'error': getattr(result, 'error', ''),
                        'width': getattr(result, 'features', {}).get('width', ''),
                        'height': getattr(result, 'features', {}).get('height', ''),
                        'aspect_ratio': getattr(result, 'features', {}).get('aspect_ratio', '')
                    }
                else:
                    row = {
                        'filename': Path(result['image_path']).name,
                        'category': result.get('category', 'unknown'),
                        'confidence': result.get('confidence', 0.0),
                        'page_type': result.get('page_type', 'unknown'),
                        'timestamp': result.get('timestamp', ''),
                        'error': result.get('error', ''),
                        'width': result.get('features', {}).get('width', ''),
                        'height': result.get('features', {}).get('height', ''),
                        'aspect_ratio': result.get('features', {}).get('aspect_ratio', '')
                    }
                writer.writerow(row)
    
    def _save_json_report(self, results: List, report_path: Path):
        """Save results as JSON file."""
        # Convert ClassificationResult objects to dictionaries
        results_dicts = []
        for result in results:
            if hasattr(result, 'image_path'):
                result_dict = {
                    'image_path': result.image_path,
                    'category': result.category,
                    'confidence': result.confidence,
                    'page_type': result.page_type,
                    'features': result.features,
                    'timestamp': result.timestamp,
                    'error': result.error
                }
            else:
                result_dict = result
            results_dicts.append(result_dict)
        
        report_data = {
            'metadata': {
                'total_files': len(results),
                'timestamp': datetime.now().isoformat(),
                'categories': self._get_category_stats(results_dicts)
            },
            'results': results_dicts
        }
        
        with open(report_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(report_data, jsonfile, indent=2, ensure_ascii=False)
    
    def _get_category_stats(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get statistics for each category."""
        stats = {}
        for result in results:
            category = result.get('category', 'unknown')
            stats[category] = stats.get(category, 0) + 1
        return stats
    
    def create_processing_checkpoint(self, results: List[Dict[str, Any]], batch_index: int) -> Path:
        """
        Create a checkpoint file for resume functionality.
        
        Args:
            results: Classification results so far
            batch_index: Current batch index
            
        Returns:
            Path to checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = Path("logs") / f"checkpoint_{timestamp}.json"
        
        # Ensure logs directory exists
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'batch_index': batch_index,
            'processed_files': len(results),
            'results': results
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Created checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint data for resuming processing.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            return {}
    
    def cleanup_temp_files(self):
        """Clean up temporary files and directories."""
        temp_dir = Path("data/temp")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                temp_dir.mkdir()
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.error(f"Error cleaning temp files: {e}")
    
    def get_supported_image_files(self, directory: str) -> List[Path]:
        """
        Get list of supported image files in directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of image file paths
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
        directory_path = Path(directory)
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory_path.rglob(f"*{ext}"))
            image_files.extend(directory_path.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        unique_files = list(set(image_files))
        unique_files.sort()
        
        logger.info(f"Found {len(unique_files)} image files in {directory}")
        return unique_files
    
    def validate_output_structure(self) -> bool:
        """
        Validate that output directory structure is correct.
        
        Returns:
            True if structure is valid
        """
        try:
            if not self.base_output_dir.exists():
                return False
            
            # Check if category directories exist
            for category_dir in self.category_dirs.values():
                if not category_dir.exists():
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating output structure: {e}")
            return False