"""
Main Qwen2-VL classifier for VOC archive pages.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import yaml

from models.qwen_interface import QwenInterface
from core.page_detector import analyze_page_structure


logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of page classification."""
    image_path: str
    category: str
    confidence: float
    page_type: str
    features: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None


class VOCPageClassifier:
    """
    Main classifier for VOC archive pages.
    
    Categories:
    - single_column
    - two_column
    - table_full
    - table_partial
    - marginalia
    - two_page_spread
    - extended_foldout
    - illustration
    - title_page
    - blank
    - seal_signature
    - mixed_layout
    - damaged_partial
    - index_list
    """
    
    CATEGORIES = [
        'single_column',
        'two_column', 
        'table_full',
        'table_partial',
        'marginalia',
        'two_page_spread',
        'extended_foldout',
        'illustration',
        'title_page',
        'blank',
        'seal_signature',
        'mixed_layout',
        'damaged_partial',
        'index_list'
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the classifier with configuration."""
        self.config = self._load_config(config_path)
        
        # Initialize with error handling
        try:
            self.qwen_interface = QwenInterface(self.config.get('model', {}))
            if not self.qwen_interface.is_available():
                logger.warning("Qwen model not available - classifications will use fallback methods")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen interface: {e}")
            self.qwen_interface = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def classify_page(self, image_path: str) -> ClassificationResult:
        """
        Single page classification.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ClassificationResult with category and confidence
        """
        try:
            # First, handle special cases (two-page spreads, foldouts)
            special_category = self.handle_special_cases(image_path)
            if special_category:
                return ClassificationResult(
                    image_path=image_path,
                    category=special_category,
                    confidence=0.9,
                    page_type=special_category,
                    features={},
                    timestamp=self._get_timestamp()
                )
            
            # Check if Qwen model is available
            if self.qwen_interface and self.qwen_interface.is_available():
                # Use Qwen model for classification
                result = self.qwen_interface.classify_image(image_path)
                
                # Check if classification succeeded
                if result.get('category') != 'error' and result.get('confidence', 0) > 0:
                    return ClassificationResult(
                        image_path=image_path,
                        category=result.get('category', 'unknown'),
                        confidence=result.get('confidence', 0.0),
                        page_type='single_page',
                        features=result.get('features', {}),
                        timestamp=self._get_timestamp()
                    )
                else:
                    logger.warning(f"Qwen classification failed for {image_path}, using fallback")
            
            # Fallback to rule-based classification
            return self._fallback_classification(image_path)
            
        except Exception as e:
            logger.error(f"Error classifying {image_path}: {e}")
            
            # Try fallback classification even on error
            try:
                return self._fallback_classification(image_path, error=str(e))
            except Exception as fallback_e:
                logger.error(f"Fallback classification also failed for {image_path}: {fallback_e}")
                return ClassificationResult(
                    image_path=image_path,
                    category='error',
                    confidence=0.0,
                    page_type='unknown',
                    features={},
                    timestamp=self._get_timestamp(),
                    error=f"Primary: {str(e)}, Fallback: {str(fallback_e)}"
                )
    
    def process_directory(self, input_dir: str) -> List[ClassificationResult]:
        """
        Batch process entire directory.
        
        Args:
            input_dir: Directory containing images to classify
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        input_path = Path(input_dir)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        # Find all image files (exclude hidden files)
        image_files = [
            f for f in input_path.rglob('*') 
            if f.suffix.lower() in image_extensions and not f.name.startswith('.')
        ]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Get batch size from config
        batch_size = self.config.get('classification', {}).get('batch_size', 1)
        
        # Process in batches if batch processing is available
        if hasattr(self.qwen_interface, 'classify_batch') and batch_size > 1:
            return self._process_directory_batched(image_files, batch_size)
        else:
            # Fall back to single image processing
            return self._process_directory_sequential(image_files)
    
    def _process_directory_sequential(self, image_files: List[Path]) -> List[ClassificationResult]:
        """Process images one by one (fallback method)."""
        results = []
        
        for image_file in image_files:
            result = self.classify_page(str(image_file))
            results.append(result)
            
            # Log progress
            if len(results) % 10 == 0:
                logger.info(f"Processed {len(results)}/{len(image_files)} images")
        
        return results
    
    def _process_directory_batched(self, image_files: List[Path], batch_size: int) -> List[ClassificationResult]:
        """Process images in batches for better performance."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        results = []
        total_files = len(image_files)
        
        # Process files in batches
        for i in range(0, total_files, batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_paths = [str(f) for f in batch_files]
            
            start_time = time.time()
            
            try:
                # Classify batch
                batch_dicts = self.qwen_interface.classify_batch(batch_paths)
                
                # Convert dictionaries to ClassificationResult objects
                batch_results = []
                for i, result_dict in enumerate(batch_dicts):
                    image_path = batch_paths[i]
                    classification_result = ClassificationResult(
                        image_path=image_path,
                        category=result_dict.get('category', 'unknown'),
                        confidence=result_dict.get('confidence', 0.0),
                        page_type='single_page',  # Default, could be enhanced
                        features=result_dict.get('features', {}),
                        timestamp=self._get_timestamp(),
                        error=result_dict.get('error')
                    )
                    batch_results.append(classification_result)
                
                results.extend(batch_results)
                
                # Log progress with timing
                batch_time = time.time() - start_time
                processed = min(i + batch_size, total_files)
                avg_time = batch_time / len(batch_files)
                logger.info(f"Processed batch {processed}/{total_files} images "
                           f"({avg_time:.2f}s per image)")
                
            except Exception as e:
                logger.warning(f"Batch processing failed for batch {i//batch_size + 1}: {e}")
                logger.info("Falling back to sequential processing for this batch")
                
                # Fall back to sequential for this batch
                for image_file in batch_files:
                    result = self.classify_page(str(image_file))
                    results.append(result)
        
        return results
    
    def handle_special_cases(self, image_path: str) -> Optional[str]:
        """
        Pre-classification for two-page spreads and foldouts.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Category string if special case detected, None otherwise
        """
        try:
            structure_analysis = analyze_page_structure(image_path)
            
            # Check for foldout
            foldout_analysis = structure_analysis.get('foldout_analysis', {})
            if foldout_analysis.get('is_foldout', False):
                return 'extended_foldout'
            
            # Check for two-page spread
            spread_analysis = structure_analysis.get('two_page_analysis', {})
            if spread_analysis.get('is_two_page', False):
                return 'two_page_spread'
            
            return None
            
        except Exception as e:
            logger.error(f"Error in special case handling for {image_path}: {e}")
            return None
    
    def _fallback_classification(self, image_path: str, error: Optional[str] = None) -> ClassificationResult:
        """
        Fallback classification using rule-based methods when ML model fails.
        
        Args:
            image_path: Path to the image file
            error: Optional error message from primary classification
            
        Returns:
            ClassificationResult using rule-based analysis
        """
        try:
            from utils.image_utils import get_image_info
            from core.page_detector import analyze_page_structure
            
            # Get basic image information
            image_info = get_image_info(image_path)
            
            # Analyze page structure
            structure = analyze_page_structure(image_path)
            
            # Rule-based classification based on aspect ratio and structure
            aspect_ratio = image_info.get('aspect_ratio', 1.0)
            width = image_info.get('width', 0)
            height = image_info.get('height', 0)
            
            # Classification rules
            category = 'unknown'
            confidence = 0.3  # Lower confidence for rule-based
            
            # Check for two-page spreads first
            if structure.get('page_type') == 'two_page_spread':
                category = 'two_page_spread'
                confidence = 0.8
            elif structure.get('page_type') == 'extended_foldout':
                category = 'extended_foldout' 
                confidence = 0.8
            elif aspect_ratio > 1.5:
                # Likely two-page or wide format
                category = 'two_page_spread' if aspect_ratio > 1.8 else 'mixed_layout'
                confidence = 0.6
            elif aspect_ratio < 0.6:
                # Very tall, likely single column
                category = 'single_column'
                confidence = 0.6
            elif width > 4000 or height > 4000:
                # High resolution, assume good quality single page
                category = 'single_column'
                confidence = 0.5
            else:
                # Default assumption
                category = 'single_column'
                confidence = 0.3
            
            features = {
                'aspect_ratio': aspect_ratio,
                'width': width,
                'height': height,
                'classification_method': 'rule_based_fallback',
                'structure_analysis': structure
            }
            
            return ClassificationResult(
                image_path=image_path,
                category=category,
                confidence=confidence,
                page_type='single_page' if category not in ['two_page_spread', 'extended_foldout'] else category,
                features=features,
                timestamp=self._get_timestamp(),
                error=error
            )
            
        except Exception as e:
            logger.error(f"Fallback classification failed for {image_path}: {e}")
            return ClassificationResult(
                image_path=image_path,
                category='error',
                confidence=0.0,
                page_type='unknown',
                features={},
                timestamp=self._get_timestamp(),
                error=f"Fallback error: {str(e)}" + (f", Primary error: {error}" if error else "")
            )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()