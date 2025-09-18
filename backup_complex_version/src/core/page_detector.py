"""
Page layout detection module for VOC archive documents.
Detects two-page spreads and foldout pages.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def detect_two_page_spread(image_path: str) -> Dict[str, Any]:
    """
    Detect if image contains two pages side by side.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing:
        - is_two_page: bool
        - confidence: float
        - split_point: int or None
        - features: dict
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Basic aspect ratio detection
        is_two_page = aspect_ratio > 1.8
        confidence = min(aspect_ratio / 1.8, 1.0) if is_two_page else 0.0
        split_point = width // 2 if is_two_page else None
        
        features = {
            'aspect_ratio': aspect_ratio,
            'width': width,
            'height': height
        }
        
        return {
            'is_two_page': is_two_page,
            'confidence': confidence,
            'split_point': split_point,
            'features': features
        }
        
    except Exception as e:
        logger.error(f"Error detecting two-page spread in {image_path}: {e}")
        return {
            'is_two_page': False,
            'confidence': 0.0,
            'split_point': None,
            'features': {}
        }


def detect_foldout(image_path: str) -> Dict[str, Any]:
    """
    Detect extended/foldout pages.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing:
        - is_foldout: bool
        - aspect_ratio: float
        - orientation: str
        - estimated_panels: int
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Detect foldouts by extreme aspect ratios
        is_foldout = aspect_ratio > 3.0 or aspect_ratio < 0.33
        
        if aspect_ratio > 3.0:
            orientation = 'horizontal'
            estimated_panels = min(int(aspect_ratio), 10)
        elif aspect_ratio < 0.33:
            orientation = 'vertical'
            estimated_panels = min(int(1 / aspect_ratio), 10)
        else:
            orientation = 'normal'
            estimated_panels = 1
        
        return {
            'is_foldout': is_foldout,
            'aspect_ratio': aspect_ratio,
            'orientation': orientation,
            'estimated_panels': estimated_panels
        }
        
    except Exception as e:
        logger.error(f"Error detecting foldout in {image_path}: {e}")
        return {
            'is_foldout': False,
            'aspect_ratio': 0.0,
            'orientation': 'unknown',
            'estimated_panels': 0
        }


def analyze_page_structure(image_path: str) -> Dict[str, Any]:
    """
    Complete structural analysis of page.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing complete structural analysis
    """
    try:
        spread_result = detect_two_page_spread(image_path)
        foldout_result = detect_foldout(image_path)
        
        return {
            'two_page_analysis': spread_result,
            'foldout_analysis': foldout_result,
            'page_type': _determine_page_type(spread_result, foldout_result)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing page structure in {image_path}: {e}")
        return {
            'two_page_analysis': {},
            'foldout_analysis': {},
            'page_type': 'unknown'
        }


def _determine_page_type(spread_result: Dict[str, Any], foldout_result: Dict[str, Any]) -> str:
    """Determine the primary page type based on analysis results."""
    if foldout_result.get('is_foldout', False):
        return 'extended_foldout'
    elif spread_result.get('is_two_page', False):
        return 'two_page_spread'
    else:
        return 'single_page'