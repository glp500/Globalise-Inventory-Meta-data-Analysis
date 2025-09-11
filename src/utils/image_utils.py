"""
Image processing utilities for VOC document analysis.
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image, ImageEnhance


logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            # Try with PIL as fallback
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get basic information about an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    try:
        image = load_image(image_path)
        if image is None:
            return {}
        
        height, width = image.shape[:2]
        aspect_ratio = width / height
        file_size = Path(image_path).stat().st_size
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'file_size_bytes': file_size,
            'channels': image.shape[2] if len(image.shape) > 2 else 1
        }
        
    except Exception as e:
        logger.error(f"Error getting image info for {image_path}: {e}")
        return {}


def calculate_aspect_ratio(image: np.ndarray) -> float:
    """Calculate aspect ratio (width/height) of an image."""
    height, width = image.shape[:2]
    return width / height


def detect_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges in image using Canny edge detection.
    
    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
        
    Returns:
        Edge map as binary image
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
        
    except Exception as e:
        logger.error(f"Error detecting edges: {e}")
        return np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image)


def find_vertical_lines(image: np.ndarray, min_line_length: int = 100) -> List[Tuple[int, int, int, int]]:
    """
    Find vertical lines in image (useful for detecting page boundaries).
    
    Args:
        image: Input image
        min_line_length: Minimum length for line detection
        
    Returns:
        List of line coordinates (x1, y1, x2, y2)
    """
    try:
        # Detect edges
        edges = detect_edges(image)
        
        # Use HoughLinesP to detect lines
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=min_line_length,
            maxLineGap=10
        )
        
        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is approximately vertical
                if abs(x2 - x1) < 10:  # Allow small deviation
                    vertical_lines.append((x1, y1, x2, y2))
        
        return vertical_lines
        
    except Exception as e:
        logger.error(f"Error finding vertical lines: {e}")
        return []


def split_image_vertically(image: np.ndarray, split_point: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image vertically at specified point.
    
    Args:
        image: Input image
        split_point: X-coordinate to split at
        
    Returns:
        Tuple of (left_image, right_image)
    """
    height, width = image.shape[:2]
    
    # Ensure split point is within image bounds
    split_point = max(0, min(split_point, width))
    
    left_image = image[:, :split_point]
    right_image = image[:, split_point:]
    
    return left_image, right_image


def enhance_document_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance historical document image for better readability.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    try:
        # Convert to PIL for easier manipulation
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Convert back to OpenCV format
        if len(image.shape) == 3:
            result = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        else:
            result = np.array(enhanced)
        
        return result
        
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return image


def crop_to_content(image: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Crop image to remove excessive white space around content.
    
    Args:
        image: Input image
        padding: Padding to add around detected content
        
    Returns:
        Cropped image
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Threshold to find content
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find bounding box of all content
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Add padding and ensure within image bounds
        h, w = image.shape[:2]
        x_min = max(0, int(x_min) - padding)
        y_min = max(0, int(y_min) - padding)
        x_max = min(w, int(x_max) + padding)
        y_max = min(h, int(y_max) + padding)
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        
        return cropped
        
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return image


def validate_image_file(file_path: str) -> bool:
    """
    Validate if file is a supported image format.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid image file
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return False
        
        # Check extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
        if path.suffix.lower() not in valid_extensions:
            return False
        
        # Try to load image
        image = load_image(file_path)
        return image is not None
        
    except Exception:
        return False