"""
Image preprocessing module for VOC archive documents.
Handles image enhancement and preparation for classification.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing for historical document analysis."""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (width, height) for resizing
        """
        self.target_size = target_size or (1024, 1024)
    
    def preprocess_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Preprocess image for classification.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save preprocessed image
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Apply preprocessing steps
            processed = self._enhance_contrast(image)
            processed = self._denoise(processed)
            processed = self._resize_image(processed)
            
            # Save if output path provided
            if output_path:
                cv2.imwrite(output_path, processed)
                logger.info(f"Saved preprocessed image to {output_path}")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Could not enhance contrast: {e}")
            return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to reduce image noise."""
        try:
            # Use Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            return denoised
            
        except Exception as e:
            logger.warning(f"Could not denoise image: {e}")
            return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio."""
        try:
            h, w = image.shape[:2]
            target_w, target_h = self.target_size
            
            # Calculate scaling factor to fit within target size
            scale = min(target_w / w, target_h / h)
            
            # Only resize if image is larger than target
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Create padded image to exact target size
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return padded
            
            return image
            
        except Exception as e:
            logger.warning(f"Could not resize image: {e}")
            return image
    
    def extract_text_regions(self, image: np.ndarray) -> list:
        """Extract potential text regions from the image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological operations to find text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            dilated = cv2.dilate(gray, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h
                
                # Filter for text-like regions
                if area > 100 and 0.1 < aspect_ratio < 10:
                    text_regions.append((x, y, w, h))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error extracting text regions: {e}")
            return []