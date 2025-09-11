"""
Page Layout Detection Module for VOC Archive Classifier
Detects two-page spreads, foldouts, and other special page layouts
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging
from scipy import signal
from skimage import filters, morphology, measure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PageDetectionResult:
    """Results from page layout detection"""
    is_two_page_spread: bool
    is_foldout: bool
    aspect_ratio: float
    width: int
    height: int
    split_point: Optional[int] = None
    confidence: float = 0.0
    orientation: str = "portrait"
    estimated_panels: int = 1
    detection_features: Dict = None
    
class VOCPageDetector:
    """
    Detector for special page layouts in VOC archives
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize detector with configuration
        
        Args:
            config: Detection configuration parameters
        """
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default detection parameters"""
        return {
            'two_page_spread': {
                'aspect_ratio_threshold': 1.8,
                'center_fold_width': 50,
                'center_fold_darkness_threshold': 0.7,
                'vertical_line_min_length': 0.6,  # % of image height
                'confidence_threshold': 0.75
            },
            'foldout': {
                'aspect_ratio_threshold': 3.0,
                'panel_detection': True,
                'min_panel_width': 400
            },
            'preprocessing': {
                'resize_for_detection': True,
                'max_detection_width': 2000,
                'enhance_contrast': True,
                'denoise': True
            }
        }
    
    def analyze_page(self, image_path: str) -> PageDetectionResult:
        """
        Complete analysis of page structure
        
        Args:
            image_path: Path to image file
            
        Returns:
            PageDetectionResult with detection information
        """
        logger.info(f"Analyzing page: {image_path}")
        
        # Load and preprocess image
        image = self._load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get basic measurements
        height, width = image.shape[:2]
        aspect_ratio = width / height
        orientation = "landscape" if aspect_ratio > 1 else "portrait"
        
        # Initialize result
        result = PageDetectionResult(
            is_two_page_spread=False,
            is_foldout=False,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            orientation=orientation,
            detection_features={}
        )
        
        # Detect two-page spread
        if aspect_ratio > self.config['two_page_spread']['aspect_ratio_threshold']:
            spread_detection = self._detect_two_page_spread(image)
            result.is_two_page_spread = spread_detection['is_spread']
            result.split_point = spread_detection.get('split_point')
            result.confidence = spread_detection.get('confidence', 0.0)
            result.detection_features.update(spread_detection.get('features', {}))
        
        # Detect foldout
        if aspect_ratio > self.config['foldout']['aspect_ratio_threshold']:
            foldout_detection = self._detect_foldout(image, aspect_ratio)
            result.is_foldout = foldout_detection['is_foldout']
            result.estimated_panels = foldout_detection.get('panels', 1)
            result.detection_features.update(foldout_detection.get('features', {}))
        
        return result
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and optionally preprocess image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Optional preprocessing
            if self.config['preprocessing']['enhance_contrast']:
                image = self._enhance_contrast(image)
            
            if self.config['preprocessing']['denoise']:
                image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Resize for detection if needed
            if self.config['preprocessing']['resize_for_detection']:
                max_width = self.config['preprocessing']['max_detection_width']
                if image.shape[1] > max_width:
                    scale = max_width / image.shape[1]
                    new_height = int(image.shape[0] * scale)
                    image = cv2.resize(image, (max_width, new_height))
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _detect_two_page_spread(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect if image contains two pages side by side
        
        Methods used:
        1. Center fold detection (dark vertical line)
        2. Vertical edge detection in center region
        3. Text column analysis
        4. Symmetry analysis
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detection_results = {
            'is_spread': False,
            'confidence': 0.0,
            'split_point': None,
            'features': {}
        }
        
        # Method 1: Detect center fold (dark vertical line)
        center_fold = self._detect_center_fold(gray)
        if center_fold['found']:
            detection_results['features']['center_fold'] = center_fold
            detection_results['confidence'] += 0.4
        
        # Method 2: Detect strong vertical edges in center region
        vertical_edges = self._detect_vertical_edges(gray)
        if vertical_edges['found']:
            detection_results['features']['vertical_edges'] = vertical_edges
            detection_results['confidence'] += 0.3
            
        # Method 3: Analyze text columns
        column_analysis = self._analyze_text_columns(gray)
        if column_analysis['two_distinct_regions']:
            detection_results['features']['columns'] = column_analysis
            detection_results['confidence'] += 0.2
        
        # Method 4: Check symmetry
        symmetry = self._check_page_symmetry(gray)
        if symmetry['is_symmetric']:
            detection_results['features']['symmetry'] = symmetry
            detection_results['confidence'] += 0.1
        
        # Determine split point
        if center_fold['found']:
            detection_results['split_point'] = center_fold['position']
        elif vertical_edges['found']:
            detection_results['split_point'] = vertical_edges['strongest_edge']
        else:
            detection_results['split_point'] = width // 2
        
        # Final determination
        threshold = self.config['two_page_spread']['confidence_threshold']
        detection_results['is_spread'] = detection_results['confidence'] >= threshold
        
        logger.info(f"Two-page spread detection: {detection_results['is_spread']} "
                   f"(confidence: {detection_results['confidence']:.2f})")
        
        return detection_results
    
    def _detect_center_fold(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect dark vertical line in center (binding fold)"""
        height, width = gray.shape
        center_region_width = self.config['two_page_spread']['center_fold_width']
        
        # Extract center region
        center = width // 2
        start = max(0, center - center_region_width // 2)
        end = min(width, center + center_region_width // 2)
        center_region = gray[:, start:end]
        
        # Find darkest vertical line
        column_means = np.mean(center_region, axis=0)
        darkest_column = np.argmin(column_means)
        darkness_ratio = column_means[darkest_column] / np.mean(column_means)
        
        # Check if dark enough to be a fold
        threshold = self.config['two_page_spread']['center_fold_darkness_threshold']
        is_fold = darkness_ratio < threshold
        
        return {
            'found': is_fold,
            'position': start + darkest_column if is_fold else None,
            'darkness_ratio': darkness_ratio,
            'confidence': 1.0 - darkness_ratio if is_fold else 0.0
        }
    
    def _detect_vertical_edges(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect strong vertical edges in center region"""
        height, width = gray.shape
        
        # Apply Sobel filter for vertical edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)
        
        # Focus on center third of image
        center_start = width // 3
        center_end = 2 * width // 3
        center_edges = sobel_x[:, center_start:center_end]
        
        # Sum edge strength vertically
        vertical_projection = np.sum(center_edges, axis=0)
        
        # Find peaks (strong vertical lines)
        threshold = np.mean(vertical_projection) + 2 * np.std(vertical_projection)
        peaks = signal.find_peaks(vertical_projection, height=threshold, distance=20)[0]
        
        if len(peaks) > 0:
            strongest_peak = peaks[np.argmax(vertical_projection[peaks])]
            edge_position = center_start + strongest_peak
            
            # Check if edge is long enough
            edge_column = sobel_x[:, edge_position]
            edge_length = np.sum(edge_column > threshold) / height
            min_length = self.config['two_page_spread']['vertical_line_min_length']
            
            is_valid = edge_length > min_length
            
            return {
                'found': is_valid,
                'strongest_edge': edge_position if is_valid else None,
                'edge_strength': float(vertical_projection[strongest_peak]),
                'edge_length_ratio': edge_length,
                'num_edges': len(peaks)
            }
        
        return {'found': False}
    
    def _analyze_text_columns(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze if text appears in two distinct regions"""
        height, width = gray.shape
        
        # Binarize image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate horizontal projection
        horizontal_projection = np.sum(255 - binary, axis=0)
        
        # Smooth projection
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(horizontal_projection, sigma=20)
        
        # Find valleys (gaps between text regions)
        valleys = signal.find_peaks(-smoothed, distance=width//10)[0]
        
        # Check for significant valley near center
        center = width // 2
        center_valleys = valleys[np.abs(valleys - center) < width // 10]
        
        if len(center_valleys) > 0:
            gap_position = center_valleys[0]
            gap_depth = 1.0 - (smoothed[gap_position] / np.max(smoothed))
            
            return {
                'two_distinct_regions': gap_depth > 0.3,
                'gap_position': int(gap_position),
                'gap_depth': float(gap_depth),
                'num_text_regions': len(valleys) + 1
            }
        
        return {'two_distinct_regions': False}
    
    def _check_page_symmetry(self, gray: np.ndarray) -> Dict[str, Any]:
        """Check if left and right halves are similar (two pages)"""
        height, width = gray.shape
        center = width // 2
        
        # Split image
        left_half = gray[:, :center]
        right_half = gray[:, center:]
        
        # Resize to same size if needed
        if left_half.shape != right_half.shape:
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
        
        # Calculate histograms
        hist_left = cv2.calcHist([left_half], [0], None, [256], [0, 256])
        hist_right = cv2.calcHist([right_half], [0], None, [256], [0, 256])
        
        # Compare histograms
        correlation = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)
        
        return {
            'is_symmetric': correlation > 0.7,
            'correlation': float(correlation)
        }
    
    def _detect_foldout(self, image: np.ndarray, aspect_ratio: float) -> Dict[str, Any]:
        """
        Detect foldout/extended pages
        
        Characteristics:
        - Extreme aspect ratio (>3:1)
        - Multiple panels/sections
        - Possible fold marks
        """
        height, width = image.shape[:2]
        
        detection_results = {
            'is_foldout': False,
            'panels': 1,
            'features': {}
        }
        
        # Check extreme aspect ratio
        if aspect_ratio > self.config['foldout']['aspect_ratio_threshold']:
            detection_results['is_foldout'] = True
            
            # Estimate number of panels
            if self.config['foldout']['panel_detection']:
                panels = self._detect_panels(image)
                detection_results['panels'] = panels['num_panels']
                detection_results['features']['panel_boundaries'] = panels.get('boundaries', [])
            else:
                # Simple estimation based on aspect ratio
                detection_results['panels'] = int(aspect_ratio / 0.7)  # Assume ~0.7 ratio per panel
            
            # Detect fold marks
            fold_marks = self._detect_fold_marks(image)
            detection_results['features']['fold_marks'] = fold_marks
        
        logger.info(f"Foldout detection: {detection_results['is_foldout']} "
                   f"(panels: {detection_results['panels']})")
        
        return detection_results
    
    def _detect_panels(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect individual panels in foldout"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        height, width = gray.shape
        
        # Detect vertical lines that might indicate panel boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough transform for vertical lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/2, threshold=int(height*0.3),
                                minLineLength=int(height*0.5), maxLineGap=20)
        
        panel_boundaries = []
        if lines is not None:
            # Filter for vertical lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 10:  # Nearly vertical
                    panel_boundaries.append(x1)
            
            # Cluster nearby lines
            panel_boundaries = sorted(set(panel_boundaries))
            min_panel_width = self.config['foldout']['min_panel_width']
            
            # Filter boundaries that are too close
            filtered_boundaries = []
            for boundary in panel_boundaries:
                if not filtered_boundaries or boundary - filtered_boundaries[-1] > min_panel_width:
                    filtered_boundaries.append(boundary)
            
            panel_boundaries = filtered_boundaries
        
        num_panels = len(panel_boundaries) + 1 if panel_boundaries else max(1, int(width / 800))
        
        return {
            'num_panels': num_panels,
            'boundaries': panel_boundaries
        }
    
    def _detect_fold_marks(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect physical fold marks in the paper"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Look for vertical lines that might be folds
        # These often appear as subtle shadows or creases
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 10, 30)  # Low thresholds for subtle marks
        
        # Count vertical edge segments
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        
        num_fold_marks = np.sum(vertical_lines > 0) // (gray.shape[0] * 10)
        
        return {
            'detected': num_fold_marks > 2,
            'count': int(num_fold_marks)
        }
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for old documents"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def split_two_page_spread(self, image_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Split a two-page spread into individual pages
        
        Args:
            image_path: Path to two-page spread image
            output_dir: Directory to save split pages
            
        Returns:
            Tuple of (left_page_path, right_page_path)
        """
        image = cv2.imread(image_path)
        result = self.analyze_page(image_path)
        
        if not result.is_two_page_spread:
            logger.warning(f"Image {image_path} is not detected as two-page spread")
            return None, None
        
        # Use detected split point or center
        split_point = result.split_point or image.shape[1] // 2
        
        # Split image
        left_page = image[:, :split_point]
        right_page = image[:, split_point:]
        
        # Save split pages
        base_name = Path(image_path).stem
        left_path = Path(output_dir) / f"{base_name}_left.jpg"
        right_path = Path(output_dir) / f"{base_name}_right.jpg"
        
        cv2.imwrite(str(left_path), left_page)
        cv2.imwrite(str(right_path), right_page)
        
        logger.info(f"Split {image_path} into {left_path} and {right_path}")
        
        return str(left_path), str(right_path)


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = VOCPageDetector()
    
    # Test on sample image
    test_image = "data/input/sample_page.jpg"
    
    try:
        result = detector.analyze_page(test_image)
        
        print(f"Analysis Results for {test_image}:")
        print(f"  Aspect Ratio: {result.aspect_ratio:.2f}")
        print(f"  Orientation: {result.orientation}")
        print(f"  Two-Page Spread: {result.is_two_page_spread}")
        print(f"  Foldout: {result.is_foldout}")
        
        if result.is_two_page_spread:
            print(f"  Split Point: {result.split_point}")
            print(f"  Confidence: {result.confidence:.2f}")
        
        if result.is_foldout:
            print(f"  Estimated Panels: {result.estimated_panels}")
        
        print(f"  Detection Features: {result.detection_features}")
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        