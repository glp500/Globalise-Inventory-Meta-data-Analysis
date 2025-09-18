"""
Qwen2-VL model interface for VOC document classification.
"""

from typing import Dict, Any, Optional, List
import logging
import torch
import os
from pathlib import Path
import json

from models.prompts import VOCClassificationPrompts

# Load environment variables for external cache
def load_cache_config():
    """Load HuggingFace cache configuration from .env file."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        logging.info(f"Loaded cache configuration from {env_file}")

# Load cache configuration on import
load_cache_config()

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class QwenInterface:
    """Interface for Qwen2-VL model operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qwen interface.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_name = config.get('name', 'Qwen2-VL-7B-Instruct')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 50)
        
        self.model = None
        self.processor = None
        self.prompts = VOCClassificationPrompts()
        
        # Initialize model if transformers is available
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            logger.error("Cannot initialize model: transformers library not available")
    
    def _initialize_model(self):
        """Initialize the Qwen2-VL model and processor."""
        try:
            logger.info(f"Loading {self.model_name} on {self.device}")
            
            # Get torch_dtype from config or auto-detect
            torch_dtype = self.config.get('torch_dtype', 'auto')
            if torch_dtype == 'float16':
                dtype = torch.float16
            elif torch_dtype == 'float32':
                dtype = torch.float32
            else:
                # Auto-detect based on device
                dtype = torch.float16 if self.device == 'cuda' else torch.float32
            
            logger.info(f"Using dtype: {dtype}")
            
            # Load model and processor with better error handling
            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map=self.device if self.device != 'cpu' else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                if self.device == 'cpu':
                    self.model = self.model.to('cpu')
                
            except Exception as model_e:
                logger.error(f"Failed to load model with device {self.device}: {model_e}")
                
                # Try fallback to CPU if enabled
                if self.config.get('fallback_to_cpu', True) and self.device != 'cpu':
                    logger.info("Attempting fallback to CPU...")
                    self.device = 'cpu'
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    ).to('cpu')
                else:
                    raise model_e
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Consider checking model name or running with --log-level DEBUG")
            self.model = None
            self.processor = None
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a single image using Qwen2-VL.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing classification results
        """
        if not self.model or not self.processor:
            logger.error("Model not initialized")
            return {
                'category': 'error',
                'confidence': 0.0,
                'features': {},
                'error': 'Model not initialized'
            }
        
        try:
            # Load and preprocess image
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            
            # Get classification prompt
            prompt = self.prompts.get_classification_prompt()
            
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response
            result = self._parse_classification_response(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")
            return {
                'category': 'error',
                'confidence': 0.0,
                'features': {},
                'error': str(e)
            }
    
    def classify_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple images in a batch for better performance.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of classification result dictionaries
        """
        if not self.model or not self.processor:
            logger.error("Model not initialized")
            return [{
                'category': 'error',
                'confidence': 0.0,
                'features': {},
                'error': 'Model not initialized'
            } for _ in image_paths]
        
        # For now, process sequentially but with optimizations
        # TODO: Implement true batch processing when model supports it
        results = []
        
        try:
            from PIL import Image
            prompt = self.prompts.get_classification_prompt()
            
            for image_path in image_paths:
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    
                    # Prepare inputs
                    inputs = self.processor(
                        text=prompt,
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_tokens,
                            temperature=self.temperature,
                            do_sample=True
                        )
                    
                    # Decode response
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Parse response
                    result = self._parse_classification_response(response)
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error classifying image {image_path}: {e}")
                    results.append({
                        'category': 'error',
                        'confidence': 0.0,
                        'features': {},
                        'error': str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch classification: {e}")
            return [{
                'category': 'error',
                'confidence': 0.0,
                'features': {},
                'error': str(e)
            } for _ in image_paths]
    
    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the model's classification response.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Parsed classification result
        """
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                # Validate required fields
                if 'category' not in result:
                    result['category'] = 'unknown'
                if 'confidence' not in result:
                    result['confidence'] = 0.5
                
                return result
            
            # Fallback: simple text parsing
            lines = response.strip().split('\n')
            category = 'unknown'
            confidence = 0.5
            
            for line in lines:
                line = line.lower()
                if any(cat in line for cat in ['single_column', 'two_column', 'table', 'illustration']):
                    # Extract first matching category
                    for cat in ['single_column', 'two_column', 'table_full', 'illustration']:
                        if cat in line:
                            category = cat
                            break
                    break
            
            return {
                'category': category,
                'confidence': confidence,
                'features': {},
                'raw_response': response
            }
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'features': {},
                'error': str(e),
                'raw_response': response
            }
    
    def is_available(self) -> bool:
        """Check if the model is available and ready to use."""
        return self.model is not None and self.processor is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'available': self.is_available(),
            'transformers_available': TRANSFORMERS_AVAILABLE
        }