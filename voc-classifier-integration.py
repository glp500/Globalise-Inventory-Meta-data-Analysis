"""
VOC Archive Classifier - Main Integration Module
Combines page detection with Qwen2-VL classification
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
import base64
from io import BytesIO

# Import the page detector module
from page_detector import VOCPageDetector, PageDetectionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/voc_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Complete classification result for a page"""
    filename: str
    original_path: str
    category: str
    confidence: str
    confidence_score: float
    is_two_page_spread: bool
    is_foldout: bool
    aspect_ratio: float
    split_files: Optional[List[str]] = None
    processing_time: float = 0.0
    detection_features: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: str = ""

class QwenVLClassifier:
    """
    Wrapper for Qwen2-VL model for VOC document classification
    """
    
    VOC_CATEGORIES = {
        "single_column": "Standard text in a single column layout",
        "two_column": "Text arranged in two columns",
        "table_full": "Full page table/ledger with rows and columns",
        "table_partial": "Page with mixed content including tables and text",
        "marginalia": "Pages with extensive margin annotations",
        "two_page_spread": "Single image containing two facing pages",
        "extended_foldout": "Oversized pages that fold out",
        "illustration": "Pages containing drawings, maps, charts, or diagrams",
        "title_page": "Cover or title pages with centered text",
        "blank": "Empty or near-empty pages",
        "seal_signature": "Pages dominated by official seals or signatures",
        "mixed_layout": "Complex pages combining multiple layout types",
        "damaged_partial": "Significantly damaged or partially visible pages",
        "index_list": "Sequential lists such as inventories or catalogues"
    }
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = None):
        """
        Initialize Qwen2-VL model
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        
        logger.info(f"Initializing Qwen2-VL model: {model_name} on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load Qwen2-VL model and processor"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            logger.info("Model loaded successfully")
            
        except ImportError:
            logger.warning("Transformers not installed, using Ollama fallback")
            self._setup_ollama()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _setup_ollama(self):
        """Fallback to Ollama for model inference"""
        try:
            import ollama
            self.ollama_client = ollama.Client()
            self.model_name = "qwen2-vl:7b"
            logger.info("Using Ollama for model inference")
        except:
            raise RuntimeError("Neither Transformers nor Ollama available")
    
    def classify_page(self, image_path: str, page_info: Optional[PageDetectionResult] = None) -> Dict[str, Any]:
        """
        Classify a single page using Qwen2-VL
        
        Args:
            image_path: Path to image file
            page_info: Pre-computed page detection results
            
        Returns:
            Classification results dictionary
        """
        # Build classification prompt
        prompt = self._build_classification_prompt(page_info)
        
        try:
            if hasattr(self, 'ollama_client'):
                result = self._classify_with_ollama(image_path, prompt)
            else:
                result = self._classify_with_transformers(image_path, prompt)
            
            return self._parse_classification_result(result)
            
        except Exception as e:
            logger.error(f"Classification error for {image_path}: {e}")
            return {
                'category': 'unknown',
                'confidence': 'low',
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    def _build_classification_prompt(self, page_info: Optional[PageDetectionResult] = None) -> str:
        """Build classification prompt with context"""
        
        # Base prompt
        prompt = """You are analyzing a page scan from the VOC (Dutch East India Company) archives, dating from the 17th-19th century.

Classify this page into exactly ONE of the following categories based on its visual layout:

1. single_column - Text in one column, like letters or reports
2. two_column - Text arranged in two distinct columns  
3. table_full - Entire page is a table/ledger with grid structure
4. table_partial - Mix of tables and regular text on same page
5. marginalia - Main text with extensive handwritten notes in margins
6. two_page_spread - Image shows both left and right pages together
7. extended_foldout - Oversized page (maps, large charts, ship plans)
8. illustration - Drawings, maps, or diagrams as primary content
9. title_page - Cover/title page with centered text and possible decoration
10. blank - Empty or nearly empty page
11. seal_signature - Official seals, signatures, or stamps dominate
12. mixed_layout - Complex combination not fitting other categories
13. damaged_partial - Too damaged or incomplete to determine layout
14. index_list - Simple vertical lists (inventories, ship manifests)

"""
        
        # Add context from page detection if available
        if page_info:
            if page_info.is_two_page_spread:
                prompt += "\nNote: This image has been detected as potentially containing two pages side by side.\n"
            if page_info.is_foldout:
                prompt += f"\nNote: This appears to be a foldout page with approximately {page_info.estimated_panels} panels.\n"
            if page_info.aspect_ratio > 2.5:
                prompt += "\nNote: This image has an unusual aspect ratio suggesting extended content.\n"
        
        prompt += """
Consider these VOC document characteristics:
- Handwritten Dutch text in 17th-18th century script
- Brown ink on aged paper
- Maritime and trade terminology
- Multiple languages may appear (Dutch, Malay, Portuguese)

Respond with ONLY:
Category: [category_name]
Confidence: [High/Medium/Low]
Features: [Brief description in 10 words or less]"""
        
        return prompt
    
    def _classify_with_transformers(self, image_path: str, prompt: str) -> str:
        """Classify using Transformers library"""
        from qwen_vl_utils import process_vision_info
        
        # Load and prepare image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare messages for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process with model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True
            )
        
        output = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output
    
    def _classify_with_ollama(self, image_path: str, prompt: str) -> str:
        """Classify using Ollama"""
        # Convert image to base64
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode()
        
        response = self.ollama_client.generate(
            model=self.model_name,
            prompt=prompt,
            images=[image_data],
            options={
                "temperature": 0.1,
                "max_tokens": 50
            }
        )
        
        return response['response']
    
    def _parse_classification_result(self, result: str) -> Dict[str, Any]:
        """Parse model output into structured result"""
        lines = result.strip().split('\n')
        
        parsed = {
            'category': 'unknown',
            'confidence': 'low',
            'confidence_score': 0.0,
            'features': ''
        }
        
        for line in lines:
            if line.startswith('Category:'):
                category = line.replace('Category:', '').strip().lower().replace(' ', '_')
                if category in self.VOC_CATEGORIES:
                    parsed['category'] = category
            elif line.startswith('Confidence:'):
                confidence = line.replace('Confidence:', '').strip().lower()
                parsed['confidence'] = confidence
                parsed['confidence_score'] = {
                    'high': 0.9,
                    'medium': 0.6,
                    'low': 0.3
                }.get(confidence, 0.3)
            elif line.startswith('Features:'):
                parsed['features'] = line.replace('Features:', '').strip()
        
        return parsed

class VOCArchiveProcessor:
    """
    Main processor for VOC archive classification pipeline
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize processor with configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.detector = VOCPageDetector(self.config.get('detection', {}))
        self.classifier = QwenVLClassifier(
            model_name=self.config.get('model', {}).get('name', 'Qwen/Qwen2-VL-7B-Instruct'),
            device=self.config.get('model', {}).get('device')
        )
        self.results = []
        self.checkpoint_file = Path('logs/checkpoint.json')
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'model': {
                'name': 'Qwen/Qwen2-VL-7B-Instruct',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'detection': {
                'two_page_spread': {
                    'aspect_ratio_threshold': 1.8,
                    'auto_split': True
                },
                'foldout': {
                    'aspect_ratio_threshold': 3.0
                }
            },
            'processing': {
                'batch_size': 10,
                'save_interval': 100,
                'create_subdirs': True
            },
            'output': {
                'format': 'csv',
                'include_detection_features': True
            }
        }
    
    def process_directory(self, 
                         input_dir: str, 
                         output_dir: str,
                         resume: bool = False) -> pd.DataFrame:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing VOC page scans
            output_dir: Directory for classified outputs
            resume: Resume from checkpoint if available
            
        Returns:
            DataFrame with classification results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of images to process
        image_files = list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.jpeg')) + \
                     list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.tiff'))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Load checkpoint if resuming
        processed_files = set()
        if resume and self.checkpoint_file.exists():
            processed_files = self._load_checkpoint()
            logger.info(f"Resuming from checkpoint, {len(processed_files)} already processed")
        
        # Process images
        for idx, image_file in enumerate(tqdm(image_files, desc="Processing pages")):
            if str(image_file) in processed_files:
                continue
            
            try:
                result = self._process_single_image(image_file, output_path)
                self.results.append(result)
                
                # Save checkpoint periodically
                if (idx + 1) % self.config['processing']['save_interval'] == 0:
                    self._save_checkpoint()
                    self._save_intermediate_results(output_path)
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                self.results.append(ClassificationResult(
                    filename=image_file.name,
                    original_path=str(image_file),
                    category='error',
                    confidence='none',
                    confidence_score=0.0,
                    is_two_page_spread=False,
                    is_foldout=False,
                    aspect_ratio=0.0,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                ))
        
        # Save final results
        df_results = self._save_final_results(output_path)
        
        # Organize files into category subdirectories if configured
        if self.config['processing']['create_subdirs']:
            self._organize_by_category(df_results, output_path)
        
        return df_results
    
    def _process_single_image(self, image_path: Path, output_dir: Path) -> ClassificationResult:
        """Process a single image through detection and classification"""
        start_time = datetime.now()
        
        # Step 1: Analyze page structure
        detection_result = self.detector.analyze_page(str(image_path))
        
        # Step 2: Handle special cases (two-page spreads)
        split_files = None
        if detection_result.is_two_page_spread and \
           self.config['detection']['two_page_spread'].get('auto_split', False):
            splits_dir = output_dir / 'splits'
            splits_dir.mkdir(exist_ok=True)
            left, right = self.detector.split_two_page_spread(
                str(image_path), str(splits_dir)
            )
            if left and right:
                split_files = [left, right]
                logger.info(f"Split {image_path.name} into two pages")
        
        # Step 3: Classify page
        classification = self.classifier.classify_page(str(image_path), detection_result)
        
        # Override category if detection is confident
        if detection_result.is_two_page_spread and detection_result.confidence > 0.8:
            classification['category'] = 'two_page_spread'
        elif detection_result.is_foldout:
            classification['category'] = 'extended_foldout'
        
        # Create result
        result = ClassificationResult(
            filename=image_path.name,
            original_path=str(image_path),
            category=classification['category'],
            confidence=classification['confidence'],
            confidence_score=classification['confidence_score'],
            is_two_page_spread=detection_result.is_two_page_spread,
            is_foldout=detection_result.is_foldout,
            aspect_ratio=detection_result.aspect_ratio,
            split_files=split_files,
            processing_time=(datetime.now() - start_time).total_seconds(),
            detection_features=detection_result.detection_features,
            error=classification.get('error'),
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    def _save_checkpoint(self):
        """Save processing checkpoint"""
        checkpoint_data = {
            'processed_files': [r.original_path for r in self.results],
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(self.results)
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {len(self.results)} files processed")
    
    def _load_checkpoint(self) -> set:
        """Load checkpoint data"""
        with open(self.checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        return set(checkpoint_data.get('processed_files', []))
    
    def _save_intermediate_results(self, output_dir: Path):
        """Save intermediate results to CSV"""
        df = pd.DataFrame([asdict(r) for r in self.results])
        csv_path = output_dir / f'results_intermediate_{datetime.now():%Y%m%d_%H%M%S}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Intermediate results saved to {csv_path}")
    
    def _save_final_results(self, output_dir: Path) -> pd.DataFrame:
        """Save final classification results"""
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save detailed CSV
        csv_path = output_dir / f'classification_results_{datetime.now():%Y%m%d_%H%M%S}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Final results saved to {csv_path}")
        
        # Generate summary report
        self._generate_summary_report(df, output_dir)
        
        return df
    
    def _generate_summary_report(self, df: pd.DataFrame, output_dir: Path):
        """Generate summary statistics report"""
        report = {
            'processing_summary': {
                'total_images': len(df),
                'successful': len(df[df['error'].isna()]),
                'errors': len(df[df['error'].notna()]),
                'processing_time': df['processing_time'].sum(),
                'average_time_per_image': df['processing_time'].mean()
            },
            'category_distribution': df['category'].value_counts().to_dict(),
            'confidence_distribution': df['confidence'].value_counts().to_dict(),
            'special_pages': {
                'two_page_spreads': df['is_two_page_spread'].sum(),
                'foldouts': df['is_foldout'].sum(),
                'pages_split': len(df[df['split_files'].notna()])
            },
            'low_confidence_pages': df[df['confidence_score'] < 0.5]['filename'].tolist()
        }
        
        report_path = output_dir / f'summary_report_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {report_path}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("CLASSIFICATION SUMMARY")
        print("="*50)
        print(f"Total images processed: {report['processing_summary']['total_images']}")
        print(f"Successful classifications: {report['processing_summary']['successful']}")
        print(f"Errors: {report['processing_summary']['errors']}")
        print(f"Total processing time: {report['processing_summary']['processing_time']:.2f} seconds")
        print(f"\nCategory Distribution:")
        for category, count in report['category_distribution'].items():
            print(f"  {category}: {count}")
        print(f"\nSpecial Pages:")
        print(f"  Two-page spreads: {report['special_pages']['two_page_spreads']}")
        print(f"  Foldouts: {report['special_pages']['foldouts']}")
        print(f"  Pages split: {report['special_pages']['pages_split']}")
        print("="*50)
    
    def _organize_by_category(self, df: pd.DataFrame, output_dir: Path):
        """Organize files into subdirectories by category"""
        import shutil
        
        categories_dir = output_dir / 'organized'
        categories_dir.mkdir(exist_ok=True)
        
        for category in df['category'].unique():
            if category != 'error':
                cat_dir = categories_dir / category
                cat_dir.mkdir(exist_ok=True)
                
                # Copy files to category directory
                category_files = df[df['category'] == category]['original_path'].tolist()
                for file_path in category_files:
                    if Path(file_path).exists():
                        dest = cat_dir / Path(file_path).name
                        shutil.copy2(file_path, dest)
        
        logger.info(f"Files organized into category subdirectories in {categories_dir}")


def main():
    """Main entry point for VOC classifier"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VOC Archive Page Classifier')
    parser.add_argument('--input', required=True, help='Input directory containing page scans')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--config', help='Configuration YAML file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--test-detection', help='Test detection on single image')
    
    args = parser.parse_args()
    
    if args.test_detection:
        # Test detection only
        detector = VOCPageDetector()
        result = detector.analyze_page(args.test_detection)
        print(f"Detection results for {args.test_detection}:")
        print(f"  Two-page spread: {result.is_two_page_spread} (confidence: {result.confidence:.2f})")
        print(f"  Foldout: {result.is_foldout}")
        print(f"  Aspect ratio: {result.aspect_ratio:.2f}")
        print(f"  Features: {result.detection_features}")
    else:
        # Run full classification pipeline
        processor = VOCArchiveProcessor(args.config)
        results = processor.process_directory(args.input, args.output, resume=args.resume)
        print(f"\nProcessing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()