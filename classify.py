#!/usr/bin/env python3
"""
Simple VOC Archive Page Classifier with Model Swapping
Usage: python classify.py <input_directory> [output_file] [--model MODEL_NAME]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import (
        AutoProcessor, AutoModelForCausalLM,
        Qwen2VLForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        InstructBlipForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("Required packages not installed. Run: pip install torch transformers pillow")

# Classification categories
CATEGORIES = [
    "single_column", "two_column", "table_full", "table_partial",
    "marginalia", "two_page_spread", "extended_foldout", "illustration",
    "title_page", "blank", "seal_signature", "mixed_layout",
    "damaged_partial", "index_list"
]

# Popular VL models for quick testing
POPULAR_MODELS = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
    "llava-1.6-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-1.6-13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
    "instructblip-7b": "Salesforce/instructblip-vicuna-7b",
    "instructblip-13b": "Salesforce/instructblip-vicuna-13b"
}

def detect_model_family(model_name: str) -> str:
    """Detect model family from model name."""
    model_name = model_name.lower()
    if "qwen2-vl" in model_name or "qwen/qwen2-vl" in model_name:
        return "qwen2-vl"
    elif "llava" in model_name:
        return "llava"
    elif "instructblip" in model_name:
        return "instructblip"
    else:
        return "auto"

# Classification prompt
CLASSIFICATION_PROMPT = """
Analyze this historical document page and classify it into one of these categories:

- single_column: Text in a single column layout
- two_column: Text arranged in two columns
- table_full: Page dominated by tabular data
- table_partial: Page with some tabular content mixed with text
- marginalia: Text with significant margin notes or annotations
- two_page_spread: Two pages side by side in one image
- extended_foldout: Unusually wide page that unfolds
- illustration: Page dominated by drawings, maps, or diagrams
- title_page: Title page or cover page
- blank: Mostly blank or empty page
- seal_signature: Page with official seals or signatures
- mixed_layout: Complex layout combining multiple elements
- damaged_partial: Damaged or partially visible page
- index_list: Index, list, or catalog format

Respond with only the category name, nothing else.
"""


class SimpleVOCClassifier:
    """Vision-Language classifier with swappable models."""

    def __init__(self, model_name: str = "qwen2-vl-7b"):
        """Initialize the classifier with specified model."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Required packages not available")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Resolve model name from shortcuts
        self.model_name = POPULAR_MODELS.get(model_name, model_name)
        self.model_family = detect_model_family(self.model_name)

        logger.info(f"Loading {self.model_name} (family: {self.model_family})")

        # Load model and processor
        self._load_model()
        logger.info("Model loaded successfully")

    def _load_model(self):
        """Load model and processor based on detected family."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            if self.model_family == "qwen2-vl":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            elif self.model_family == "llava":
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            elif self.model_family == "instructblip":
                self.model = InstructBlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                # Try auto-detection
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            logger.info("Falling back to auto-loading...")
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                self.model_family = "auto"
            except Exception as e2:
                raise RuntimeError(f"Failed to load model {self.model_name}: {e2}")

    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """Classify a single image."""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')

            # Generate classification based on model family
            if self.model_family == "qwen2-vl":
                response = self._classify_qwen2vl(image)
            elif self.model_family == "llava":
                response = self._classify_llava(image)
            elif self.model_family == "instructblip":
                response = self._classify_instructblip(image)
            else:
                response = self._classify_generic(image)

            # Find best matching category
            category = self._match_category(response)

            return {
                "image_path": str(image_path),
                "category": category,
                "raw_response": response,
                "status": "success",
                "model": self.model_name
            }

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                "image_path": str(image_path),
                "category": "error",
                "raw_response": str(e),
                "status": "error",
                "model": self.model_name
            }

    def _classify_qwen2vl(self, image: Image.Image) -> str:
        """Classify using Qwen2-VL format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": CLASSIFICATION_PROMPT}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().lower()

    def _classify_llava(self, image: Image.Image) -> str:
        """Classify using LLaVA format."""
        prompt = f"USER: <image>\n{CLASSIFICATION_PROMPT}\nASSISTANT:"
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.split("ASSISTANT:")[-1].strip().lower()

    def _classify_instructblip(self, image: Image.Image) -> str:
        """Classify using InstructBLIP format."""
        inputs = self.processor(images=image, text=CLASSIFICATION_PROMPT, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()

    def _classify_generic(self, image: Image.Image) -> str:
        """Classify using generic format."""
        try:
            # Try Qwen2-VL format first
            return self._classify_qwen2vl(image)
        except:
            try:
                # Try LLaVA format
                return self._classify_llava(image)
            except:
                try:
                    # Try InstructBLIP format
                    return self._classify_instructblip(image)
                except:
                    raise RuntimeError("Unable to classify with any known format")

    def _match_category(self, response: str) -> str:
        """Match response to valid category."""
        response = response.lower().strip()

        # Direct match
        if response in CATEGORIES:
            return response

        # Partial match
        for category in CATEGORIES:
            if category in response or response in category:
                return category

        # Default fallback
        return "mixed_layout"

    def classify_directory(self, input_dir: str) -> List[Dict[str, Any]]:
        """Classify all images in a directory."""
        input_path = Path(input_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = [
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return []

        logger.info(f"Found {len(image_files)} images to classify")

        results = []
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
            result = self.classify_image(str(image_file))
            results.append(result)

        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="VOC Archive Page Classifier with Model Swapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available model shortcuts:
{chr(10).join([f'  {k}: {v}' for k, v in POPULAR_MODELS.items()])}

Examples:
  python classify.py /path/to/images
  python classify.py /path/to/images --model qwen2-vl-2b
  python classify.py /path/to/images --model llava-1.6-7b --output results.json
  python classify.py /path/to/images --model "microsoft/kosmos-2-patch14-224"
        """)

    parser.add_argument("input_dir", nargs="?", help="Input directory containing images")
    parser.add_argument("--output", "-o", default="classification_results.json",
                        help="Output JSON file (default: classification_results.json)")
    parser.add_argument("--model", "-m", default="qwen2-vl-7b",
                        help="Model name or shortcut (default: qwen2-vl-7b)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available model shortcuts and exit")

    args = parser.parse_args()

    if args.list_models:
        print("Available model shortcuts:")
        for shortcut, full_name in POPULAR_MODELS.items():
            print(f"  {shortcut}: {full_name}")
        sys.exit(0)

    if not args.input_dir:
        parser.error("input_dir is required unless using --list-models")

    try:
        # Initialize classifier with specified model
        classifier = SimpleVOCClassifier(model_name=args.model)

        # Process directory
        results = classifier.classify_directory(args.input_dir)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        total = len(results)
        successful = len([r for r in results if r['status'] == 'success'])
        errors = total - successful

        print(f"\nClassification complete!")
        print(f"Model used: {classifier.model_name}")
        print(f"Total images: {total}")
        print(f"Successfully classified: {successful}")
        print(f"Errors: {errors}")
        print(f"Results saved to: {args.output}")

        if successful > 0:
            # Show category distribution
            categories = {}
            for result in results:
                if result['status'] == 'success':
                    cat = result['category']
                    categories[cat] = categories.get(cat, 0) + 1

            print("\nCategory distribution:")
            for cat, count in sorted(categories.items()):
                print(f"  {cat}: {count}")

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()