#!/usr/bin/env python3
"""
Simple VOC Archive Page Classifier with Model Swapping
Usage: python classify.py <input_directory> [output_file] [--model MODEL_NAME]
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
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
    # Try to import newer Qwen models if available
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        QWEN25_AVAILABLE = True
    except ImportError:
        QWEN25_AVAILABLE = False

    # Check for accelerate availability
    try:
        import accelerate
        ACCELERATE_AVAILABLE = True
    except ImportError:
        ACCELERATE_AVAILABLE = False

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    QWEN25_AVAILABLE = False
    ACCELERATE_AVAILABLE = False
    logger.error("Required packages not installed. Run: pip install torch transformers pillow")

# Classification questions and their possible answers
CLASSIFICATION_QUESTIONS = {
    "empty_status": ["blank", "non_blank"],
    "page_size": ["standard_page", "extended_foldout", "two_page_spread"],
    "text_layout": ["single_column", "two_column", "mixed_layout"],
    "table_presence": ["table_full", "table_partial", "table_none"],
    "marginalia": ["marginalia", "no_marginalia"],
    "illustrations": ["illustration", "no_illustration"],
    "title_page": ["title_page", "non_title_page"],
    "seals_signatures": ["seal_signature", "no_signatures"],
    "damage": ["damaged_partial", "no_damage"],
    "index_list": ["index_list", "no_index_list"]
}

# Popular VL models for quick testing
POPULAR_MODELS = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "llava-1.6-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-1.6-13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
    "instructblip-7b": "Salesforce/instructblip-vicuna-7b",
    "instructblip-13b": "Salesforce/instructblip-vicuna-13b"
}

def detect_model_family(model_name: str) -> str:
    """Detect model family from model name."""
    model_name = model_name.lower()
    if "qwen2.5-vl" in model_name or "qwen/qwen2.5-vl" in model_name:
        return "qwen2.5-vl"
    elif "qwen2-vl" in model_name or "qwen/qwen2-vl" in model_name:
        return "qwen2-vl"
    elif "llava" in model_name:
        return "llava"
    elif "instructblip" in model_name:
        return "instructblip"
    else:
        return "auto"

# Multi-question classification prompt
CLASSIFICATION_PROMPT = """
Analyze this historical document page and provide a detailed analysis. Please respond in the exact format shown below:

SUMMARY: [Write a 2-3 sentence summary describing the page's visual and layout features]

QUESTION 1 - Empty or not?
Answer: [blank OR non_blank]

QUESTION 2 - Is it a standard or special page size?
Answer: [standard_page OR extended_foldout OR two_page_spread]

QUESTION 3 - What is the layout of the text on the page?
Answer: [single_column OR two_column OR mixed_layout]

QUESTION 4 - Is there a table or tabular data present on the page?
Answer: [table_full OR table_partial OR table_none]

QUESTION 5 - Does the page contain marginalia?
Answer: [marginalia OR no_marginalia]

QUESTION 6 - Does the page contain illustrations or visuals?
Answer: [illustration OR no_illustration]

QUESTION 7 - Is this a title page?
Answer: [title_page OR non_title_page]

QUESTION 8 - Does the page contain seals or signatures?
Answer: [seal_signature OR no_signatures]

QUESTION 9 - Is the page damaged or partially visible?
Answer: [damaged_partial OR no_damage]

QUESTION 10 - Does the page contain indexes, lists, or catalogs?
Answer: [index_list OR no_index_list]

Respond in exactly this format with the exact answer options provided.
"""


class SimpleVOCClassifier:
    """Vision-Language classifier with swappable models."""

    def __init__(self, model_name: str = "qwen2-vl-7b"):
        """Initialize the classifier with specified model."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Required packages not available")

        # Determine best available device
        self.device = self._get_best_device()
        logger.info(f"Using device: {self.device}")

        # Resolve model name from shortcuts
        self.model_name = POPULAR_MODELS.get(model_name, model_name)
        self.model_family = detect_model_family(self.model_name)

        logger.info(f"Loading {self.model_name} (family: {self.model_family})")

        # Load model and processor
        self._load_model()
        logger.info("Model loaded successfully")

    def _get_best_device(self) -> str:
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            logger.info(f"CUDA GPU detected: {device_name}")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Metal Performance Shaders (MPS) detected")
            return "mps"
        else:
            logger.info("Using CPU (no GPU acceleration available)")
            return "cpu"

    def _load_model(self):
        """Load model and processor based on detected family."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            # Determine optimal dtype and device mapping
            use_gpu = self.device in ["cuda", "mps"]
            torch_dtype = torch.float16 if use_gpu else torch.float32

            # Use device_map="auto" only if accelerate is available and CUDA is used
            if self.device == "cuda" and ACCELERATE_AVAILABLE:
                device_map = "auto"
                logger.info("Using accelerate for automatic device mapping")
            else:
                device_map = None
                if self.device == "cuda" and not ACCELERATE_AVAILABLE:
                    logger.warning("Accelerate not available, using manual device assignment")

            logger.info(f"Loading with dtype: {torch_dtype}")

            if self.model_family == "qwen2.5-vl":
                if QWEN25_AVAILABLE:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=True
                    )
                else:
                    logger.warning("Qwen2.5-VL model requested but not available. Trying fallback...")
                    raise ImportError("Qwen2.5-VL not available")
            elif self.model_family == "qwen2-vl":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            elif self.model_family == "llava":
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            elif self.model_family == "instructblip":
                self.model = InstructBlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            else:
                # Try auto-detection
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )

            # Move to device if not using auto device mapping
            if device_map is None:
                self.model = self.model.to(self.device)

            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            logger.info("Falling back to auto-loading without device mapping...")
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Use float32 for fallback
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
                self.model_family = "auto"
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e2:
                raise RuntimeError(f"Failed to load model {self.model_name}: {e2}")

    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """Classify a single image using multi-question approach."""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')

            # Generate classification based on model family
            if self.model_family in ["qwen2-vl", "qwen2.5-vl"]:
                response = self._classify_qwen2vl(image)
            elif self.model_family == "llava":
                response = self._classify_llava(image)
            elif self.model_family == "instructblip":
                response = self._classify_instructblip(image)
            else:
                response = self._classify_generic(image)

            # Parse multi-question response
            classification_results = self._parse_multi_question_response(response)

            return {
                "image_path": str(image_path),
                "summary": classification_results["summary"],
                "classifications": {
                    "empty_status": classification_results["empty_status"],
                    "page_size": classification_results["page_size"],
                    "text_layout": classification_results["text_layout"],
                    "table_presence": classification_results["table_presence"],
                    "marginalia": classification_results["marginalia"],
                    "illustrations": classification_results["illustrations"],
                    "title_page": classification_results["title_page"],
                    "seals_signatures": classification_results["seals_signatures"],
                    "damage": classification_results["damage"],
                    "index_list": classification_results["index_list"]
                },
                "raw_response": response,
                "status": "success",
                "model": self.model_name
            }

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                "image_path": str(image_path),
                "summary": "",
                "classifications": {
                    "empty_status": "error",
                    "page_size": "error",
                    "text_layout": "error",
                    "table_presence": "error",
                    "marginalia": "error",
                    "illustrations": "error",
                    "title_page": "error",
                    "seals_signatures": "error",
                    "damage": "error",
                    "index_list": "error"
                },
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _classify_llava(self, image: Image.Image) -> str:
        """Classify using LLaVA format."""
        prompt = f"USER: <image>\n{CLASSIFICATION_PROMPT}\nASSISTANT:"
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=False)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.split("ASSISTANT:")[-1].strip()

    def _classify_instructblip(self, image: Image.Image) -> str:
        """Classify using InstructBLIP format."""
        inputs = self.processor(images=image, text=CLASSIFICATION_PROMPT, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=False)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _classify_generic(self, image: Image.Image) -> str:
        """Classify using generic format."""
        try:
            # Try Qwen2-VL format first
            return self._classify_qwen2vl(image)
        except Exception:
            try:
                # Try LLaVA format
                return self._classify_llava(image)
            except Exception:
                try:
                    # Try InstructBLIP format
                    return self._classify_instructblip(image)
                except Exception:
                    raise RuntimeError("Unable to classify with any known format")

    def _parse_multi_question_response(self, response: str) -> Dict[str, Any]:
        """Parse multi-question response format."""
        result = {
            "summary": "",
            "empty_status": "unknown",
            "page_size": "unknown",
            "text_layout": "unknown",
            "table_presence": "unknown",
            "marginalia": "unknown",
            "illustrations": "unknown",
            "title_page": "unknown",
            "seals_signatures": "unknown",
            "damage": "unknown",
            "index_list": "unknown"
        }

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Extract summary
            if line.startswith('SUMMARY:'):
                result["summary"] = line.replace('SUMMARY:', '').strip()

            # Extract answers - look for "Answer:" followed by the response
            elif 'Answer:' in line:
                answer_part = line.split('Answer:')[-1].strip().lower()

                # Match to question categories
                for question_key, valid_answers in CLASSIFICATION_QUESTIONS.items():
                    for valid_answer in valid_answers:
                        if valid_answer.lower() in answer_part or answer_part in valid_answer.lower():
                            result[question_key] = valid_answer
                            break

        return result

    def classify_directory(self, input_dir: str) -> List[Dict[str, Any]]:
        """Classify all images in a directory."""
        input_path = Path(input_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        # Find all image files (exclude macOS metadata files)
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = [
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions and not f.name.startswith('._')
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

            # Clear GPU cache after each image for large models
            if self.device == "cuda":
                torch.cuda.empty_cache()

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

        print("\nClassification complete!")
        print(f"Model used: {classifier.model_name}")
        print(f"Total images: {total}")
        print(f"Successfully classified: {successful}")
        print(f"Errors: {errors}")
        print(f"Results saved to: {args.output}")

        if successful > 0:
            # Show category distribution for each question
            question_stats = {}
            for question_key in CLASSIFICATION_QUESTIONS.keys():
                question_stats[question_key] = {}

            for result in results:
                if result['status'] == 'success':
                    classifications = result['classifications']
                    for question_key, answer in classifications.items():
                        if answer not in question_stats[question_key]:
                            question_stats[question_key][answer] = 0
                        question_stats[question_key][answer] += 1

            print("\nClassification distribution:")
            for question_key, answers in question_stats.items():
                print(f"\n{question_key.replace('_', ' ').title()}:")
                for answer, count in sorted(answers.items()):
                    print(f"  {answer}: {count}")

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()