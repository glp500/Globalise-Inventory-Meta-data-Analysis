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

# Enhanced historical document classification prompt
CLASSIFICATION_PROMPT = """
You are analyzing a historical document page from the VOC (Dutch East India Company) archives, dating from approximately 1602-1800. These documents typically contain:
- Dutch handwritten text in period scripts (secretary hand, cursive)
- Official records, letters, inventories, and administrative documents
- Various ink types (iron gall, carbon-based) that may have faded or changed color
- Aged paper with potential staining, foxing, or physical damage
- Formal Dutch administrative formatting and layout conventions

Examine this page carefully and provide a detailed analysis in the EXACT format below:

SUMMARY: [Write 2-3 sentences describing the page's visual appearance, text density, writing style, condition, and overall layout characteristics you observe]

QUESTION 1 - Content presence: Is this page substantially empty or does it contain meaningful content?
Answer: [blank OR non_blank]
Explanation: [Describe what you see - blank space, faint marks, substantial text, etc.]

QUESTION 2 - Page format: What is the physical format of this document page?
Answer: [standard_page OR extended_foldout OR two_page_spread]
Explanation: [Describe the aspect ratio, if you see fold marks, multiple page boundaries, or unusual dimensions]

QUESTION 3 - Text arrangement: How is the written text organized on the page?
Answer: [single_column OR two_column OR mixed_layout]
Explanation: [Describe the column structure, text blocks, and how content is spatially arranged]

QUESTION 4 - Tabular content: Are there tables, lists, or structured data arrangements?
Answer: [table_full OR table_partial OR table_none]
Explanation: [Describe any tabular structures, ruled lines, columns of data, or list formatting you observe]

QUESTION 5 - Marginal annotations: Are there notes, comments, or text written in the margins?
Answer: [marginalia OR no_marginalia]
Explanation: [Describe any marginal text, whether it appears to be contemporary or later additions]

QUESTION 6 - Visual elements: Does the page contain drawings, diagrams, maps, or decorative elements?
Answer: [illustration OR no_illustration]
Explanation: [Describe any non-textual visual content including decorative initials, sketches, or diagrams]

QUESTION 7 - Document type: Is this a title page, cover, or introductory page?
Answer: [title_page OR non_title_page]
Explanation: [Describe positioning of text, presence of titles, formal presentation suggesting a cover page]

QUESTION 8 - Authentication marks: Are there official seals, stamps, or signature marks?
Answer: [seal_signature OR no_signatures]
Explanation: [Describe any wax seals, official stamps, signature marks, or authentication elements]

QUESTION 9 - Physical condition: Is the page damaged, torn, stained, or partially obscured?
Answer: [damaged_partial OR no_damage]
Explanation: [Describe any damage, staining, holes, tears, fading, or areas where content is illegible]

QUESTION 10 - Content type: Does this page contain systematic lists, indexes, or catalog-style entries?
Answer: [index_list OR no_index_list]
Explanation: [Describe if content appears to be an organized list, index, inventory, or catalog format]

Respond in exactly this format. Be specific about what you observe in the historical document context.
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
                    "empty_status": {
                        "answer": classification_results["empty_status"]["answer"],
                        "explanation": classification_results["empty_status"]["explanation"]
                    },
                    "page_size": {
                        "answer": classification_results["page_size"]["answer"],
                        "explanation": classification_results["page_size"]["explanation"]
                    },
                    "text_layout": {
                        "answer": classification_results["text_layout"]["answer"],
                        "explanation": classification_results["text_layout"]["explanation"]
                    },
                    "table_presence": {
                        "answer": classification_results["table_presence"]["answer"],
                        "explanation": classification_results["table_presence"]["explanation"]
                    },
                    "marginalia": {
                        "answer": classification_results["marginalia"]["answer"],
                        "explanation": classification_results["marginalia"]["explanation"]
                    },
                    "illustrations": {
                        "answer": classification_results["illustrations"]["answer"],
                        "explanation": classification_results["illustrations"]["explanation"]
                    },
                    "title_page": {
                        "answer": classification_results["title_page"]["answer"],
                        "explanation": classification_results["title_page"]["explanation"]
                    },
                    "seals_signatures": {
                        "answer": classification_results["seals_signatures"]["answer"],
                        "explanation": classification_results["seals_signatures"]["explanation"]
                    },
                    "damage": {
                        "answer": classification_results["damage"]["answer"],
                        "explanation": classification_results["damage"]["explanation"]
                    },
                    "index_list": {
                        "answer": classification_results["index_list"]["answer"],
                        "explanation": classification_results["index_list"]["explanation"]
                    }
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
                    "empty_status": {"answer": "error", "explanation": ""},
                    "page_size": {"answer": "error", "explanation": ""},
                    "text_layout": {"answer": "error", "explanation": ""},
                    "table_presence": {"answer": "error", "explanation": ""},
                    "marginalia": {"answer": "error", "explanation": ""},
                    "illustrations": {"answer": "error", "explanation": ""},
                    "title_page": {"answer": "error", "explanation": ""},
                    "seals_signatures": {"answer": "error", "explanation": ""},
                    "damage": {"answer": "error", "explanation": ""},
                    "index_list": {"answer": "error", "explanation": ""}
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=600, temperature=0.1, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _classify_llava(self, image: Image.Image) -> str:
        """Classify using LLaVA format."""
        prompt = f"USER: <image>\n{CLASSIFICATION_PROMPT}\nASSISTANT:"
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=600, temperature=0.1, do_sample=False)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.split("ASSISTANT:")[-1].strip()

    def _classify_instructblip(self, image: Image.Image) -> str:
        """Classify using InstructBLIP format."""
        inputs = self.processor(images=image, text=CLASSIFICATION_PROMPT, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=600, temperature=0.1, do_sample=False)

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
        """Parse multi-question response format with explanations - robust version."""
        result = {
            "summary": "",
            "empty_status": {"answer": "unknown", "explanation": ""},
            "page_size": {"answer": "unknown", "explanation": ""},
            "text_layout": {"answer": "unknown", "explanation": ""},
            "table_presence": {"answer": "unknown", "explanation": ""},
            "marginalia": {"answer": "unknown", "explanation": ""},
            "illustrations": {"answer": "unknown", "explanation": ""},
            "title_page": {"answer": "unknown", "explanation": ""},
            "seals_signatures": {"answer": "unknown", "explanation": ""},
            "damage": {"answer": "unknown", "explanation": ""},
            "index_list": {"answer": "unknown", "explanation": ""}
        }

        # Question mapping with multiple possible identifiers
        question_patterns = {
            "empty_status": ["content presence", "question 1", "empty or not"],
            "page_size": ["page format", "question 2", "page size", "standard or special"],
            "text_layout": ["text arrangement", "question 3", "text layout", "layout of"],
            "table_presence": ["tabular content", "question 4", "table", "tabular data"],
            "marginalia": ["marginal annotations", "question 5", "marginalia"],
            "illustrations": ["visual elements", "question 6", "illustrations", "visuals"],
            "title_page": ["document type", "question 7", "title page"],
            "seals_signatures": ["authentication marks", "question 8", "seals", "signatures"],
            "damage": ["physical condition", "question 9", "damage", "damaged"],
            "index_list": ["content type", "question 10", "index", "list", "catalog"]
        }

        lines = response.strip().split('\n')
        current_question = None
        collecting_explanation = False
        current_explanation = ""

        for line in lines:
            line_lower = line.strip().lower()
            line = line.strip()

            # Extract summary
            if line.startswith('SUMMARY:') or line_lower.startswith('summary:'):
                result["summary"] = line.split(':', 1)[1].strip() if ':' in line else ""
                continue

            # Identify which question we're processing - more robust matching
            question_found = False
            for question_key, patterns in question_patterns.items():
                for pattern in patterns:
                    if pattern in line_lower and ('question' in line_lower or ':' in line):
                        current_question = question_key
                        collecting_explanation = False
                        question_found = True
                        break
                if question_found:
                    break

            # Extract answers - multiple formats
            if current_question and ('answer:' in line_lower or
                                   (line_lower.strip().startswith('[') and line_lower.strip().endswith(']'))):

                # Handle "Answer: [option]" format
                if 'answer:' in line_lower:
                    answer_part = line.split(':', 1)[1].strip().lower()
                    answer_part = answer_part.replace('[', '').replace(']', '')
                # Handle "[option]" format
                else:
                    answer_part = line.strip().lower().replace('[', '').replace(']', '')

                # Match to question categories with fuzzy matching
                if current_question in CLASSIFICATION_QUESTIONS:
                    for valid_answer in CLASSIFICATION_QUESTIONS[current_question]:
                        if (valid_answer.lower() in answer_part or
                            answer_part in valid_answer.lower() or
                            self._fuzzy_match(answer_part, valid_answer.lower())):
                            result[current_question]["answer"] = valid_answer
                            break

            # Extract explanations - handle multiline
            elif current_question and ('explanation:' in line_lower or collecting_explanation):
                if 'explanation:' in line_lower:
                    # Start collecting explanation
                    explanation_start = line.split(':', 1)[1].strip()
                    current_explanation = explanation_start
                    collecting_explanation = True
                elif collecting_explanation and line.strip():
                    # Continue collecting explanation until we hit another section
                    if not any(pattern in line_lower for patterns in question_patterns.values() for pattern in patterns):
                        if not line_lower.startswith('answer:') and not line_lower.startswith('question'):
                            current_explanation += " " + line
                    else:
                        # Hit a new section, save current explanation
                        if current_question and current_explanation.strip():
                            result[current_question]["explanation"] = current_explanation.strip()
                        collecting_explanation = False
                        current_explanation = ""

        # Save any remaining explanation
        if current_question and current_explanation.strip():
            result[current_question]["explanation"] = current_explanation.strip()

        return result

    def _fuzzy_match(self, text: str, target: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for answer validation."""
        # Remove common separators and check similarity
        text_clean = text.replace('_', ' ').replace('-', ' ')
        target_clean = target.replace('_', ' ').replace('-', ' ')

        # Check if significant portion matches
        words_text = text_clean.split()
        words_target = target_clean.split()

        if not words_text or not words_target:
            return False

        matches = sum(1 for word in words_text if word in words_target)
        return matches / max(len(words_text), len(words_target)) >= threshold

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
                    for question_key, classification_data in classifications.items():
                        answer = classification_data['answer']
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