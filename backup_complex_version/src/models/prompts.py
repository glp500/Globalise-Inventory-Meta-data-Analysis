"""
Classification prompts for VOC document analysis.
"""

from typing import List, Dict, Any


class VOCClassificationPrompts:
    """Prompts for VOC document classification using Qwen2-VL."""
    
    def __init__(self):
        self.categories = [
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
    
    def get_classification_prompt(self) -> str:
        """
        Get the main classification prompt for VOC documents.
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze this historical VOC (Dutch East India Company) document page and classify it into one of these categories:

Categories:
- single_column: Text arranged in a single column
- two_column: Text arranged in two columns
- table_full: Full page table with rows and columns
- table_partial: Partial table mixed with text
- marginalia: Page with extensive margin notes
- two_page_spread: Two pages displayed side by side
- extended_foldout: Extended/folded page with multiple panels
- illustration: Page primarily containing drawings, maps, or diagrams
- title_page: Title or cover page
- blank: Mostly empty page
- seal_signature: Page with official seals or signatures
- mixed_layout: Mixed content types with no clear primary layout
- damaged_partial: Damaged page with missing or unclear content
- index_list: Index, list, or catalog page

Consider these aspects:
1. Text layout and column structure
2. Presence of tables, lists, or structured data
3. Visual elements like illustrations, seals, or decorations
4. Overall page organization and content type
5. Historical context of 17th-19th century VOC documents

Respond with JSON format:
{{
    "category": "category_name",
    "confidence": 0.0-1.0,
    "features": {{
        "text_density": "high/medium/low",
        "has_tables": true/false,
        "has_illustrations": true/false,
        "layout_type": "description"
    }}
}}"""
        
        return prompt
    
    def get_two_page_detection_prompt(self) -> str:
        """Get prompt for two-page spread detection."""
        return """Analyze this image to determine if it shows two pages side by side.

Look for:
- Two distinct page boundaries
- Center fold or binding line
- Symmetrical layout on left and right sides
- Different content on each side
- Aspect ratio suggesting side-by-side pages

Respond with JSON:
{
    "is_two_page": true/false,
    "confidence": 0.0-1.0,
    "evidence": "description of visual evidence"
}"""
    
    def get_foldout_detection_prompt(self) -> str:
        """Get prompt for foldout page detection."""
        return """Analyze this image to determine if it shows an extended or foldout page.

Look for:
- Unusual aspect ratio (very wide or very tall)
- Multiple panel sections
- Fold lines or creases
- Content that extends beyond normal page dimensions
- Maps, large diagrams, or extensive tables

Respond with JSON:
{
    "is_foldout": true/false,
    "confidence": 0.0-1.0,
    "orientation": "horizontal/vertical",
    "estimated_panels": number
}"""
    
    def get_damage_assessment_prompt(self) -> str:
        """Get prompt for assessing page damage."""
        return """Assess the condition and readability of this historical document page.

Look for:
- Missing or torn sections
- Stains, discoloration, or fading
- Illegible text areas
- Physical damage affecting content
- Overall preservation quality

Respond with JSON:
{
    "damage_level": "none/minor/moderate/severe",
    "readability": "excellent/good/fair/poor",
    "damage_types": ["list", "of", "damage", "types"],
    "content_accessibility": 0.0-1.0
}"""
    
    def get_content_type_prompt(self) -> str:
        """Get prompt for identifying content types."""
        return """Identify the primary content types present in this VOC document page.

Content types to identify:
- Administrative text (orders, reports, correspondence)
- Financial records (accounts, inventories, payments)
- Legal documents (contracts, agreements)
- Ship logs and navigation records
- Trade documentation (manifests, bills of lading)
- Personal correspondence
- Official proclamations
- Maps and charts
- Illustrations and diagrams

Respond with JSON:
{
    "primary_content": "content_type",
    "secondary_content": ["list", "of", "other", "types"],
    "confidence": 0.0-1.0,
    "language": "Dutch/Latin/other",
    "era_indicators": ["list", "of", "dating", "clues"]
}"""