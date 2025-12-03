from text_parser.text_parser import TextParser
from typing import List
import re


class TextParserImpl2(TextParser):
    def parse(self, text: str) -> List[str]:
        """
        "a chair in front of a mirror, both the chair and its reflection are visible"
        return ["a chair"]

        "a dog, a cat, a chair in front of a mirror, both the objects and their reflections are visible"
        return ["a dog", "a cat", "a chair"]
        """
        # Remove the "both ... are visible" clause from the end
        # This pattern matches "both the X and Y reflection(s) are visible"
        text = re.sub(r',\s*both\s+the\s+.+?\s+and\s+(?:its|their)\s+reflections?\s+are\s+visible\s*$', '', text, flags=re.IGNORECASE)
        
        # Split by comma to get individual items
        items = [item.strip() for item in text.split(',')]
        
        # Remove "in front of a mirror" phrase from items
        cleaned_items = []
        for item in items:
            # Remove "in front of a mirror" or "in front of the mirror"
            cleaned = re.sub(r'\s*in\s+front\s+of\s+(a\s+)?mirror\s*', '', item, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned:  # Only add non-empty items
                cleaned_items.append(cleaned)
        
        return cleaned_items