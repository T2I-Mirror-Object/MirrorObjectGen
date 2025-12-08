from text_parser.text_parser import TextParser
from typing import List
import re

class TextParserImpl2(TextParser):
    def parse(self, text: str) -> List[str]:
        """
        Example:
        "a chair in front of a mirror, both the chair and its perfect mirror reflection are visible"
        return ["a chair"]

        "a dog, a cat, a chair in front of a mirror, both the objects and their reflections are visible"
        return ["a dog", "a cat", "a chair"]
        """

        # Remove the trailing clause:
        # "both the X and its/... reflection(s) are visible"
        text = re.sub(
            r',\s*both\s+the\s+.+?\s+and\s+(?:its|their)\s+.+?reflections?\s+are\s+visible\s*$', 
            '', 
            text, 
            flags=re.IGNORECASE
        )

        # Split by comma
        items = [item.strip() for item in text.split(',')]

        # Remove "in front of a mirror/the mirror"
        cleaned_items = []
        for item in items:
            cleaned = re.sub(
                r'\s*in\s+front\s+of\s+(?:a|the)?\s*mirror\s*', 
                '', 
                item, 
                flags=re.IGNORECASE
            ).strip()

            if cleaned:
                cleaned_items.append(cleaned)

        return cleaned_items
