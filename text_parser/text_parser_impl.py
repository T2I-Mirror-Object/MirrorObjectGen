from text_parser.text_parser import TextParser
from typing import List
import re


class TextParserImpl(TextParser):
    def parse(self, text: str) -> List[str]:
        """
        "a yellow dog, a cute cat, a white lamp in front of the mirror"
        return ["a yellow dog", "a cute cat", "a white lamp"] no mirror

        "a yellow dog, a cute cat, a white lamp, a red chair in front of the mirror"
        return ["a yellow dog", "a cute cat", "a white lamp", "a red chair"]
        """
        items = [item.strip() for item in text.split(',')]
        
        # Remove "in front of the mirror" from each item (case-insensitive)
        cleaned_items = []
        for item in items:
            # Remove "in front of the mirror" phrase (case-insensitive)
            # Handle variations: "in front of the mirror", "in front of mirror", etc.
            cleaned = re.sub(r'\s*in\s+front\s+of\s+(the\s+)?mirror\s*', '', item, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned:  # Only add non-empty items
                cleaned_items.append(cleaned)
        
        return cleaned_items