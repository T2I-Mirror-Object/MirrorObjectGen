from abc import ABC, abstractmethod
from typing import List


class TextTo3D(ABC):
    @abstractmethod
    def convert_text_to_3d(self, text: str):
        raise NotImplementedError("Needs to be implemented by subclass")

    @abstractmethod
    def convert_multiple_texts_to_3d(self, texts: List[str], output_dir: str) -> List[str]:
        """
        Convert multiple texts to 3d objects and save them to directories.
        """
        raise NotImplementedError("Needs to be implemented by subclass")