from abc import ABC, abstractmethod
from typing import Optional

class EdgeMap:
    image_path: str

class EdgeExtractor(ABC):
    @abstractmethod
    def extract_edge_map(self, scene) -> EdgeMap:
        """
        Return an edge map for the scene.
        """
        raise NotImplementedError("Needs to be implemented by subclass")
