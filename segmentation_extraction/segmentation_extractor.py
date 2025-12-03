from abc import ABC, abstractmethod
from typing import Optional

class SegmentationMap:
    image_path: str
    json_path: Optional[str]

class SegmentationExtractor(ABC):
    @abstractmethod
    def extract_segmentation_map(self, scene) -> SegmentationMap:
        """
        Return a segmentation map for the scene.
        """
        raise NotImplementedError("Needs to be implemented by subclass")