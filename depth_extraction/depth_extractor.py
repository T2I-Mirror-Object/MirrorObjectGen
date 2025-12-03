from abc import ABC, abstractmethod
from typing import Optional

class DepthMap:
    image_path: str

class DepthExtractor(ABC):
    @abstractmethod
    def extract_depth_map(self, scene) -> DepthMap:
        """
        Return a depth map for the scene.
        """
        raise NotImplementedError("Needs to be implemented by subclass")