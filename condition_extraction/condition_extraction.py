from abc import ABC, abstractmethod

class SegmentationMap:
    image_path: str
    json_path: str

class DepthMap:
    image_path: str

class ConditionExtraction(ABC):
    @abstractmethod
    def extract_depthmap(self, scene) -> DepthMap:
        """
        Retu
        """
        raise NotImplementedError("Needs to be implemented by subclass")
    @abstractmethod
    def extract_segmentation_map(self, scene) -> SegmentationMap:
        raise NotImplementedError("Needs to be implemented by subclass")