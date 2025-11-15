from abc import ABC, abstractmethod
from condition_extraction import DepthMap, SegmentationMap


class ConditionToImage(ABC):
    @abstractmethod
    def convert_condition_to_image(self, depthmap: DepthMap=None, segmentation_map: SegmentationMap=None):
        raise NotImplementedError("Needs to be implemented by subclass")