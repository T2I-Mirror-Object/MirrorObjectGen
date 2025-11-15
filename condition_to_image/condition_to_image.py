from abc import ABC, abstractmethod
from condition_extraction import ConditionMap


class ConditionToImage(ABC):
    @abstractmethod
    def convert_condition_to_image(self, condition_map: ConditionMap):
        raise NotImplementedError("Needs to be implemented by subclass")