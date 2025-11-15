from abc import ABC, abstractmethod

class ConditionMap:
    image_path: str
    json_path: str | None

class ConditionExtractor(ABC):
    @abstractmethod
    def extract_condition_map(self, scene) -> ConditionMap:
        """
        Return a condition map for the scene.
        """
        raise NotImplementedError("Needs to be implemented by subclass")