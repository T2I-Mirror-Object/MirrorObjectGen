from abc import ABC, abstractmethod
from typing import List

class SceneComposition(ABC):
    @abstractmethod
    def compose_scene(self, objs: List[str]):
        """
        return a scene with instance id.
        """
        raise NotImplementedError("Needs to be implemented by subclass")
    