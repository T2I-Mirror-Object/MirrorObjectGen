from abc import ABC, abstractmethod
from typing import List


class TextParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> List[str]:
        raise NotImplementedError("Needs to be implemented by subclass")