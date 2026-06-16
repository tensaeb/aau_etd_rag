from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class Embedder(ABC):
    """Interface for text embedding models."""
    @abstractmethod
    def encode(self, texts: List[str], normalize: bool = True) -> Any:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

class Reranker(ABC):
    """Interface for reranking models."""
    @abstractmethod
    def predict(self, pairs: List[List[str]]) -> List[float]:
        pass

class LLMProvider(ABC):
    """Interface for LLM interaction."""
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
