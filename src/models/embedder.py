from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from .interfaces import Embedder
from ..core.logger import setup_logger

logger = setup_logger("embedder")

class SentenceTransformerEmbedder(Embedder):
    """Implementation of Embedder using SentenceTransformers."""
    
    def __init__(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)

    @property
    def dimension(self) -> int:
        return self._dimension
