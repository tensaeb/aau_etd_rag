from typing import List
from sentence_transformers import CrossEncoder
from .interfaces import Reranker
from ..core.logger import setup_logger

logger = setup_logger("reranker")

class CrossEncoderReranker(Reranker):
    """Implementation of Reranker using CrossEncoders."""
    
    def __init__(self, model_name: str):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def predict(self, pairs: List[List[str]]) -> List[float]:
        return self.model.predict(pairs).tolist()
