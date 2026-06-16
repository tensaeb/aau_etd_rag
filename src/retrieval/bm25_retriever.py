import pickle
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from ..core.logger import setup_logger

logger = setup_logger("bm25_retriever")

class BM25Retriever:
    """Encapsulates BM25 keyword search operations."""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.bm25 = None
        self.doc_mapping = None

    def load(self):
        """Loads the BM25 index from disk."""
        if not self.index_path.exists():
            logger.error(f"BM25 index not found at {self.index_path}")
            raise FileNotFoundError(f"Index not found: {self.index_path}")
            
        logger.info(f"Loading BM25 index from {self.index_path}")
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.doc_mapping = data["doc_mapping"]

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Searches the BM25 index."""
        if self.bm25 is None:
            self.load()
            
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        # Get top k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [self.doc_mapping[i] for i in top_indices]

    def build_and_save(self, corpus: List[List[str]], doc_mapping: List[Dict[str, Any]]):
        """Builds a new BM25 index and saves it."""
        logger.info(f"Building BM25 index for {len(corpus)} documents")
        bm25 = BM25Okapi(corpus)
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "doc_mapping": doc_mapping}, f)
            
        logger.info(f"Saved BM25 index to {self.index_path}")
        self.bm25 = bm25
        self.doc_mapping = doc_mapping
