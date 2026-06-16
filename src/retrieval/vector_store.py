import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ..core.logger import setup_logger

logger = setup_logger("vector_store")

class FaissStore:
    """Encapsulates FAISS vector store operations."""
    
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = None
        self.metadata = None

    def load(self):
        """Loads the FAISS index and metadata from disk."""
        if not self.index_path.exists():
            logger.error(f"FAISS index not found at {self.index_path}")
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        
        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        
        logger.info(f"Loading metadata from {self.metadata_path}")
        self.metadata = np.load(self.metadata_path, allow_pickle=True)

    def search(self, query_vec: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Searches the index for the top k nearest neighbors."""
        if self.index is None:
            self.load()
            
        distances, indices = self.index.search(query_vec, k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

    def build_and_save(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Builds a new FAISS index and saves it with metadata."""
        dim = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP with dimension {dim}")
        
        # Normalize for Inner Product (Cosine Similarity)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(self.index_path))
        np.save(self.metadata_path, metadata)
        
        logger.info(f"Saved FAISS index to {self.index_path} and metadata to {self.metadata_path}")
        self.index = index
        self.metadata = metadata
