from typing import List, Dict, Any, Tuple
from .vector_store import FaissStore
from .bm25_retriever import BM25Retriever
from ..models.interfaces import Embedder
from ..core.logger import setup_logger

logger = setup_logger("hybrid_retriever")

class HybridRetriever:
    """Combines FAISS and BM25 results using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, faiss_store: FaissStore, bm25_retriever: BM25Retriever, embedder: Embedder):
        self.faiss_store = faiss_store
        self.bm25_retriever = bm25_retriever
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Performs hybrid retrieval."""
        logger.info(f"Hybrid retrieval for: '{query}'")
        
        # 1. Semantic Search
        query_vec = self.embedder.encode([f"query: {query}"], normalize=True)
        faiss_results = self.faiss_store.search(query_vec, k=k)
        
        # 2. Keyword Search
        bm25_results = self.bm25_retriever.search(query, k=k)
        
        # 3. Reciprocal Rank Fusion
        return self._rrf([faiss_results, bm25_results], k=rrf_k)

    def _rrf(self, result_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        ranked_items = {}
        for results in result_lists:
            for rank, result in enumerate(results):
                doc_key = (result['source'], result['chunk_id'])
                if doc_key not in ranked_items:
                    ranked_items[doc_key] = 0
                ranked_items[doc_key] += 1 / (k + rank + 1)
        
        sorted_items = sorted(ranked_items.items(), key=lambda x: x[1], reverse=True)
        
        return [{"source": key[0], "chunk_id": key[1]} for key, score in sorted_items]
