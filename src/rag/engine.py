from typing import List, Dict, Any
from ..retrieval.hybrid_retriever import HybridRetriever
from ..models.interfaces import Reranker, LLMProvider
from .context_builder import ContextBuilder
from ..core.logger import setup_logger

logger = setup_logger("rag_engine")

class RAGEngine:
    """The main RAG orchestrator that ties all components together."""
    
    def __init__(
        self, 
        retriever: HybridRetriever, 
        reranker: Reranker, 
        llm: LLMProvider, 
        context_builder: ContextBuilder,
        config: Dict[str, Any]
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.context_builder = context_builder
        self.config = config

    def ask(self, query: str) -> str:
        """Processes a query through the full RAG pipeline."""
        logger.info(f"Processing question: {query}")
        
        # 1. Retrieve
        candidates = self.retriever.retrieve(
            query, 
            k=self.config["rag"]["retrieval_candidates"]
        )
        
        # 2. Rerank
        logger.info("Reranking candidates...")
        pairs = [
            (query, self.context_builder.load_chunk(item["source"], item["chunk_id"])) 
            for item in candidates
        ]
        scores = self.reranker.predict(pairs)
        
        scored_items = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_k = self.config["rag"]["top_k"]
        final_items = [item for item, score in scored_items[:top_k]]
        
        # 3. Build Context
        context = self.context_builder.build(
            final_items, 
            max_context_chars=self.config["rag"]["max_context_chars"]
        )
        
        # 4. Generate Answer
        prompt = self._build_prompt(query, context)
        logger.info("Generating answer from LLM...")
        return self.llm.generate(prompt)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Answer ONLY using the context below. If the answer is not present, say so.

Context:
{context}

Question:
{query}

Answer:
"""
