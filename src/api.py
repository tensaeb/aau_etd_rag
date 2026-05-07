import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Dict, Any

from .core.config_loader import ConfigLoader
from .core.logger import setup_logger
from .models.embedder import SentenceTransformerEmbedder
from .models.reranker import CrossEncoderReranker
from .models.llm import LMStudioLLM
from .retrieval.vector_store import FaissStore
from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.hybrid_retriever import HybridRetriever
from .rag.context_builder import ContextBuilder
from .rag.engine import RAGEngine

logger = setup_logger("api")

# --- GLOBAL STATE ---
# We use a dictionary to hold our components so they can be initialized in lifespan
state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    logger.info("Initializing RAG components...")
    try:
        config_loader = ConfigLoader()
        config = config_loader.all
        
        # 1. Models
        embedder = SentenceTransformerEmbedder(config["models"]["embedding"])
        reranker = CrossEncoderReranker(config["models"]["cross_encoder"])
        llm = LMStudioLLM(
            url=config["lm_studio"]["url"],
            model=config["lm_studio"]["model"],
            temperature=config["lm_studio"]["temperature"]
        )
        
        # 2. Retrieval
        faiss_store = FaissStore(
            index_path=config_loader.get_path("faiss.index_path"),
            metadata_path=config_loader.get_path("data.embedding_dir") / config["faiss"]["metadata_file"]
        )
        faiss_store.load()
        
        bm25_retriever = BM25Retriever(
            index_path=config_loader.get_path("faiss.index_path").parent / "bm25_index.pkl"
        )
        bm25_retriever.load()
        
        hybrid_retriever = HybridRetriever(faiss_store, bm25_retriever, embedder)
        
        # 3. RAG Engine
        context_builder = ContextBuilder(chunk_dir=config_loader.get_path("data.chunk_dir"))
        engine = RAGEngine(hybrid_retriever, reranker, llm, context_builder, config)
        
        state["engine"] = engine
        logger.info("RAG components initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
        # We don't exit here so the app can still start and return 503s
    
    yield
    logger.info("Shutting down...")
    state.clear()

# --- API SETUP ---
app = FastAPI(
    title="AAU ETD RAG API",
    description="Refactored production-ready RAG API following SOLID principles.",
    version="2.0.0",
    lifespan=lifespan
)

# --- DATA MODELS ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, example="What are the main findings of the study?")

class AskResponse(BaseModel):
    answer: str

# --- API ENDPOINTS ---
@app.get("/health")
async def health_check():
    if "engine" not in state:
        return {"status": "unhealthy", "reason": "Pipeline not initialized"}
    return {"status": "healthy"}

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    if "engine" not in state:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not ready.")
    
    try:
        answer = state["engine"].ask(request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during RAG processing.")

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    server_cfg = cfg.get("server", {})
    uvicorn.run(app, host=server_cfg.get("host", "0.0.0.0"), port=server_cfg.get("port", 8000))
