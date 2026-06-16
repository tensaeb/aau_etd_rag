from ..core.config_loader import ConfigLoader
from ..core.logger import setup_logger
from ..models.embedder import SentenceTransformerEmbedder
from ..retrieval.vector_store import FaissStore
from ..rag.context_builder import ContextBuilder

logger = setup_logger("show_passage")

def main():
    config_loader = ConfigLoader()
    config = config_loader.all
    
    # 1. Initialize components
    embedder = SentenceTransformerEmbedder(config["models"]["embedding"])
    
    faiss_store = FaissStore(
        index_path=config_loader.get_path("faiss.index_path"),
        metadata_path=config_loader.get_path("data.embedding_dir") / config["faiss"]["metadata_file"]
    )
    
    context_builder = ContextBuilder(chunk_dir=config_loader.get_path("data.chunk_dir"))
    
    # 2. Retrieve
    query = "What are the benefits of artificial intelligence in universities?"
    logger.info(f"Query: {query}")
    
    query_vec = embedder.encode([f"query: {query}"], normalize=True)
    results = faiss_store.search(query_vec, k=3)
    
    # 3. Display
    print("\n" + "=" * 80)
    print("RELEVANT PASSAGES")
    print("=" * 80)
    
    for rank, meta in enumerate(results, 1):
        content = context_builder.load_chunk(meta['source'], meta['chunk_id'])
        print(f"\nRANK {rank}")
        print(f"Source: {meta['source']} | Chunk: {meta['chunk_id']}")
        print("-" * 40)
        print(content)
        print("-" * 80)

if __name__ == "__main__":
    main()
