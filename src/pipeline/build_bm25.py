import sys
from pathlib import Path
from ..core.config_loader import ConfigLoader
from ..core.logger import setup_logger
from ..retrieval.bm25_retriever import BM25Retriever

logger = setup_logger("build_bm25")

def main():
    try:
        config_loader = ConfigLoader()
        chunk_dir = config_loader.get_path("data.chunk_dir")
        index_path = config_loader.get_path("faiss.index_path").parent / "bm25_index.pkl"

        logger.info("Starting BM25 index build...")
        corpus = []
        doc_mapping = []

        chunk_files = list(chunk_dir.glob("*.txt"))
        if not chunk_files:
            logger.warning("No chunk files found.")
            return

        for file in chunk_files:
            # Fix source key: remove "_chunks" to match retrieval expectations
            source_name = file.stem
            
            content = file.read_text(encoding="utf-8")
            chunks = content.split("\n\n---\n\n")
            
            for i, chunk in enumerate(chunks):
                corpus.append(chunk.split()) # BM25 works on tokenized text
                doc_mapping.append({"source": source_name, "chunk_id": i})

        retriever = BM25Retriever(str(index_path))
        retriever.build_and_save(corpus, doc_mapping)

        logger.info("BM25 index build completed.")

    except Exception as e:
        logger.critical(f"Critical error in BM25 build: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
