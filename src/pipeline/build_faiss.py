import sys
import numpy as np
from pathlib import Path
from ..core.config_loader import ConfigLoader
from ..core.logger import setup_logger
from ..retrieval.vector_store import FaissStore

logger = setup_logger("build_faiss")

def main():
    try:
        config_loader = ConfigLoader()
        config = config_loader.all
        
        emb_dir = config_loader.get_path("data.embedding_dir")
        embeddings_file = emb_dir / config["faiss"]["embeddings_file"]
        metadata_file = emb_dir / config["faiss"]["metadata_file"]
        index_path = config_loader.get_path("faiss.index_path")

        if not embeddings_file.exists():
            logger.error(f"Embeddings file not found: {embeddings_file}")
            return

        logger.info(f"Loading embeddings from: {embeddings_file}")
        embeddings = np.load(embeddings_file)
        metadata = np.load(metadata_file, allow_pickle=True)

        store = FaissStore(str(index_path), str(metadata_file))
        store.build_and_save(embeddings, metadata)

        logger.info("FAISS index built successfully.")

    except Exception as e:
        logger.critical(f"Critical error in FAISS build: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
