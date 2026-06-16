import sys
import numpy as np
from pathlib import Path
from ..core.config_loader import ConfigLoader
from ..core.logger import setup_logger
from ..models.embedder import SentenceTransformerEmbedder

logger = setup_logger("embed_chunks")

def main():
    try:
        config_loader = ConfigLoader()
        config = config_loader.all
        
        chunk_dir = config_loader.get_path("data.chunk_dir")
        emb_dir = config_loader.get_path("data.embedding_dir")
        emb_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings_file = emb_dir / config["faiss"]["embeddings_file"]
        metadata_file = emb_dir / config["faiss"]["metadata_file"]

        embedder = SentenceTransformerEmbedder(config["models"]["embedding"])

        all_chunks = []
        metadata = []

        # Fix glob: only process files with _chunks.txt suffix
        chunk_files = list(chunk_dir.glob("*.txt"))
        if not chunk_files:
            logger.warning("No chunk files found.")
            return

        for file in chunk_files:
            source_name = file.stem
            chunks = file.read_text(encoding="utf-8").split("\n\n---\n\n")
            
            for i, chunk in enumerate(chunks):
                # Add required prefix for e5 models
                all_chunks.append(f"passage: {chunk}")
                metadata.append({"source": source_name, "chunk_id": i})

        logger.info(f"Embedding {len(all_chunks)} chunks...")
        embeddings = embedder.encode(all_chunks, normalize=True)

        logger.info(f"Saving embeddings (shape={embeddings.shape}) to {embeddings_file}")
        np.save(embeddings_file, embeddings)
        np.save(metadata_file, metadata)

        logger.info("Embedding completed successfully.")

    except Exception as e:
        logger.critical(f"Critical error in embedding pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
