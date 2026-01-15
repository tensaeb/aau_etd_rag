"""Show relevant passages from ETD documents for a query."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Load resources
index = faiss.read_index("indexes/aau_faiss.index")
metadata = np.load("data/embeddings/metadata.npy", allow_pickle=True)
embed_model = SentenceTransformer("intfloat/e5-base-v2")
CHUNK_DIR = Path("data/chunks")

def load_chunk(source, chunk_id):
    """Load a specific chunk from a chunk file."""
    file = CHUNK_DIR / f"{source}.txt"
    if not file.exists():
        return None
    chunks = file.read_text(encoding="utf-8").split("\n\n---\n\n")
    if chunk_id >= len(chunks):
        return None
    return chunks[chunk_id]

def retrieve_and_show(query, k=3):
    """Retrieve and display the most relevant passages."""
    print(f"\nQuery: {query}\n")
    print("=" * 80)
    
    # Retrieve
    q_vec = embed_model.encode([f"query: {query}"])
    D, I = index.search(q_vec, k)
    
    # Display results
    for rank, (idx, distance) in enumerate(zip(I[0], D[0]), 1):
        meta = metadata[idx]
        chunk_content = load_chunk(meta['source'], meta['chunk_id'])
        
        if chunk_content:
            print(f"\n{'='*80}")
            print(f"Rank {rank} (Similarity Distance: {distance:.4f})")
            print(f"Source: {meta['source']}")
            print(f"Chunk ID: {meta['chunk_id']}")
            print(f"{'='*80}")
            print(f"\n{chunk_content}")
            print(f"\n{'='*80}\n")

if __name__ == "__main__":
    # Example query - you can change this
    query = "What are the benefits of artificial intelligence in universities?"
    retrieve_and_show(query, k=3)

