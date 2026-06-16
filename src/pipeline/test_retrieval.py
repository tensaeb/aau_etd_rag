import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

index = faiss.read_index("indexes/aau_faiss.index")
metadata = np.load("data/embeddings/metadata.npy", allow_pickle=True)
model = SentenceTransformer("intfloat/e5-base-v2")

CHUNK_DIR = Path("data/chunks")

query = "What are the traditional networking paradigms?"
q_vec = model.encode([f"query: {query}"])

D, I = index.search(q_vec, k=3)

print(f"Query: {query}\n")
print("=" * 80)

for rank, (idx, distance) in enumerate(zip(I[0], D[0]), 1):
    meta = metadata[idx]
    source_file = CHUNK_DIR / f"{meta['source']}.txt"
    
    # Load the chunk content
    if source_file.exists():
        chunks = source_file.read_text(encoding="utf-8").split("\n\n---\n\n")
        chunk_content = chunks[meta['chunk_id']]
    else:
        chunk_content = "[Chunk file not found]"
    
    print(f"\nRank {rank} (Distance: {distance:.4f})")
    print(f"Source: {meta['source']}")
    print(f"Chunk ID: {meta['chunk_id']}")
    print(f"\nContent:\n{chunk_content[:500]}..." if len(chunk_content) > 500 else f"\nContent:\n{chunk_content}")
    print("=" * 80)
