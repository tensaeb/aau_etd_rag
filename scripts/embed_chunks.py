from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

MODEL_NAME = "intfloat/e5-base-v2"
model = SentenceTransformer(MODEL_NAME)

CHUNK_DIR = Path("data/chunks")
EMB_DIR = Path("data/embeddings")
EMB_DIR.mkdir(exist_ok=True)

all_chunks = []
metadata = []

for file in CHUNK_DIR.glob("*_chunks.txt"):
    chunks = file.read_text(encoding="utf-8").split("\n\n---\n\n")
    for i, chunk in enumerate(chunks):
        all_chunks.append(f"passage: {chunk}")
        metadata.append({
            "source": file.stem,
            "chunk_id": i
        })

embeddings = model.encode(all_chunks, show_progress_bar=True)

np.save(EMB_DIR / "embeddings.npy", embeddings)
np.save(EMB_DIR / "metadata.npy", metadata)

print("Embeddings shape:", embeddings.shape)
