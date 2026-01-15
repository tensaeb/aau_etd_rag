from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configuration
MODEL_NAME = config["models"]["embedding"]
CHUNK_DIR = Path(config["data"]["chunk_dir"])
EMB_DIR = Path(config["data"]["embedding_dir"])
EMBEDDINGS_FILE = EMB_DIR / config["faiss"]["embeddings_file"]
METADATA_FILE = EMB_DIR / config["faiss"]["metadata_file"]

# Initialize model
model = SentenceTransformer(MODEL_NAME)
EMB_DIR.mkdir(exist_ok=True)

# Prepare lists
all_chunks = []
metadata = []

# Process each chunk file
for file in CHUNK_DIR.glob("*.txt"):
    # The stem for chunk files is like "doc1_chunks.txt", so remove "_chunks"
    source_stem = file.stem.replace("_chunks", "")
    chunks = file.read_text(encoding="utf-8").split("\n\n---\n\n")
    for i, chunk in enumerate(chunks):
        # Add the required prefix for the e5 model
        all_chunks.append(f"passage: {chunk}")
        metadata.append({
            "source": source_stem,
            "chunk_id": i
        })

# Generate embeddings
print(f"Embedding {len(all_chunks)} chunks with model: {MODEL_NAME}...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# Save embeddings and metadata
np.save(EMBEDDINGS_FILE, embeddings)
np.save(METADATA_FILE, metadata)

print("Embeddings shape:", embeddings.shape)
print(f"Saved embeddings to: {EMBEDDINGS_FILE}")
print(f"Saved metadata to: {METADATA_FILE}")
