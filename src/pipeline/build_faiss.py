import faiss
import numpy as np
import yaml
from pathlib import Path

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configuration from YAML
EMB_DIR = Path(config["data"]["embedding_dir"])
EMBEDDINGS_FILE = EMB_DIR / config["faiss"]["embeddings_file"]
INDEX_PATH = Path(config["faiss"]["index_path"])

# Load embeddings
print(f"Loading embeddings from: {EMBEDDINGS_FILE}")
embeddings = np.load(EMBEDDINGS_FILE)
dim = embeddings.shape[1]
print(f"Embeddings loaded. Shape: {embeddings.shape}")

# Build the FAISS index
print("Building FAISS index...")
index = faiss.IndexFlatL2(dim)  # Using L2 distance for similarity
index.add(embeddings)

# Save the index
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX_PATH))

print(f"FAISS index built successfully.")
print(f"  - Total vectors: {index.ntotal}")
print(f"  - Index saved to: {INDEX_PATH}")
