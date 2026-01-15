
import yaml
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHUNK_DIR = Path(config["data"]["chunk_dir"])
INDEX_PATH = Path(config["faiss"]["index_path"]).parent # Store in the same dir as FAISS
BM25_INDEX_FILE = INDEX_PATH / "bm25_index.pkl"

# --- MAIN EXECUTION ---
print("Starting BM25 index build...")
corpus = []
doc_mapping = [] # To map corpus index back to source file and chunk_id

# 1. Load all chunks into a corpus
print("  - Loading and tokenizing chunks...")
for file in CHUNK_DIR.glob("*_chunks.txt"):
    chunks = file.read_text(encoding="utf-8").split("\n\n---\n\n")
    for i, chunk in enumerate(chunks):
        corpus.append(chunk.split()) # BM25 works on tokenized text
        doc_mapping.append({"source": file.stem, "chunk_id": i})

# 2. Create the BM25 index from the corpus
print(f"  - Building BM25 index for {len(corpus)} documents...")
bm25 = BM25Okapi(corpus)

# 3. Save the index and the document mapping
print(f"  - Saving BM25 index to: {BM25_INDEX_FILE}")
with open(BM25_INDEX_FILE, "wb") as f:
    pickle.dump({"bm25": bm25, "doc_mapping": doc_mapping}, f)

print("\nBM25 index build completed successfully.")
