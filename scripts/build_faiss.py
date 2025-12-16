import faiss
import numpy as np

embeddings = np.load("data/embeddings/embeddings.npy")
dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "indexes/aau_faiss.index")
print("Total vectors:", index.ntotal)
