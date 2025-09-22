import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
PKL_PATH = "title_embeddings.pkl"
INDEX_PATH = "faiss_index.faiss"

# Load model (must be the same as used for embeddings.pkl)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Load ids and embeddings metadata
with open(PKL_PATH, "rb") as f:
    ids, _ = pickle.load(f)

# Load FAISS index
index = faiss.read_index(INDEX_PATH)
print(f"Loaded FAISS index with {index.ntotal} vectors")

def query_index(text, k=3):
    # Encode query
    query_vec = model.encode([text]).astype("float32")

    # Search top k
    D, I = index.search(query_vec, k)

    results = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        title = ids[idx]  # map back using ids
        results.append((rank+1, title, dist))

    return results

if __name__ == "__main__":
    query = "gardening book"
    top_results = query_index(query, k=3)

    print(f"\n🔎 Query: {query}\nTop 3 results:")
    for rank, title, dist in top_results:
        print(f"{rank}. {title} (distance={dist:.4f})")
