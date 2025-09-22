import pickle
import faiss
import numpy as np

# Path to your pickle file
PKL_PATH = "title_embeddings.pkl"
# Path to save the FAISS index
INDEX_PATH = "faiss_index.faiss"

def build_faiss_index(pkl_path, index_path):
    # Load pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # If tuple (ids, embeddings), extract embeddings
    if isinstance(data, tuple) and len(data) == 2:
        ids, embeddings = data
    else:
        embeddings = data

    # Ensure float32 for FAISS
    embeddings = np.array(embeddings).astype("float32")

    # Get dimension
    d = embeddings.shape[1]

    # Create FAISS index (L2 distance)
    index = faiss.IndexFlatL2(d)

    # Add embeddings
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index with {index.ntotal} vectors to {index_path}")

if __name__ == "__main__":
    build_faiss_index(PKL_PATH, INDEX_PATH)
