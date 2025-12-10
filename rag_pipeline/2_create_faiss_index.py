import pickle
import faiss
import numpy as np

PKL_PATH = "product_embeddings.pkl"
INDEX_PATH = "product_faiss_index.faiss"
META_PATH = "product_faiss_metadata.pkl"

def build_faiss_index_stream(pkl_path, index_path, meta_path):
    index = None
    all_meta = []

    with open(pkl_path, "rb") as f:
        while True:
            try:
                print("Loading next batch from pickle...")
                meta, embeddings = pickle.load(f)
                embeddings = np.array(embeddings, dtype="float32")

                # Initialize index once we know the embedding dimension
                if index is None:
                    d = embeddings.shape[1]
                    index = faiss.IndexFlatL2(d)

                # Add embeddings batch-wise
                index.add(embeddings)
                all_meta.extend(meta)

            except EOFError:
                break

    # Save index and metadata
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(all_meta, f)

    print(f"Saved FAISS index with {index.ntotal} vectors")
    print(f"Saved metadata for {len(all_meta)} items")

if __name__ == "__main__":
    build_faiss_index_stream(PKL_PATH, INDEX_PATH, META_PATH)
