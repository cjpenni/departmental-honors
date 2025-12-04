# import pickle
# import faiss
# import numpy as np

# # Path to your pickle file
# PKL_PATH = "title_embeddings.pkl"
# # Path to save the FAISS index
# INDEX_PATH = "title_faiss/faiss_index.faiss"

# def build_faiss_index(pkl_path, index_path):
#     # Load pickle
#     with open(pkl_path, "rb") as f:
#         data = pickle.load(f)

#     # If tuple (ids, embeddings), extract embeddings
#     if isinstance(data, tuple) and len(data) == 2:
#         ids, embeddings = data
#     else:
#         embeddings = data

#     # Ensure float32 for FAISS
#     embeddings = np.array(embeddings).astype("float32")

#     # Get dimension
#     d = embeddings.shape[1]

#     # Create FAISS index (L2 distance)
#     index = faiss.IndexFlatL2(d)

#     # Add embeddings
#     index.add(embeddings)

#     # Save FAISS index
#     faiss.write_index(index, index_path)
#     print(f"✅ Saved FAISS index with {index.ntotal} vectors to {index_path}")

# if __name__ == "__main__":
#     build_faiss_index(PKL_PATH, INDEX_PATH)



# import pickle
# import faiss
# import numpy as np

# # Path to your pickle file
# PKL_PATH = "product_embeddings.pkl"
# # Path to save the FAISS index
# INDEX_PATH = "product_faiss_index.faiss"
# # Path to save metadata
# META_PATH = "product_faiss_metadata.pkl"

# def build_faiss_index(pkl_path, index_path, meta_path):
#     all_embeddings = []
#     all_meta = []

#     # Read all batches from pickle file
#     with open(pkl_path, "rb") as f:
#         while True:
#             try:
#                 meta, embeddings = pickle.load(f)
#                 all_embeddings.append(embeddings)
#                 all_meta.extend(meta)
#             except EOFError:
#                 break

#     # Concatenate all embeddings
#     embeddings = np.vstack(all_embeddings).astype("float32")
#     d = embeddings.shape[1]

#     # Create FAISS index (L2 distance)
#     index = faiss.IndexFlatL2(d)
#     index.add(embeddings)

#     # Save FAISS index and metadata
#     faiss.write_index(index, index_path)
#     with open(meta_path, "wb") as f:
#         pickle.dump(all_meta, f)

#     print(f"✅ Saved FAISS index with {index.ntotal} vectors to {index_path}")
#     print(f"✅ Saved metadata for {len(all_meta)} items to {meta_path}")

# if __name__ == "__main__":
#     build_faiss_index(PKL_PATH, INDEX_PATH, META_PATH)

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
                print("⏳ Loading next batch from pickle...")
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

    print(f"✅ Saved FAISS index with {index.ntotal} vectors")
    print(f"✅ Saved metadata for {len(all_meta)} items")

if __name__ == "__main__":
    build_faiss_index_stream(PKL_PATH, INDEX_PATH, META_PATH)
