# import json
# import pickle
# import os
# from sentence_transformers import SentenceTransformer
# import numpy as np

# # ===== CONFIG =====
# DATA_DIR = "/scratch/cjpenni/departmental-honors/rag_pipeline/amazon_products"  # folder containing all your .jsonl files
# OUTPUT_FILE = "title_embeddings.pkl"

# # ===== LOAD TITLES AND DESCRIPTIONS FROM ALL FILES =====
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# with open(OUTPUT_FILE, "wb") as f_out:
#     for filename in os.listdir(DATA_DIR):
#         print(f"Processing file: {filename}")
#         if filename.endswith(".jsonl"):
#             filepath = os.path.join(DATA_DIR, filename)
#             try:
#                 titles = []
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     for line in f:
#                         data = json.loads(line)
#                         if "title" in data:
#                             titles.append(data["title"])
#                 if titles:
#                     embeddings = model.encode(titles, convert_to_numpy=True, normalize_embeddings=True)
#                     # Save this batch as a tuple (titles, embeddings)
#                     pickle.dump((titles, embeddings), f_out)
#                 print(f"Processed and saved {len(titles)} titles from {filename}")
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")

# print(f"All batches saved to {OUTPUT_FILE}")


import json
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# ===== CONFIG =====
DATA_DIR = "/scratch/cjpenni/departmental-honors/rag_pipeline/amazon_products/with_desc"  # folder containing all your .jsonl files
OUTPUT_FILE = "product_embeddings.pkl"

# ===== LOAD TITLES AND DESCRIPTIONS FROM ALL FILES =====
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with open(OUTPUT_FILE, "wb") as f_out:
    for filename in os.listdir(DATA_DIR):
        print(f"Processing file: {filename}")
        if filename.endswith(".jsonl"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                texts = []
                meta = []
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        title = data.get("title", "")
                        description = data.get("description", "")
                        if title or description:
                            combined = f"title: {title}; description: {description}"
                            texts.append(combined)
                            meta.append({"title": title, "description": description})
                if texts:
                    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                    # Save this batch as a tuple (meta, embeddings)
                    pickle.dump((meta, embeddings), f_out)
                print(f"Processed and saved {len(texts)} items from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

print(f"All batches saved to {OUTPUT_FILE}")
