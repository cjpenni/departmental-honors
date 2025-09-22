import json
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# ===== CONFIG =====
DATA_DIR = "/scratch/cjpenni/departmental-honors/rag_pipeline/amazon_products"  # folder containing all your .jsonl files
OUTPUT_FILE = "title_embeddings.pkl"
#EMBEDDING_MODEL = "hkunlp/instructor-large"  # fast and high-quality

# ===== LOAD TITLES FROM ALL FILES =====
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with open(OUTPUT_FILE, "wb") as f_out:
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                titles = []
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if "title" in data:
                            titles.append(data["title"])
                if titles:
                    embeddings = model.encode(titles, convert_to_numpy=True, normalize_embeddings=True)
                    # Save this batch as a tuple (titles, embeddings)
                    pickle.dump((titles, embeddings), f_out)
                print(f"Processed and saved {len(titles)} titles from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

print(f"All batches saved to {OUTPUT_FILE}")
