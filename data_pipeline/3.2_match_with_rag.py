# import json
# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import ast

# # Load your inference data
# json_path = '/scratch/cjpenni/departmental-honors/data_pipeline/inference_data/06OCT2025/new_inferences_20251113_093404.json'
# output_path = '/scratch/cjpenni/departmental-honors/data_pipeline/inference_data/06OCT2025/new_inferences_rag.json'
# with open(json_path, 'r') as f:
#     data = json.load(f)

# # Paths to your FAISS index and embeddings
# PKL_PATH = "/scratch/cjpenni/departmental-honors/rag_pipeline/product_embeddings.pkl"
# INDEX_PATH = "/scratch/cjpenni/departmental-honors/rag_pipeline/product_faiss_index.faiss"

# # Load model (must be the same as used for embeddings.pkl)
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# model = SentenceTransformer(MODEL_NAME)

# # Load ids and embeddings metadata - read ALL batches
# all_meta = []
# with open(PKL_PATH, "rb") as f:
#     while True:
#         try:
#             meta, _ = pickle.load(f)
#             all_meta.extend(meta)
#         except EOFError:
#             break

# ids = all_meta
# print(f"Loaded metadata for {len(ids)} items")

# # Load FAISS index
# index = faiss.read_index(INDEX_PATH)
# print(f"Loaded FAISS index with {index.ntotal} vectors")

# def query_index(text, k=3):
#     """Query the FAISS index and return top k matches"""
#     query_vec = model.encode([text]).astype("float32")
#     D, I = index.search(query_vec, k)
#     results = []
#     for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
#         title = ids[idx]  # map back using ids
#         results.append({
#             "rank": rank + 1,
#             "matched_title": title,
#             "distance": float(dist)
#         })
#     return results

# def safe_load_json(possible_json):
#     """Try to safely parse malformed JSON strings."""
#     if not isinstance(possible_json, str):
#         return possible_json

#     try:
#         # Try normal JSON first
#         return json.loads(possible_json)
#     except json.JSONDecodeError:
#         try:
#             # Try to parse as Python dict-like string (handles single quotes, etc.)
#             return ast.literal_eval(possible_json)
#         except Exception:
#             # Fallback: return empty dict so the loop continues
#             return {}

# # Process each entry
# for entry in data:
#     gpt_output = safe_load_json(entry.get("gpt_output", {}))
#     final_recs = gpt_output.get("final_product_recommendations", [])
#     matched_recs = []

#     if isinstance(final_recs, list):
#         for rec in final_recs:
#             title = rec.get("title", "")
#             desc = rec.get("description", "")
#             combined_text = f"title: {title}; description: {desc}".strip()

#             top_result = query_index(combined_text, k=1)
#             if top_result:
#                 matched_recs.append({
#                     "title": title,
#                     "description": desc,
#                     "rag_matched_title": top_result[0]["matched_title"],
#                     "rag_match_distance": top_result[0]["distance"]
#                 })

#     # Save results under a new key
#     entry["final_product_recommendations_matched"] = matched_recs

# with open(output_path, 'w') as f:
#     json.dump(data, f, indent=4)

# print(f"\nProcessed {len(data)} entries")
# print(f"Output saved to: {output_path}")

import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ast

# Load your inference data
json_path = '/scratch/cjpenni/departmental-honors/data_pipeline/inference_data/06OCT2025/new_inferences_20251113_093404.json'
output_path = '/scratch/cjpenni/departmental-honors/data_pipeline/inference_data/06OCT2025/new_inferences_rag.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Paths to your FAISS index and embeddings
PKL_PATH = "/scratch/cjpenni/departmental-honors/rag_pipeline/product_embeddings.pkl"
INDEX_PATH = "/scratch/cjpenni/departmental-honors/rag_pipeline/product_faiss_index.faiss"

# Load model (must be the same as used for embeddings.pkl)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Load ids and embeddings metadata - read ALL batches
all_meta = []
with open(PKL_PATH, "rb") as f:
    while True:
        try:
            meta, _ = pickle.load(f)   # your file format: (meta_list, embeddings) per pickle batch
            all_meta.extend(meta)
        except EOFError:
            break

ids = all_meta
print(f"Loaded metadata for {len(ids)} items")

# Load FAISS index
index = faiss.read_index(INDEX_PATH)
print(f"Loaded FAISS index with {index.ntotal} vectors")

def _flatten_description(desc):
    """Normalize description which may be a list or string."""
    if desc is None:
        return ""
    if isinstance(desc, list):
        # join list items into single string (safe fallback)
        return " ".join(str(x) for x in desc if x)
    return str(desc)

def query_index(text, k=1):
    """Query the FAISS index and return top k matches as metadata + distance."""
    # ensure float32 numpy vector
    query_vec = model.encode([text]).astype("float32")
    D, I = index.search(query_vec, k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(ids):
            # guard against invalid index
            results.append({"meta": None, "distance": float(dist)})
            continue
        meta = ids[idx]  # metadata dict from your pickle
        # extract fields if present (fall back to the whole meta if keys missing)
        matched_title = meta.get("title") if isinstance(meta, dict) else meta
        matched_description = _flatten_description(meta.get("description") if isinstance(meta, dict) else None)
        results.append({
            "meta": meta,
            "matched_title": matched_title,
            "matched_description": matched_description,
            "distance": float(dist)
        })
    return results

def safe_load_json(possible_json):
    """Try to safely parse malformed JSON strings."""
    if not isinstance(possible_json, str):
        return possible_json

    try:
        # Try normal JSON first
        return json.loads(possible_json)
    except json.JSONDecodeError:
        try:
            # Try to parse as Python dict-like string (handles single quotes, etc.)
            return ast.literal_eval(possible_json)
        except Exception:
            # Fallback: return empty dict so the loop continues
            return {}

# Process each entry
for entry in data:
    gpt_output = safe_load_json(entry.get("gpt_output", {}))
    final_recs = gpt_output.get("final_product_recommendations", [])

    if isinstance(final_recs, list):
        for rec in final_recs:
            title = rec.get("title", "")
            desc = rec.get("description", "")
            combined_text = f"title: {title}; description: {desc}".strip()

            top_result = query_index(combined_text, k=1)
            if top_result and top_result[0]["meta"] is not None:
                match = top_result[0]
                # Option 1: overwrite title/description directly with match fields
                rec["title"] = match["matched_title"]
                rec["description"] = match["matched_description"]

    # write modified gpt_output back in case it was a string originally
    entry["gpt_output"] = gpt_output

# Save to new file
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"\nProcessed {len(data)} entries")
print(f"Output saved to: {output_path}")
