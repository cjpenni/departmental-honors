import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# ========== CONFIGURATION ==========
INPUT_CSV = "combined_inferences.csv"
CATEGORY_COL = "category"
INFERENCE_COL = "inference"
OUTPUT_CSV = "unified_output.csv"
CATEGORY_THRESHOLD = 0.65
INFERENCE_THRESHOLD = 0.0

# ========== LOAD DATA ==========
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")
print(f"Unique categories before unification: {df[CATEGORY_COL].nunique()}")
print(f"Unique inferences before unification: {df[INFERENCE_COL].nunique()}")

# ========== INITIALIZE MODEL ==========
model = SentenceTransformer('all-MiniLM-L6-v2')

# --------- FUNCTION TO SEMANTICALLY CLUSTER AND UNIFY A COLUMN ---------
def unify_column(entries, distance_threshold):
    entries = [str(e) for e in entries]  # ensure strings
    embeddings = model.encode(entries, convert_to_tensor=True)
    embeddings_np = normalize(embeddings.cpu().numpy())

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    clustering.fit(embeddings_np)
    cluster_labels = clustering.labels_

    cluster_map = {}
    for cluster_id in set(cluster_labels):
        cluster_entries = [entries[i] for i in range(len(entries)) if cluster_labels[i] == cluster_id]
        cluster_embeds = model.encode(cluster_entries, convert_to_tensor=True)
        cosine_scores = util.cos_sim(cluster_embeds, cluster_embeds)
        avg_scores = cosine_scores.mean(dim=1)
        rep = cluster_entries[avg_scores.argmax().item()]
        cluster_map[cluster_id] = rep

    unified_entries = [cluster_map[cluster_labels[i]] for i in range(len(entries))]
    return unified_entries

# ========== UNIFY CATEGORIES ==========
df[CATEGORY_COL] = unify_column(df[CATEGORY_COL], CATEGORY_THRESHOLD)
print(f"Unique categories after unification: {df[CATEGORY_COL].nunique()}")

# ========== UNIFY INFERENCES (AND ADD "interested in") ==========
df[INFERENCE_COL] = unify_column(df[INFERENCE_COL], INFERENCE_THRESHOLD)
df[INFERENCE_COL] = df[INFERENCE_COL].apply(lambda x: f"interested in {x}" if not x.lower().startswith("interested in") else x)
print(f"Unique inferences after unification: {df[INFERENCE_COL].nunique()}")

# ========== SAVE OUTPUT ==========
df.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Output saved to {OUTPUT_CSV}")
