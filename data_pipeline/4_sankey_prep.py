import pandas as pd
import numpy as np
import re
import json
import requests
import plotly.graph_objects as go

OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'gpt-oss:20b'

# ---- Load single JSON file ----
json_path = '/scratch/cjpenni/departmental-honors/data_pipeline/inference_data/06OCT2025/inferences_user_combinations_20251006_101522.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# ---- Combine and flatten JSON data ----
combined_data = []
bad_entries = 0

for entry in data:
    gpt_output = entry.get("gpt_output")
    
    if not isinstance(gpt_output, dict):
        bad_entries += 1
        continue
    
    source = gpt_output.get("columns", "")
    inferences = gpt_output.get("inferences", [])
    
    if not source or not isinstance(inferences, list):
        bad_entries += 1
        continue
    
    for inf in inferences:
        if not isinstance(inf, dict):
            continue
        
        flattened_entry = {
            "source": source,
            "combo_size": entry.get("combo_size", None),
            "combined_cols": entry.get("combined_cols", []),
            **inf
        }
        combined_data.append(flattened_entry)

print(f"✅ Loaded {len(combined_data)} valid entries")
if bad_entries > 0:
    print(f"⚠️ Skipped {bad_entries} malformed entries")

# ---- Extract unique source types from JSON ----
all_source_cols = set()
for entry in data:
    combined_cols = entry.get("combined_cols", [])
    all_source_cols.update(combined_cols)

source_types = sorted(list(all_source_cols))
print(f"✅ Detected source types from JSON: {source_types}")

# ---- Transform to binary DataFrame ----
rows = []
for item in combined_data:
    sources = [s.strip() for s in item['source'].split('AND')]
    row = {src: 1 if src in sources else 0 for src in source_types}
    row['Inference'] = item.get('inference', '')
    row['Recommendation'] = item.get('recommendation', '')
    row['Sensitivity Score'] = item.get('sensitivity', np.nan)
    row['Commonness Score'] = item.get('commonness', np.nan)
    row['Uncommonness Score'] = item.get('uncommonness', np.nan)
    row['combo_size'] = item.get('combo_size', np.nan)
    rows.append(row)

df = pd.DataFrame(rows)
print(f"✅ Created binary DataFrame with {len(df)} rows and {len(df.columns)} columns")

# ---- Column ordering ----
ordered_columns = source_types + ['Inference', 'Recommendation', 'Sensitivity Score', 'Commonness Score', 'Uncommonness Score', 'combo_size']
binary_df = df[ordered_columns]
print("✅ Reordered DataFrame columns")

# ---- Clean Inference text ----
pattern = r'Interested in |interested in '
binary_df['Inference'] = binary_df['Inference'].str.replace(pattern, '', regex=True)
print("✅ Cleaned Inference text")

# ---- Define Run Prompt Function ----
def run_prompt(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                full_response += data["response"]
    return full_response.strip()

# ---- Interest Label Grouping ----
def generate_interest_labels(df):
    interest_list = df['Inference'].dropna().unique().tolist()

    prompt = f"""
      1. Relabel duplicates AND similar values in {interest_list} under the same label.
      2. If no similar values are found, return them as-is in both columns (i.e., same value for interest and label).
      3. Ensure that the assigned label are meaningful, concise, and representative of the grouped interests.
      4. Follow EXACT output format

      |interest|label|
      |interest 1|label 1|
      |interest 2|label 1|
      ...
      |interest n|label n|
      |interest n+1|label n|

      DO NOT include any extra text or explanation.
      Return in the EXACT format
    """
    assistant_reply = run_prompt(prompt)

    mapping = {}
    for line in assistant_reply.strip().split("\n")[1:]:
        if line.startswith("|") and line.count("|") == 3:
            parts = line.split("|")
            if len(parts) == 4:
                 _, interest, label, _ = parts
                 mapping[interest.strip().lower()] = label.strip()
            else:
                 print(f"Skipping malformed line: {line}")

    df["Grouped Inference"] = df["Inference"].apply(
        lambda x: mapping.get(x.lower(), x) if isinstance(x, str) else x
    )
    return df

binary_df = generate_interest_labels(binary_df)
print("✅ Generated Grouped Inference labels")

# ---- Sensitivity/Commonness/Uncommonness Filtering ----
binary_df['Sensitivity Score'] = pd.to_numeric(binary_df['Sensitivity Score'], errors='coerce')
binary_df['Commonness Score'] = pd.to_numeric(binary_df['Commonness Score'], errors='coerce')
binary_df['Uncommonness Score'] = pd.to_numeric(binary_df['Uncommonness Score'], errors='coerce')

sensitive_rows = binary_df['Sensitivity Score'] > 7
uncommon_rows = binary_df['Uncommonness Score'] > 7
common_rows = binary_df['Commonness Score'] < 4

binary_df.loc[sensitive_rows, 'Grouped Inference'] = binary_df.loc[sensitive_rows, 'Inference']
binary_df.loc[uncommon_rows, 'Grouped Inference'] = binary_df.loc[uncommon_rows, 'Inference']
binary_df.loc[common_rows, 'Grouped Inference'] = binary_df.loc[common_rows, 'Inference']

print("✅ Applied Sensitivity, Commonness, and Uncommonness filtering")

# ---- Source Count ----
source_cols = [src for src in source_types if src in binary_df.columns]
binary_df['source_count'] = binary_df[source_cols].sum(axis=1)
print("✅ Calculated source_count")

# ---- Sankey Table Construction ----
source_inference_links = []
for _, row in binary_df.iterrows():
    weight = 1 / row['source_count'] if row['source_count'] > 0 else 0
    for col in source_cols:
        if row[col] == 1:
            source_inference_links.append({
                'Source': col,
                'Target': row['Grouped Inference'],
                'Weight': weight
            })

inference_rec_links = [
    {
        'Source': row['Grouped Inference'],
        'Target': row['Recommendation'],
        'Weight': 1
    } for _, row in binary_df.iterrows()
]

sankey_table = pd.DataFrame(source_inference_links + inference_rec_links)
print("✅ Constructed Sankey table")

all_nodes = pd.unique(sankey_table[['Source', 'Target']].values.ravel())
node_map = {label: i for i, label in enumerate(all_nodes)}
sankey_table['source_id'] = sankey_table['Source'].map(node_map)
sankey_table['target_id'] = sankey_table['Target'].map(node_map)

true_sources = sorted(set(sankey_table['Source']) - set(sankey_table['Target']))
print(f"✅ Identified {len(true_sources)} true source columns")

# ---- Strictly Isolated Sankey Helper ----
def make_sankey_df_strictly_isolated(filtered_sources=None):
    """
    Create Sankey data for a specific combination of sources.
    Only includes inferences that come from EXACTLY these sources.
    """
    if filtered_sources is None or not filtered_sources:
        return dict(source=[], target=[], value=[], label=[])
    
    if isinstance(filtered_sources, str):
        selected_sources = [filtered_sources]
    else:
        selected_sources = filtered_sources
    
    combined_condition = pd.Series([True] * len(binary_df))
    
    print(f"Filtering for EXACT combination: {selected_sources}")
    
    # Must have ALL selected sources = 1
    for source in selected_sources:
        if source in source_cols:
            combined_condition = combined_condition & (binary_df[source] == 1)
    
    # Must have ALL other sources = 0
    other_sources = [src for src in source_cols if src not in selected_sources]
    for source in other_sources:
        if source in source_cols:
            combined_condition = combined_condition & (binary_df[source] == 0)
    
    filtered_binary_df = binary_df[combined_condition].copy()
    
    if filtered_binary_df.empty:
        print(f"  ⚠️ No data found for combination: {selected_sources}")
        return dict(source=[], target=[], value=[], label=[])
    
    print(f"  ✅ Found {len(filtered_binary_df)} rows for this combination")
    
    # Build links
    strict_source_inference_links = []
    for _, row in filtered_binary_df.iterrows():
        weight = 1
        for col in selected_sources:
            strict_source_inference_links.append({
                'Source': col,
                'Target': row['Grouped Inference'],
                'Weight': weight
            })
    
    strict_inference_rec_links = [
        {
            'Source': row['Grouped Inference'],
            'Target': row['Recommendation'],
            'Weight': 1
        } for _, row in filtered_binary_df.iterrows()
    ]
    
    if strict_source_inference_links or strict_inference_rec_links:
        strict_sankey_data = pd.concat([
            pd.DataFrame(strict_source_inference_links),
            pd.DataFrame(strict_inference_rec_links)
        ], ignore_index=True)
    else:
        strict_sankey_data = pd.DataFrame(columns=['Source', 'Target', 'Weight'])
    
    if strict_sankey_data.empty:
        return dict(source=[], target=[], value=[], label=[])
    
    # Map nodes
    filtered_nodes = pd.unique(strict_sankey_data[['Source', 'Target']].values.ravel())
    filtered_node_map = {label: i for i, label in enumerate(filtered_nodes)}
    
    strict_sankey_data['source_id'] = strict_sankey_data['Source'].map(filtered_node_map)
    strict_sankey_data['target_id'] = strict_sankey_data['Target'].map(filtered_node_map)
    
    node_labels = filtered_nodes.tolist()
    
    return dict(
        source=strict_sankey_data['source_id'].tolist(),
        target=strict_sankey_data['target_id'].tolist(),
        value=strict_sankey_data['Weight'].tolist(),
        label=node_labels
    )

# ---- Extract unique combinations from JSON ----
unique_combinations = []
for entry in data:
    combined_cols = entry.get("combined_cols", [])
    combo_size = entry.get("combo_size", 0)
    combo_tuple = tuple(sorted(combined_cols))
    
    if combo_tuple and combo_tuple not in unique_combinations:
        unique_combinations.append(combo_tuple)

unique_combinations = sorted(unique_combinations, key=lambda x: (len(x), x))

print(f"✅ Extracted {len(unique_combinations)} unique combinations from JSON")
print(f"   Combo sizes present: {sorted(set(len(c) for c in unique_combinations))}")

# ---- Sankey Plot with buttons ----
print("Creating initial Sankey figure...")
fig = go.Figure()

initial_sankey_trace = go.Sankey(
    arrangement='snap',
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes.tolist()
    ),
    link=dict(
        source=sankey_table['source_id'].tolist(),
        target=sankey_table['target_id'].tolist(),
        value=sankey_table['Weight'].tolist()
    )
)
fig.add_trace(initial_sankey_trace)

# Create buttons - start with "All (Unfiltered)"
buttons = [
    dict(
        label="All (Unfiltered)",
        method="update",
        args=[
            {
                "link": [{
                    "source": sankey_table['source_id'].tolist(),
                    "target": sankey_table['target_id'].tolist(),
                    "value": sankey_table['Weight'].tolist()
                }],
                "node": [{
                    "label": all_nodes.tolist(),
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5}
                }]
            }
        ]
    )
]

# Add a button for each combination that exists in the JSON
for combo in unique_combinations:
    combo_list = list(combo)
    combo_label = " & ".join(combo_list)
    
    # Call the function ONCE and store result
    sankey_data = make_sankey_df_strictly_isolated(combo_list)
    
    # Skip if no data found
    if not sankey_data['label']:
        print(f"  ⚠️ Skipping button for {combo_label} (no data)")
        continue
    
    # Create properly formatted button
    buttons.append(
        dict(
            label=combo_label,
            method="update",
            args=[
                {
                    "link": [{
                        "source": sankey_data['source'],
                        "target": sankey_data['target'],
                        "value": sankey_data['value']
                    }],
                    "node": [{
                        "label": sankey_data['label'],
                        "pad": 15,
                        "thickness": 20,
                        "line": {"color": "black", "width": 0.5}
                    }]
                }
            ]
        )
    )
    print(f"  ✅ Created button for: {combo_label}")

print(f"✅ Created {len(buttons)} buttons total")

# Create dropdown menu
updatemenus = [
    dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.1,
        y=1.1
    )
]

fig.update_layout(
    title_text="Strictly Isolated Source-Filtered Sankey Diagram",
    font_size=12,
    updatemenus=updatemenus
)

print("Saving strictly isolated sankey figure...")
output_path = "/scratch/cjpenni/departmental-honors/data_pipeline/sankey_strictly_isolated.html"
fig.write_html(output_path)
print(f"✅ Saved to: {output_path}")