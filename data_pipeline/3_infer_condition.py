import pandas as pd
import re
import json
import itertools
from pathlib import Path
from datetime import datetime
import requests
import os

# ========== CONFIGURATION ==========
OLLAMA_URL = 'http://localhost:11434/api/generate'  # Use /api/generate for streaming
OLLAMA_MODEL = 'gpt-oss:20b'  # Change to your preferred local model
POWERSET_CSV = '/scratch/cjpenni/departmental-honors/data_pipeline/power_set/powerset_by_year/allActivity_2024.csv'
OUTPUT_FOLDER = '/scratch/cjpenni/departmental-honors/data_pipeline/inference_data/06OCT2025'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def count_tokens(text):
    return len(text.split())  # Fallback: rough estimate

def truncate_list(data_list, max_tokens):
    truncated_list = []
    token_count = 0
    for item in data_list:
        item_tokens = count_tokens(str(item))
        if token_count + item_tokens <= max_tokens:
            truncated_list.append(item)
            token_count += item_tokens
        else:
            break
    return truncated_list

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

def clean_json_string(json_string):
    cleaned = re.sub(r"```json|```", " ", json_string)
    cleaned = re.sub(r"[^\u0000-\uFFFF]", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse JSON, returning original: {json_string[:50]}...")
        return cleaned

def write_json(output_filename, inference_json):
    if not inference_json:
        print("⚠️ No data to write. JSON file not created.")
        return None
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(inference_json, f, indent=4, ensure_ascii=False)
    print(f"✅ JSON saved successfully as {output_filename}")
    return output_filename

def build_prompts_for_user_combinations(
    powerSubset2023,
    inference_no,
    rows,
    combo_sizes,
    final_rec_no=3,
    folder="outputs"
):
    if isinstance(combo_sizes, int):
        combo_sizes = [combo_sizes]
    col_names = list(powerSubset2023.columns)
    inference_json_list = []
    for combo_size in combo_sizes:
        if combo_size < 1 or combo_size > len(col_names):
            print(f"⚠️ Skipping invalid combination size: {combo_size}")
            continue
        all_combos = list(itertools.combinations(col_names, combo_size))
        for selected_cols in all_combos:
            columns_str = " AND ".join(selected_cols)
            col_data_list = [powerSubset2023[:rows][col] for col in selected_cols]
            col_data_str = " AND ".join(
                f"{col_data.dropna().tolist()} in {col_name}"
                for col_data, col_name in zip(col_data_list, selected_cols)
            )
            prompt = f"""
            User wants to know what {col_data_str} tells about them.

            1. Generate {inference_no} inferences about the user based on their online behavior, in the form "interested in X".
            2. For each inference, rate:
            - Uncommonness (1–10): How rare this interest is among typical users.
            - Sensitivity (1–10): How sensitive or private this inference is.
            3. Then, using ALL {inference_no} inferences together, recommend EXACTLY {final_rec_no} products that best align with the user's inferred interests.
            4. Each of the {final_rec_no} recommendations must include a 'title' and a 'description'.
            5. Follow the STRICT JSON OUTPUT FORMAT below (no extra text, comments, or explanations outside JSON).

            ```json
            {{
                "columns": "{columns_str}",
                "inferences": [
                    {', '.join([json.dumps({
                        "inference": f"<output for inference {i+1}>",
                        "uncommonness": "<1–10>",
                        "sensitivity": "<1–10>",
                        "explanation": f"<explanation for inference {i+1}>"
                    }) for i in range(inference_no)])}
                ],
                "final_product_recommendations": [
                    {{
                        "title": "<Product 1 title>",
                        "description": "<Product 1 description>"
                    }},
                    ... for {final_rec_no} products ...
                ]
            }}
            ```
            """

            # prompt = f"""
            # User wants to know what {col_data_str} tells about them.
            # 1. {inference_no} inferences about the user based on their online behavior in the form "interested in X".
            # 2. Rate uncommonness (1–10) and sensitivity (1–10) for each inference.
            # 3. Recommend EXACTLY ONE product for each group.
            # 4. Recommend EXACTLY ONE product based on all {inference_no} inferences for {columns_str}.
            #  **STRICT JSON OUTPUT FORMAT (No extra text or explanations, just JSON):**
            # ```json
            # {{
            #     "columns": "{columns_str}",
            #     "inferences": [
            #     {', '.join([json.dumps({
            #     "inference": f"<output for inference {i+1}>",
            #     "explanation_inference": f"<explanation for inference {i+1}>",
            #     "recommendation": f"<recommendation for inference {i+1}>"
            #     }) for i in range(inference_no)])}
            #     ],
            #     "final_product_recommendation": {{
            #     "ONE recommendation based on ALL inferences": "<Product name and company>",
            #     }}
            # }}
            # ```
            # """
            # prompt = f"""
            # You are a precise JSON-generating AI. You must output ONLY valid JSON — no explanations, Markdown formatting, code fences, or commentary. 
            # If your response includes anything other than valid JSON, it will be rejected.

            # The user wants to know what {col_data_str} tells about them.

            # Follow these instructions carefully (do not include them in the output):
            # 1. Generate {inference_no} inferences about the user based on their online behavior in the form "interested in X".
            # 2. For each inference, include:
            # - "inference": the insight (in the form "interested in X"),
            # - "explanation_inference": a one-sentence explanation of why you made this inference,
            # - "recommendation": EXACTLY ONE relevant product recommendation (include product name and company),
            # - "uncommonness": integer 1–10 (10 = most uncommon),
            # - "sensitivity": integer 1–10 (10 = most sensitive).
            # 3. After creating all inferences, analyze them *collectively* and recommend EXACTLY THREE new products that fit the user’s overall interests and behavior pattern.
            # - These final recommendations must NOT be copies of any previous recommendations.
            # - Each must include both product name and company.

            # Your entire response MUST strictly match this JSON structure (no code fences, no text before or after):

            # {{
            #     "columns": "{columns_str}",
            #     "inferences": [
            #         {{
            #             "inference": "<output for inference 1>",
            #             "explanation_inference": "<explanation for inference 1>",
            #             "recommendation": "<Product name and company>",
            #             "uncommonness": <1–10>,
            #             "sensitivity": <1–10>
            #         }},
            #         {{
            #             "inference": "<output for inference 2>",
            #             "explanation_inference": "<explanation for inference 2>",
            #             "recommendation": "<Product name and company>",
            #             "uncommonness": <1–10>,
            #             "sensitivity": <1–10>
            #         }}
            #         ...
            #     ],
            #     "final_product_recommendations": [
            #         "<FIRST new product name and company>",
            #         "<SECOND new product name and company>",
            #         "<THIRD new product name and company>"
            #     ]
            # }}
            # """
            # prompt = f"""
            # You are a precise JSON-generating AI. You must output ONLY valid JSON — no explanations, Markdown formatting, code fences, or commentary. 
            # If your response includes anything other than valid JSON, it will be rejected.

            # The user wants to know what {col_data_str} tells about them.

            # Follow these instructions carefully (do not include them in the output):
            # 1. Generate {inference_no} inferences about the user based on their online behavior in the form "interested in X".
            # 2. For each inference, include:
            # - "inference": the insight (in the form "interested in X"),
            # - "explanation_inference": a one-sentence explanation of why you made this inference,
            # - "recommendations": a list of EXACTLY THREE relevant product recommendations (each with product name and company),
            # - "uncommonness": integer 1–10 (10 = most uncommon),
            # - "sensitivity": integer 1–10 (10 = most sensitive).
            # 3. Do NOT include any final or overall recommendations — only recommendations tied to each inference.

            # Your entire response MUST strictly match this JSON structure (no code fences, no text before or after):

            # {{
            #     "columns": "{columns_str}",
            #     "inferences": [
            #         {{
            #             "inference": "<output for inference 1>",
            #             "explanation_inference": "<explanation for inference 1>",
            #             "recommendations": [
            #                 "<FIRST product name and company>",
            #                 "<SECOND product name and company>",
            #                 "<THIRD product name and company>"
            #             ],
            #             "uncommonness": <1–10>,
            #             "sensitivity": <1–10>
            #         }},
            #         {{
            #             "inference": "<output for inference 2>",
            #             "explanation_inference": "<explanation for inference 2>",
            #             "recommendations": [
            #                 "<FIRST product name and company>",
            #                 "<SECOND product name and company>",
            #                 "<THIRD product name and company>"
            #             ],
            #             "uncommonness": <1–10>,
            #             "sensitivity": <1–10>
            #         }}
            #         ...
            #     ]
            # }}
            # """
            print(f"🔹 Sending prompt to Ollama for columns: {columns_str}...")
            assistant_reply = run_prompt(prompt)
            inference_json_list.append({
                "combined_cols": selected_cols,
                "combo_size": combo_size,
                "gpt_output": assistant_reply
            })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"new_inferences_{timestamp}.json"
    filepath = Path(folder) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return inference_json_list, filepath

def main():
    # Load powerset data
    powerset2024 = pd.read_csv(POWERSET_CSV)

    # Select columns with less than 600 non-NaN rows
    misc_columns = powerset2024.columns[powerset2024.notna().sum() < 600]
    powerset2024['Misc_Search Title'] = powerset2024[misc_columns].apply(
        lambda row: ', '.join(row.dropna().astype(str)),
        axis=1)
    powerset2024 = powerset2024.drop(columns=misc_columns)
    powerset2024 = powerset2024.drop_duplicates(subset=['Chrome_Search Title'], keep='first')
    powerset2024 = powerset2024.rename(columns={
        # "takeout1_imageSearch_MyActivity_Search Title": "Image_Search",
        # "takeout1_chrome_MyActivity_Search Title": "Browser_history",
        # "takeout1_maps_MyActivity_Search Title": "Location_history",
        # "takeout1_YT_search-history_Search Title": "YT_search_history",
        # "takeout1_YT_watch-history_Search Title": "YT_watch_history",
        # "takeout1_misc_MyActivity_Search Title": "Misc"
        "Chrome_Search Title": "Browser_history",
        "Maps_Search Title": "Location_history",
        "Search_Search Title": "Google_search_history",
        "YouTube_Search Title": "YT_search_history",
    })

    # Ask user to select four columns
    print("\nPlease select four datasets from the options (datasets should not be the same):")
    for i, col in enumerate(powerset2024.columns, 1):
        print(f"{i}. {col}")
    selected_columns = []
    while len(selected_columns) < 4:
        try:
            col_num = int(input(f"\nSelect column {len(selected_columns) + 1} by number (1-{len(powerset2024.columns)}): "))
            if 1 <= col_num <= len(powerset2024.columns):
                col_name = powerset2024.columns[col_num - 1]
                if col_name not in selected_columns:
                    selected_columns.append(col_name)
                else:
                    print("You've already selected this column. Try another.")
            else:
                print("Invalid number. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    powerSubset2023 = powerset2024[selected_columns]
    print("\nSelected DataFrame:")
    print(powerSubset2023.head())

    # Optionally, save to a CSV file
    save_option = input("\nWould you like to save this selection to a CSV file? (yes/no): ").strip().lower()
    if save_option == "yes":
        output_csv = os.path.join(OUTPUT_FOLDER, "chrome_maps_YTwatch_misc.csv")
        powerSubset2023.to_csv(output_csv, index=False)
        print(f"File saved as '{output_csv}'.")

    # Ask user for number of inferences
    options = {'a': 3, 'b': 6, 'c': 9, 'd': 12}
    print("How many inferences would you like to generate per grouping?")
    print("Please select one of the following options:\n")
    for key, value in options.items():
        print(f"  {key}) {value} inferences")

    user_input = input("\nEnter your selection (a–d): ").lower().strip()

    if user_input in options:
        inference_no = options[user_input]
        print(f"\nYou selected: {inference_no} inferences.")
    else:
        print("\nInvalid selection. Please enter a letter between a and d.")
        return
    
    # Ask user for number of final product recommendations
    rec_options = {'a': 1, 'b': 2, 'c': 3, 'd': 5}
    print("\nHow many final product recommendations should the model provide?")
    print("Please select one of the following options:\n")
    for key, value in rec_options.items():
        print(f"  {key}) {value} recommendation{'s' if value != 1 else ''}")

    rec_input = input("\nEnter your selection (a–d): ").lower().strip()

    if rec_input in rec_options:
        final_rec_no = rec_options[rec_input]
        print(f"\nYou selected: {final_rec_no} final recommendation{'s' if final_rec_no != 1 else ''}.")
    else:
        print("\nInvalid selection for recommendations. Please enter a letter between a and d.")
        return

    # Build prompts and get GPT output
    inference_json, output_file = build_prompts_for_user_combinations(
        powerSubset2023,
        inference_no=inference_no,
        # Play with number of rows to see trade-off.
        # use all rows, not just 200, for best results
        rows=powerSubset2023.shape[0],
        combo_sizes=[1, 2, 3, 4],
        final_rec_no=final_rec_no,
        folder=OUTPUT_FOLDER
    )

    # Clean GPT outputs
    for entry in inference_json:
        if isinstance(entry.get("gpt_output"), str):
            entry["gpt_output"] = clean_json_string(entry["gpt_output"])

    # Write to JSON
    write_json(output_file, inference_json)

if __name__ == "__main__":
    main()