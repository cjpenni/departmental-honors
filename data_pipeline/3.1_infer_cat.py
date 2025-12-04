import pandas as pd
import numpy as np
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
FILE_PATH = '/scratch/cjpenni/departmental-honors/data_pipeline/inference_data/06OCT2025/new_inferences_rag.json'
# OUTPUT_FOLDER = '/scratch/cjpenni/departmental-honors/data_pipeline/inference_data/06OCT2025'
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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


def extract_inference_details(inference_text):
    """
    Use GPT to split an inference into category, activity, and reason.
    Falls back to default parsing if model output isn't valid JSON.
    """

    prompt = f"""
    You are given an inference about a user's behavior or interest.
    Extract and return a JSON object with the following keys:
    - category: a short theme (e.g., 'Health & Fitness', 'Travel', 'Cooking')
    - activity: what the user is doing or interested in
    - reason: the evidence or rationale given in the sentence

    Example:
    Input: "The user is interested in health and fitness, as indicated by their search for exercises like 'leg raises' and 'calf raises', as well as their queries about tendonitis and Pilates."
    Output: {{
      "category": "Health & Fitness",
      "activity": "Exercise Routine like Pilates",
      "reason": "User searched for exercises such as leg raises and calf raises, and looked up information on tendonitis and Pilates."
    }}

    Now extract for this input:
    "{inference_text}"
    Return ONLY valid JSON.
    """

    reply = run_prompt(prompt)  # Your existing GPT call
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback default structure if parsing fails
    return {
        "category": "Other",
        "activity": inference_text.split(",")[0].replace("The user", "User").strip(),
        "reason": inference_text
    }

def add_inference_details(data):
    for entry in data:
        gpt_output = entry.get("gpt_output", {})

        if isinstance(gpt_output, str):
            try:
                gpt_output = json.loads(gpt_output)
            except json.JSONDecodeError:
                gpt_output = {}

        inferences = gpt_output.get("inferences", [])

        for inf in inferences:
            details = extract_inference_details(inf["inference"])
            inf.update(details)

    return data


with open(FILE_PATH, 'r') as f:
    data = json.load(f)

# --- Extract all inferences ---
all_inferences = []
for entry in data:
    gpt_output = entry.get("gpt_output", {})

    # Handle stringified JSON
    if isinstance(gpt_output, str):
        try:
            gpt_output = json.loads(gpt_output)
        except json.JSONDecodeError:
            gpt_output = {}

    inferences = gpt_output.get("inferences", [])


updated_data = add_inference_details(data)

with open("inferences_with_cat.json", "w") as f:
    json.dump(updated_data, f, indent=2)