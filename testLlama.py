import requests
import json

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.1:8b",  # must match the model you pulled
    "prompt": "Give me a one-sentence summary of general relativity."
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode("utf-8"))
        if "response" in data:
            print(data["response"], end="", flush=True)
