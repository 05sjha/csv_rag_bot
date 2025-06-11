import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

def query_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        raise RuntimeError(f"Ollama API error: {response.text}")
