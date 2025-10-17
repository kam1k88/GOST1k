import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

payload = {
    "model": "Qwen2:7B-Instruct",
    "prompt": "<|system|>Ты ассистент. <|user|>Привет!",
    "stream": False
}

try:
    r = requests.post(OLLAMA_URL, json=payload, timeout=10)
    r.raise_for_status()
    response = r.json().get("response", "")
    print("[✓] Модель Qwen2 работает!")
    print("Ответ:", response.strip())
except Exception as e:
    print("[!] Ollama или модель Qwen2:7B-Instruct недоступна.")
    print("Ошибка:", e)
