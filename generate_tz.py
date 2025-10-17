import os
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
SYSTEM_PROMPT = """
Ты — инженер по стандартизации. На основе темы составь структуру технического задания (ТЗ), опираясь на ГОСТ, СТО и регламенты.
Выдели основные разделы, кратко опиши, что в них должно быть. Формат — JSON со списком разделов.
Не выдумывай, если тема вне области ГОСТ — скажи об этом прямо.
"""


def generate(topic: str):
    payload = {
        "model": "Qwen2:7B-Instruct",
        "stream": False,
        "prompt": f"<|system|>{SYSTEM_PROMPT}<|user|>Тема: {topic}\nОтвет:",
        "options": {"temperature": 0.4}
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code != 200:
        print("[!] Ошибка запроса к Ollama")
        return

    text = response.json().get("response", "").strip()
    print("\n[✓] Сгенерировано ТЗ:\n")
    print(text)
    try:
        data = json.loads(text)
        with open("tz_output.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("\n[Сохранено] Результат записан в tz_output.json")
    except Exception:
        print("[!] Ответ не в формате JSON, вывод только в консоль.")


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "защита информации в СКЗИ"
    generate(topic)
