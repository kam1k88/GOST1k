import json, os

# === Пути к файлам ===
RETRIEVE_FILE = "logs/gost_eval.jsonl"        # тут answer = retrieved
ANSWER_FILE   = "logs/gost_eval_self.jsonl"   # тут answer = реальный ответ LLM
OUT_FILE      = "logs/gost_eval_ready.jsonl"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

# === Загрузка данных ===
print("[🔍] Загружаем данные...")
retrieve_data = load_jsonl(RETRIEVE_FILE)
answer_data = load_jsonl(ANSWER_FILE)

# === Индексация по query ===
answers_by_query = {a["query"].strip(): a for a in answer_data}

merged = []
for idx, r in enumerate(retrieve_data, 1):
    q = r.get("query", "").strip()
    if not q:
        continue
    a = answers_by_query.get(q)
    if not a:
        continue
    merged.append({
        "id": idx,
        "query": q,
        "retrieved": r.get("answer", ""),   # из файла, где answer = retrieved
        "answer": a.get("answer", "")       # из файла, где answer = ответ модели
    })

# === Сортировка (по id или query) ===
merged.sort(key=lambda x: x["id"])

# === Сохранение ===
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for m in merged:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

# === Проверка и сводка ===
print(f"[✅] Объединено {len(merged)} записей")
print(f"[💾] Сохранено в {OUT_FILE}")

# Пример для контроля
if merged:
    print("\nПример объединения:")
    for m in merged[:2]:
        print(json.dumps(m, ensure_ascii=False, indent=2)[:600], "\n---")
