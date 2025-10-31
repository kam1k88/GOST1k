import os
from dotenv import load_dotenv
load_dotenv()
import re
import asyncio
import httpx
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import torch

# === Общие настройки ===
os.environ["CHROMA_TELEMETRY"] = "False"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "gost1k"
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "gost1k.log")

# === Логирование ===
def log_event(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

# === Определение устройства ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[⚙️] Используется устройство: {DEVICE.upper()}")

# === Инициализация моделей ===
embedder = SentenceTransformer("intfloat/multilingual-e5-small", device=DEVICE)
reranker = CrossEncoder("BAAI/bge-reranker-base", device=DEVICE)

# === Инициализация клиента Chroma ===
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=None)

# === Утилиты ===
def clean_text(t):
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def normalize_query(q: str) -> str:
    """Добавляет E5-префикс для лучшего семантического поиска"""
    return f"query: {q.strip().lower()}"

def format_doc(d: str) -> str:
    return f"passage: {d.strip()}"


# === Проверка коллекции ===
def check_collection():
    try:
        print("[🧩] Проверка коллекции Chroma...")
        count = collection.count()
        print(f"[📦] Всего фрагментов: {count}")
        peek = collection.peek()
        if not peek or "documents" not in peek or not peek["documents"]:
            print("[⚠️] Коллекция пуста или не содержит документов.")
            return
        doc = peek["documents"][0][0] if isinstance(peek["documents"][0], list) else peek["documents"][0]
        meta = peek["metadatas"][0][0] if isinstance(peek["metadatas"][0], list) else peek["metadatas"][0]
        print("[📄] Пример документа:\n", clean_text(doc[:300]))
        print("[🗂️] Пример метаданных:\n", meta)
    except Exception as e:
        print(f"[❌] Ошибка при проверке коллекции: {e}")


# === Поиск ===
def dense_query(q, top_k=50):
    """Dense-поиск по эмбеддингам E5"""
    try:
        q_emb = embedder.encode([normalize_query(q)], normalize_embeddings=True)
        results = collection.query(query_embeddings=q_emb, n_results=top_k)
        if not results.get("documents") or not results["documents"][0]:
            print("[⚠️] Пустая коллекция или нет совпадений.")
            return []
        docs = [
            {"text": d, "source": s}
            for d, s in zip(results["documents"][0], results["metadatas"][0])
        ]
        return docs
    except Exception as e:
        print(f"[❌] Ошибка dense_query: {e}")
        return []


# === Реранкинг ===
def rerank_docs(q, docs):
    if not docs:
        return []
    pairs = [[normalize_query(q), format_doc(d["text"])] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked]


# === Генерация через Ollama ===
async def ollama_generate(prompt, model="qwen2.5:7b-instruct-q4_K_M"):
    torch.cuda.empty_cache()
    async with httpx.AsyncClient(timeout=300.0) as c:
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
            r = await c.post(
                f"{ollama_host}/api/generate",
		json={"model": model, "prompt": prompt, "stream": False},
            )
            data = r.json()
            if "response" in data and data["response"].strip():
                return data["response"].strip()
            if "message" in data and isinstance(data["message"], dict):
                return data["message"].get("content", "").strip()
            text = data.get("output", "") or str(data)
            if "CUDA error" in text:
                print("[⚠️] CUDA error — fallback → Qwen2:1.5b-instruct")
                torch.cuda.empty_cache()
                return await ollama_generate(prompt, "Qwen2:1.5b-instruct")
            return text.strip() if text else "[⚠️] Пустой ответ от Ollama"
        except Exception as e:
            if "CUDA" in str(e):
                print("[⚠️] Ошибка CUDA — fallback → Qwen2:1.5b-instruct")
                torch.cuda.empty_cache()
                return await ollama_generate(prompt, "Qwen2:1.5b-instruct")
            return f"[⚠️] Ollama недоступна: {e}"
        finally:
            torch.cuda.empty_cache()


# === Основная логика RAG ===
async def answer(query: str):
    print(f"[🔍] Запрос: {query}")
    sys_query = normalize_query(query)
    print(f"[🧠] Сформировано: {sys_query}")
    log_event(f"\n[USER QUERY] {query}")
    log_event(f"[SYSTEM QUERY] {sys_query}")

    docs = dense_query(sys_query)
    print(f"[~] Найдено кандидатов: {len(docs)}")
    log_event(f"[CHROMA] Найдено кандидатов: {len(docs)}")

    if not docs:
        msg = "[⚠️] Нет совпадений в коллекции."
        print(msg)
        log_event(msg)
        return msg

    reranked = rerank_docs(sys_query, docs)
    top_docs = reranked[:5]

    print("\n[🏆] Топ-5 фрагментов:")
    for i, d in enumerate(top_docs, 1):
        snippet = clean_text(d["text"][:150])
        src = d.get("source", "?")
        print(f"{i:>2}. {snippet} ...\n   [source={src}]")
        log_event(f"[DOC {i}] {snippet} | source={src}")

    context = "\n\n".join([clean_text(d["text"][:800]) for d in top_docs])

    intro = (
        "Ты — эксперт по информационной безопасности. "
        "Отвечай **только на русском языке**, строго в контексте нормативных документов ТК 362 'Защита информации'. "
        "Опирайся на требования и рекомендации ГОСТ, СТО, РД. "
        "Формулируй ответ кратко и по существу, без перевода или повторов на других языках."
    )

    wc = len(query.split())
    mode = (
        "reflective" if wc <= 3
        else "structured" if wc <= 15
        else "deep_analytic"
    )
    print(f"[🧩] Mode: {mode}")

    prompt = f"""{intro}

Вопрос: {query}

Контекст:
{context}

Ответ:"""

    ans = await ollama_generate(prompt)
    print(f"\n[💡] Ответ сгенерирован:\n{ans}")
    log_event(f"[ANSWER | {mode}] {ans}\n{'='*80}\n")

    torch.cuda.empty_cache()
    return ans


def process_query(query: str, mode: str = "structured"):
    """
    Универсальная обёртка для вызова RAG из других модулей (API, UI).
    """
    try:
        # Проверка на пустые запросы
        if not query or not query.strip():
            return "[⚠️] Пустой запрос"

        # Асинхронный запуск
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(answer(query))
        loop.close()

        return result
    except Exception as e:
        return f"[❌] Ошибка process_query: {e}"

# === CLI ===
if __name__ == "__main__":
    check_collection()
    q = "аудит информационной безопасности и регистрация событий"
    asyncio.run(answer(q))
