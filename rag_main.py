import sys; sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import asyncio
import httpx
import numpy as np
import time
from datetime import datetime
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
import chromadb
import torch

# === Общие настройки ===
os.environ["CHROMA_TELEMETRY"] = "False"
load_dotenv()

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
embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=DEVICE)
reranker = CrossEncoder("BAAI/bge-reranker-base", device=DEVICE)

# === Прогрев GPU ===
_ = embedder.encode(["warmup"], batch_size=1, max_length=128)
torch.cuda.empty_cache()

# === Инициализация клиента Chroma ===
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=None)

# === Утилиты ===
def clean_text(t):
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def normalize_query(q: str) -> str:
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
            print("[⚠️] Коллекция пуста.")
            return
        doc = peek["documents"][0][0] if isinstance(peek["documents"][0], list) else peek["documents"][0]
        meta = peek["metadatas"][0][0] if isinstance(peek["metadatas"][0], list) else peek["metadatas"][0]
        print("[📄] Пример документа:\n", clean_text(doc[:300]))
        print("[🗂️] Пример метаданных:\n", meta)
    except Exception as e:
        print(f"[❌] Ошибка при проверке коллекции: {e}")

# === Гибридный поиск (dense + sparse fusion) ===
def hybrid_search(q, top_k=50, alpha=0.65):
    """
    alpha - вес dense. 0.65 оптимально для ГОСТов (семантика важнее, но термины учитываем).
    """
    try:
        t0 = time.time()

        # 1) гибридные эмбеддинги запроса
        #   в FlagEmbedding>=1.3 не нужно указывать normalize_embeddings,
        #   и аргументы должны быть return_dense, return_sparse
        res = embedder.encode([q], return_dense=True, return_sparse=True)
        dense_vec = res["dense_vecs"]
        sparse_weights = res["lexical_weights"][0] if "lexical_weights" in res else {}

        # 2) dense-кандидаты из Chroma
        cres = collection.query(query_embeddings=dense_vec, n_results=top_k * 2)
        docs0 = cres.get("documents", [[]])[0]
        metas0 = cres.get("metadatas", [[]])[0]
        dists0 = cres.get("distances", [[]])[0]

        if not docs0:
            print("[⚠️] Коллекция пуста или нет релевантных фрагментов.")
            return []

        # === dense part ===
        dense_scores = {}
        for m, dist in zip(metas0, dists0):
            sim = 1.0 - float(dist)
            dense_scores[m["source"]] = max(dense_scores.get(m["source"], 0.0), sim)

        # === sparse part ===
        sparse_scores = {}
        for doc, meta in zip(docs0, metas0):
            if not sparse_weights:
                continue
            score = sum(sparse_weights.get(tok, 0.0) for tok in doc.split())
            sparse_scores[meta["source"]] = max(sparse_scores.get(meta["source"], 0.0), score)

        # === fusion ===
        fused = {}
        keys = set(dense_scores.keys()) | set(sparse_scores.keys())
        for k in keys:
            fused[k] = alpha * dense_scores.get(k, 0.0) + (1 - alpha) * sparse_scores.get(k, 0.0)

        # === отладочные топы ===
        if fused:
            top_dense = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_sparse = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print("\n🧠 Top dense:")
            [print(f"  {n}: {s:.3f}") for n, s in top_dense]
            print("🪶 Top sparse:")
            [print(f"  {n}: {s:.3f}") for n, s in top_sparse]

        # === формируем итог ===
        ranked_ids = [k for k, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        docs = [{"text": d, "source": s} for d, s in zip(docs0, metas0) if s["source"] in ranked_ids]

        total_time = time.time() - t0
        log_event(f"[⏱️] BGE-M3 hybrid dense+sparse: {total_time:.2f} сек ({len(docs)} docs)")
        return docs

    except Exception as e:
        print(f"[❌] Ошибка hybrid_search: {e}")
        log_event(f"[❌] Ошибка hybrid_search: {e}")
        return []

# === Реранкинг ===
def rerank_docs(q, docs):
    if not docs:
        return []
    t0 = time.time()
    pairs = [[normalize_query(q), format_doc(d["text"])] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    rerank_time = time.time() - t0
    log_event(f"[⏱️] Rerank: {rerank_time:.2f} сек ({len(docs)} docs)")
    return [r[0] for r in ranked]

# === Генерация через Ollama ===
async def ollama_generate(prompt, model=None):
    torch.cuda.empty_cache()
    model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    async with httpx.AsyncClient(timeout=300.0) as c:
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
            t0 = time.time()
            r = await c.post(f"{ollama_host}/api/generate",
                             json={"model": model, "prompt": prompt, "stream": False})
            data = r.json()
            total_time = time.time() - t0
            log_event(f"[⏱️] Ollama ответ: {total_time:.2f} сек")
            if "response" in data and data["response"].strip():
                return data["response"].strip()
            return data.get("output", "") or str(data)
        except Exception as e:
            return f"[⚠️] Ollama недоступна: {e}"
        finally:
            torch.cuda.empty_cache()

# === Основная логика RAG ===
async def answer(query: str):
    t0 = time.time()
    print(f"[🔍] Запрос: {query}")
    sys_query = normalize_query(query)
    log_event(f"\n[USER QUERY] {query}")
    log_event(f"[SYSTEM QUERY] {sys_query}")

    # === 1. Поиск релевантных документов ===
    docs = hybrid_search(sys_query)
    print(f"[~] Найдено кандидатов: {len(docs)}")

    if not docs:
        msg = "[⚠️] Нет совпадений в коллекции."
        print(msg)
        log_event(msg)
        return msg

    # === 2. Определение намерения (intent detection) ===
    q_lower = query.lower()
    num_docs = len(docs)
    mode = "structured"  # базовый режим по умолчанию

    if any(w in q_lower for w in ["перечисли", "все гост", "все стандарты", "список", "перечень"]) or num_docs > 15:
        mode = "list"
        top_k = min(num_docs, 30)
    elif any(w in q_lower for w in ["требования", "критерии", "условия", "характеристики"]):
        mode = "criteria"
        top_k = min(num_docs, 8)
    elif any(w in q_lower for w in ["письмо", "пример", "текст", "документ", "составь", "напиши"]):
        mode = "long"
        top_k = min(num_docs, 10)
    elif any(w in q_lower for w in ["сравни", "различия", "отличие", "vs"]):
        mode = "compare"
        top_k = min(num_docs, 12)
    elif any(w in q_lower for w in ["анализ", "объясни", "поясни", "связь", "влияние", "как связано"]) or len(q_lower.split()) > 20:
        mode = "analytic"
        top_k = min(num_docs, 15)
    elif any(w in q_lower for w in ["кратко", "резюме", "в двух словах", "итог"]):
        mode = "summary"
        top_k = min(num_docs, 5)
    else:
        mode = "structured"
        top_k = min(num_docs, 7)

    print(f"[⚙️] Режим: {mode}, top_docs={top_k}")

    # === 3. Реранкинг ===
    reranked = rerank_docs(sys_query, docs)
    top_docs = reranked[:top_k]
    context = "\n\n".join([clean_text(d["text"][:800]) for d in top_docs])

    # === 4. Инструкции для LLM ===
    intros = {
        "structured": "Ты — эксперт по информационной безопасности и нормативным документам ТК 362 'Защита информации'. Отвечай кратко, точно и по фактам, с приоритетом ГОСТов.",
        "criteria": "Кратко перечисли основные критерии или требования по теме. Используй маркированный список и укажи ГОСТы, если они есть.",
        "list": "Перечисли все найденные стандарты (ГОСТ, СТО, ISO и т.д.) с их номерами и названиями. Не давай пояснений — только список.",
        "long": "Составь развернутый пример документа, письма или пояснительного текста в официальном стиле, опираясь на стандарты ТК 362.",
        "compare": "Сравни упомянутые стандарты или подходы по ключевым аспектам. Укажи различия и совпадения в формате списка.",
        "analytic": "Проанализируй взаимосвязь и смысл требований, указанных в ГОСТах. Используй рассуждение, но оставайся в нормативных рамках. Приводи номера стандартов, если они есть.",
        "summary": "Кратко подведи итоги по теме, в 2-3 предложениях, без деталей и нумерации."
    }
    intro = intros.get(mode, intros["structured"])

    prompt = f"{intro}\n\nВопрос: {query}\n\nКонтекст:\n{context}\n\nОтвет:"

    # === 5. Генерация через Ollama ===
    ans = await ollama_generate(prompt)
    total_time = time.time() - t0
    print(f"\n[💡] Ответ сгенерирован ({total_time:.2f} сек, режим={mode}):\n{ans}")
    log_event(f"[ANSWER | {mode}] ({total_time:.2f} сек)\n{ans}\n{'='*80}\n")
    return ans


# === CLI ===
if __name__ == "__main__":
    check_collection()
    q = "аудит информационной безопасности и регистрация событий"
    asyncio.run(answer(q))
