import sys
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

import os, re, time, asyncio, httpx, torch, chromadb
from datetime import datetime
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

# === 1. Базовая настройка ===
load_dotenv()
BASE = os.path.dirname(os.path.abspath(__file__))
CHROMA = os.path.join(BASE, "chroma_db")
LOG = os.path.join(BASE, "logs", "gost1k.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[⚙️] Устройство: {DEVICE}")

# === 2. Модели ===
embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=DEVICE)
reranker = CrossEncoder("BAAI/bge-reranker-base", device=DEVICE)

# === 3. Warmup ===
print("[*] Warmup...")
_ = embedder.encode(["warmup"], batch_size=1, max_length=128)
torch.cuda.synchronize()
torch.cuda.empty_cache()
print("[✅] Модели готовы\n")


# === Асинхронный гибридный поиск (RRF fusion) ===
async def hybrid_rrf(query, topk=40, k=60):
    """
    Reciprocal Rank Fusion (RRF) объединяет dense и sparse результаты.
    Асинхронно, с GPU reuse и warmup.
    """
    t0 = time.time()
    query_norm = f"query: {query.lower().strip()}"

    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, lambda: embedder.encode(
        [query_norm], return_dense=True, return_sparse=True
    ))

    dense_vec = res["dense_vecs"]
    sparse_weights = res.get("lexical_weights", [{}])[0]

    client = chromadb.PersistentClient(path=CHROMA)
    coll = client.get_or_create_collection("gost1k")
    cres = await loop.run_in_executor(None, lambda: coll.query(
        query_embeddings=dense_vec, n_results=topk * 2
    ))

    docs = cres["documents"][0]
    metas = cres["metadatas"][0]
    dists = cres["distances"][0]

    if not docs:
        return []

    dense_rank = {m["source"]: i for i, m in enumerate(
        [meta for _, meta in sorted(zip(dists, metas), key=lambda x: x[0])]
    )}

    sparse_rank = {}
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        score = sum(sparse_weights.get(tok, 0) for tok in doc.split())
        sparse_rank[meta["source"]] = i if score > 0 else topk * 2

    fused = {}
    all_docs = set(dense_rank.keys()) | set(sparse_rank.keys())
    for d in all_docs:
        r1 = dense_rank.get(d, topk * 2)
        r2 = sparse_rank.get(d, topk * 2)
        fused[d] = 1 / (k + r1) + 1 / (k + r2)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    ids = [k for k, _ in ranked[:topk]]

    results = [{"text": docs[i], "meta": metas[i]} for i, m in enumerate(metas) if m["source"] in ids]

    log(f"[⏱️] RRF fusion: {time.time() - t0:.2f} сек ({len(results)} docs)")
    return results


# === Асинхронный реранкер ===
async def rerank_async(query, docs, top_k=10):
    if not docs:
        return []
    t0 = time.time()
    loop = asyncio.get_event_loop()

    def _predict():
        pairs = [[f"query: {query}", d["text"]] for d in docs]
        scores = reranker.predict(pairs)
        return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    ranked = await loop.run_in_executor(None, _predict)
    top = [r[0] for r in ranked[:top_k]]
    log(f"[⏱️] Rerank: {time.time() - t0:.2f} сек ({len(top)} docs)")
    return top


# === Асинхронная генерация через Ollama ===
async def ollama_generate(prompt):
    url = "http://127.0.0.1:11501/api/generate"
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")

    async with httpx.AsyncClient(timeout=300.0) as c:
        try:
            r = await c.post(url, json={"model": model, "prompt": prompt, "stream": False})
            return r.json().get("response", "").strip()
        except Exception as e:
            return f"[⚠️] Ollama ошибка: {e}"


# --- утилита очистки текста ---
def clean(text: str) -> str:
    """Простая очистка текста перед генерацией"""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "").strip()
    return text

async def answer(query):
    t0 = time.time()
    print(f"[🔍] Запрос: {query}")
    log(f"[USER] {query}")

    docs = await hybrid_rrf(query, topk=50)
    if not docs:
        msg = "⚠️ Нет совпадений в коллекции."
        log(msg)
        return msg

    # rerank 10
    top_docs = await rerank_async(query, docs, top_k=10)
    context = "\n\n".join(clean(d["text"][:800]) for d in top_docs)

    intro = (
        "Ты эксперт по информационной безопасности и стандартам ТК 362. Отвечай официально на русском языке."
    )

    prompt = f"{intro}\n\nВопрос: {query}\n\nКонтекст:\n{context}\n\nОтвет:"

    ans = await ollama_generate(prompt)
    total = time.time() - t0
    log(f"[ANS] ({total:.2f} сек)\n{ans}\n{'='*80}")
    print(f"[💡] {ans}")
    return ans
