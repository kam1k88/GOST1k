import httpx, asyncio
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GOST1k API Gateway")

CHROMA_URL = "http://chroma:8000"
EMBEDDER_URL = "http://embedder:7000"
RERANKER_URL = "http://reranker:7001"
OLLAMA_URL = "http://llm:11434"

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query(req: QueryRequest):
    async with httpx.AsyncClient(timeout=180) as client:
        # 1. Эмбеддинг запроса
        emb = await client.post(f"{EMBEDDER_URL}/embed", json={"texts": [f"query: {req.query}"]})
        query_vec = emb.json()["embeddings"][0]

        # 2. Поиск кандидатов в Chroma
        chroma_res = await client.post(f"{CHROMA_URL}/api/v1/query", json={
            "query_embeddings": [query_vec],
            "n_results": 20
        })
        docs = [d for d in chroma_res.json()["documents"][0]]

        # 3. Реранкинг
        rerank_res = await client.post(f"{RERANKER_URL}/rerank", json={
            "query": req.query,
            "docs": docs,
            "top_n": 5
        })
        top_docs = [r["text"] for r in rerank_res.json()["results"]]

        # 4. Генерация ответа через LLM (Ollama)
        ctx = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(top_docs))
        payload = {
            "model": "qwen2.5:7b-instruct-q4_K_M",
            "prompt": f"Вопрос: {req.query}\n\nКонтекст:\n{ctx}",
            "stream": False
        }
        llm_res = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        answer = llm_res.json()["response"]

        return {"answer": answer, "context": top_docs}

@app.post("/rebuild")
async def rebuild():
    import shutil
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    os.system("python ingest.py")
    return {"status": "ok", "message": "Полная пересборка завершена"}

@app.get("/health")
def health(): return {"status": "ok"}
