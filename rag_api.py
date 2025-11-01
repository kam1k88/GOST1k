import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# импортируем корректную функцию из rag_main
from rag_main import answer as rag_answer  # async def answer(query: str)

load_dotenv()

app = FastAPI(title="GOST1k RAG API", version="1.0")

class QueryRequest(BaseModel):
    query: str
    mode: str = "structured"  # оставляем для совместимости UI, но пока не используем

@app.post("/api/query")
async def run_rag(request: QueryRequest):
    # rag_answer уже async, поэтому просто await
    result = await rag_answer(request.query)
    return {"query": request.query, "result": result}

@app.get("/api/health")
async def health():
    return {"status": "ok", "ollama_host": os.getenv("OLLAMA_HOST", "not set")}

if __name__ == "__main__":
    # 0.0.0.0 - чтобы было доступно и из WSL, и из Windows
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
