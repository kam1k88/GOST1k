import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from rag_main import process_query  # импорт из твоего основного модуля

load_dotenv()

app = FastAPI(title="GOST1k RAG API", version="1.0")

class QueryRequest(BaseModel):
    query: str
    mode: str = "structured"

@app.post("/api/query")
async def run_rag(request: QueryRequest):
    result = await asyncio.to_thread(process_query, request.query, request.mode)
    return {"query": request.query, "result": result}

@app.get("/api/health")
async def health():
    return {"status": "ok", "ollama_host": os.getenv("OLLAMA_HOST", "not set")}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
