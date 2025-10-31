from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI(title="GOST1k Reranker")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-reranker-base", device=device)

class RerankRequest(BaseModel):
    query: str
    docs: list[str]
    top_n: int = 5

@app.post("/rerank")
def rerank(req: RerankRequest):
    pairs = [[req.query, d] for d in req.docs]
    scores = model.similarity(pairs)
    ranked = sorted(zip(req.docs, scores), key=lambda x: float(x[1]), reverse=True)[:req.top_n]
    return {"results": [{"text": t, "score": float(s)} for t, s in ranked]}

@app.get("/health")
def health(): return {"status": "ok"}
