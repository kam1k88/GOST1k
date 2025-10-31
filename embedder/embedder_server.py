from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI(title="GOST1k Embedder")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)

class EmbedRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
def embed(req: EmbedRequest):
    vectors = model.encode(req.texts, normalize_embeddings=True, convert_to_numpy=True)
    return {"embeddings": vectors.tolist()}

@app.get("/health")
def health(): return {"status": "ok"}
