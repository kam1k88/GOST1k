import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils.text_loader import load_documents

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHROMA_DIR = "chroma_db"
DOCS_DIR = "docs"

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    device="cuda",
    normalize_embeddings=True
)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("gost1k", embedding_function=embedding_fn)

def ingest():
    print("[~] Загружаем документы из:", DOCS_DIR)
    docs = load_documents(DOCS_DIR)
    print(f"[✓] Загружено {len(docs)} документов. Создаём векторный индекс...")

    for i, doc in enumerate(docs):
        collection.add(  # ChromaDB добавляет батчом
            documents=["passage: " + doc["text"]],
            metadatas=[{"source": doc["source"]}],
            ids=[f"doc_{i:04d}"]
        )

    print("[✓] Индексация завершена. База сохранена в chroma_db/")

if __name__ == "__main__":
    ingest()