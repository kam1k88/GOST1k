import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import torch

# === Настройки ===
MODEL_EMBEDDING = "intfloat/multilingual-e5-small"
MODEL_RERANK = "BAAI/bge-reranker-base"
CHROMA_DIR = "chroma_db"
TOP_K = 20
TOP_N = 5

# === Инициализация моделей ===
retriever_model = SentenceTransformer(MODEL_EMBEDDING, device="cuda")
reranker = CrossEncoder(MODEL_RERANK, device="cuda")
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name=MODEL_EMBEDDING,
    device="cuda",
    normalize_embeddings=True
)

# === Новый клиент ChromaDB ===
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("gost1k", embedding_function=embedding_fn)

# === Поиск и реранк ===
def search(query_text, streamlit_output=False):
    if not streamlit_output:
        print(f"[?] Запрос: {query_text}\n")

    query = "query: " + query_text.strip()
    q_emb = retriever_model.encode(query, normalize_embeddings=True)

    results = collection.query(query_embeddings=[q_emb], n_results=TOP_K, include=["documents", "metadatas", "ids"])
    passages = results["documents"][0]
    metadatas = results["metadatas"][0]

    pairs = [(query_text, passage.replace("passage: ", "")) for passage in passages]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(scores, passages, metadatas), key=lambda x: x[0], reverse=True)[:TOP_N]

    if streamlit_output:
        import streamlit as st
        for i, (score, passage, meta) in enumerate(reranked, 1):
            st.markdown(f"**[{i}]** ({score:.4f}) `{meta['source']}`")
            st.write(passage.strip())
            st.markdown("---")
    else:
        for i, (score, passage, meta) in enumerate(reranked, 1):
            print(f"[{i}] ({score:.4f}) {meta['source']} :: {passage[:200]}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]))
    else:
        search("какие требования к защите информации")
