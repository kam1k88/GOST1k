import os
import chromadb
from sentence_transformers import SentenceTransformer

# === Настройки ===
CHROMA_DIR = r"C:\Users\kam1k88\GOST1k\chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 3  # количество ближайших результатов

# Отключаем телеметрию Chroma
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# === Инициализация модели и клиента ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("gost1k")

print(f"[✓] Подключено к Chroma DB: {CHROMA_DIR}")
print(f"[✓] Коллекция: gost1k (существует или создана)\n")

# === Поисковая функция ===
def search(query: str, top_k: int = TOP_K, streamlit_output=False):
    print(f"\n🔍 Запрос: {query}")

    # Генерация эмбеддинга запроса
    q_emb = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Преобразуем NumPy → list
    q_emb = q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb

    # Запрос к Chroma (возвращаем векторы, а не расстояния)
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "embeddings"]
    )

    # === Консольный вывод ===
    for i, (doc, meta, emb) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]),
        1
    ):
        snippet = doc[:200].replace("\n", " ")
        emb_preview = ", ".join([f"{x:.4f}" for x in emb[:6]]) + " ..."
        print(f"\n#{i}. {meta.get('source', '—')}")
        print(f"   Вектор: [{emb_preview}]")
        print(f"   Текст: {snippet}...")

    # === Streamlit-вывод (если используется) ===
    if streamlit_output:
        import streamlit as st
        st.subheader("Результаты поиска:")
        for i, (doc, meta, emb) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]),
            1
        ):
            emb_preview = ", ".join([f"{x:.4f}" for x in emb[:6]]) + " ..."
            st.markdown(f"**{i}. {meta.get('source', '—')}**")
            st.caption(f"Вектор: [{emb_preview}]")
            st.write(doc[:600] + "...")

# === Пример ручного запуска ===
if __name__ == "__main__":
    search("требования к защите информации")
