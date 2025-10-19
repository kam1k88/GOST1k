import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# === Настройки ===
CHROMA_DIR = r"C:\Users\kam1k88\GOST1k\chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "Qwen2:7B-Instruct"  # имя модели в Ollama
TOP_K = 8  # сколько документов подавать в контекст

# Отключаем телеметрию Chroma
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# === Инициализация моделей ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client_chroma.get_or_create_collection("gost1k")

# Подключаем локальный Ollama API (OpenAI-совместимый)
client_llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

print(f"[✓] Подключено к Chroma DB: {CHROMA_DIR}")
print(f"[✓] Коллекция: gost1k (существует или создана)")
print(f"[✓] Модель LLM: {LLM_MODEL}\n")

# === Функция RAG-поиска ===
def rag_query(query: str, top_k: int = TOP_K):
    print(f"🔍 Запрос: {query}\n")

    # 1️⃣ Получаем эмбеддинг запроса
    q_emb = embedding_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    q_emb = q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb

    # 2️⃣ Делаем поиск в Chroma
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # 3️⃣ Формируем контекст
    context_blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        context_blocks.append(f"[{i}] Источник: {meta.get('source', '—')}\n{doc[:1000]}")
    context = "\n\n".join(context_blocks)

    # 4️⃣ Отправляем запрос в LLM (через Ollama)
    prompt = f"""
Ты — интеллектуальный ассистент, работающий с нормативными документами.
Используй приведённые фрагменты ГОСТов как контекст.
Если информации недостаточно, явно укажи это.

Контекст:
{context}

Вопрос: {query}
Ответ:
"""
    response = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )

    answer = response.choices[0].message.content.strip()
    print("\n🧠 Ответ LLM:\n")
    print(answer)

    return answer


# === Пример запуска ===
if __name__ == "__main__":
    rag_query("требования к защите информации при обработке персональных данных")
