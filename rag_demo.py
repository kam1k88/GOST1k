import chromadb

# Подключаемся к локальной базе
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection("gost_r")

# Пример запроса
query = "аудит информационной безопасности и регистрация событий"
results = collection.query(query_texts=[query], n_results=5)

print("\n🔍 Запрос:", query)
print("\n🏆 Топ-5 совпадений:")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
    print(f"{i}. {doc[:250]}...")
    print(f"   [source={meta.get('source')}]")
