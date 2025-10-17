from sentence_transformers import SentenceTransformer, CrossEncoder

# Би-энкодер (E5-small)
print("[+] Загружаем intfloat/multilingual-e5-small...")
SentenceTransformer("intfloat/multilingual-e5-small")

# Кросс-энкодер (BGE reranker)
print("[+] Загружаем BAAI/bge-reranker-base...")
CrossEncoder("BAAI/bge-reranker-base")

print("[✓] Все модели загружены и сохранены в кэше.")