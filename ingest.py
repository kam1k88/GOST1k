import os
import fitz  # PyMuPDF
from docx import Document
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

# === Пути и настройки ===
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./docs"
COLLECTION_NAME = "gost1k"
BATCH_SIZE = 3000  # ограничение для Chroma

# === Инициализация клиента ===
client = chromadb.PersistentClient(path=CHROMA_DIR)
embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")

# === Извлечение текста ===
def extract_text_from_pdf(path):
    text = ""
    try:
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text("text")
    except Exception as e:
        print(f"[⚠️] Ошибка чтения PDF {path}: {e}")
    return text

def extract_text_from_docx(path):
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"[⚠️] Ошибка чтения DOCX {path}: {e}")
        return ""

# === Разбиение на чанки ===
def chunk_text(text, max_len=512):
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
    chunks, cur = [], ""
    for para in paras:
        if len(cur) + len(para) < max_len:
            cur += " " + para
        else:
            chunks.append(cur.strip())
            cur = para
    if cur:
        chunks.append(cur.strip())
    return chunks

# === Индексация ===
def ingest_docs():
    collection = client.get_or_create_collection(COLLECTION_NAME)

    files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith((".pdf", ".docx"))]
    print(f"[ℹ️] Найдено {len(files)} документов ТК 362 для индексации.")

    all_chunks, all_meta = [], []

    for file in tqdm(files):
        path = os.path.join(DOCS_DIR, file)
        text = ""
        if file.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif file.lower().endswith(".docx"):
            text = extract_text_from_docx(path)

        if not text.strip():
            print(f"[⚠️] Пропущен пустой файл: {file}")
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append("passage: " + chunk)
            all_meta.append({
                "source": file,
                "chunk_id": i,
                "abs_path": path
            })

    print(f"[+] Добавлено {len(all_chunks)} фрагментов. Генерируем эмбеддинги...")

    # === Генерация эмбеддингов и upsert по батчам ===
    total = len(all_chunks)
    for i in range(0, total, BATCH_SIZE):
        batch_docs = all_chunks[i:i+BATCH_SIZE]
        batch_meta = all_meta[i:i+BATCH_SIZE]
        batch_ids = [f"id_{i+j}" for j in range(len(batch_docs))]

        embeddings = embedder.encode(batch_docs, normalize_embeddings=True)
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings
        )
        print(f"[📦] Индексировано {i+len(batch_docs)}/{total} фрагментов")

    print(f"[✓] Индексация завершена. Всего документов: {len(files)}, фрагментов: {len(all_chunks)}")

# === Пересоздание коллекции ===
def rebuild():
    print("[♻️] Удаляем старую коллекцию...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"[⚠️] Ошибка при удалении: {e}")

    print("[🆕] Создаём новую коллекцию...")
    _ = client.get_or_create_collection(COLLECTION_NAME)
    ingest_docs()

# === CLI ===
if __name__ == "__main__":
    import sys
    if "--rebuild" in sys.argv:
        rebuild()
    else:
        ingest_docs()
