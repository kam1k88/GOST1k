import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from striprtf.striprtf import rtf_to_text
import pdfplumber
from docx import Document

# === Настройки ===
CHROMA_DIR = r"C:\Users\kam1k88\GOST1k\chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DOCS_DIR = "docs"

# Отключаем телеметрию Chroma
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# Создаём папку под базу, если её нет
os.makedirs(CHROMA_DIR, exist_ok=True)

# === Инициализация моделей и клиента ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("gost1k")
print(f"[✓] Подключено к Chroma DB: {CHROMA_DIR}")
print(f"[✓] Коллекция: gost1k (существует или создана)\n")

# === Читалки файлов ===
def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_rtf(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return rtf_to_text(f.read())

def read_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

# === Универсальный загрузчик ===
def load_documents(folder: str):
    docs = []
    for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".txt":
                raw = read_txt(path)
            elif ext == ".rtf":
                raw = read_rtf(path)
            elif ext == ".docx":
                raw = read_docx(path)
            elif ext == ".pdf":
                raw = read_pdf(path)
            else:
                continue

            text = " ".join(raw.split())
            if text.strip():
                docs.append({"text": text, "source": os.path.basename(path)})

        except Exception as e:
            print(f"[!] Не удалось прочитать {path}: {e}")
    return docs

# === Индексация ===
def ingest():
    print("[~] Загружаем документы из:", DOCS_DIR)
    docs = load_documents(DOCS_DIR)
    print(f"[✓] Загружено {len(docs)} документов. Создаём векторный индекс...")

    texts = ["passage: " + d["text"] for d in docs]
    metadatas = [{"source": d["source"]} for d in docs]
    ids = [f"doc_{i:04d}" for i in range(len(docs))]

    # Вычисляем эмбеддинги (GPU ускорение)
    print("[~] Генерируем эмбеддинги...")
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,
        convert_to_numpy=True
    )

    # === Ключевая часть ===
    collection.add(
        ids=ids,
        documents=texts,            # ← сохраняем тексты
        metadatas=metadatas,
        embeddings=embeddings.tolist()   # ← исправлено: ndarray → list
    )

    print(f"[✓] Индексация завершена. В базе {len(docs)} документов.")
    print(f"[✓] Файлы сохранены в: {CHROMA_DIR}")

if __name__ == "__main__":
    ingest()
