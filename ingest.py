import os
import re
import time
import torch
import chromadb
import shutil
import gc

from tqdm import tqdm
from datetime import datetime
from FlagEmbedding import BGEM3FlagModel
from docx import Document
from PyPDF2 import PdfReader

# === Настройки ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "gost1k"
os.makedirs(CHROMA_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
MIN_LEN = 64
MAX_LEN = 2048

# === Логирование ===
def log(msg, icon="💬"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{icon}] {msg}")

# === Подсчет страниц и адаптация чанков ===
def get_pdf_page_count(path):
    try:
        reader = PdfReader(path)
        return len(reader.pages)
    except Exception:
        return 1

def adapt_chunk_params(pdf_path):
    pages = get_pdf_page_count(pdf_path)
    if pages > 250:
        chunk_size, overlap, label = 1024, 256, "📘 Большой документ"
    elif pages > 80:
        chunk_size, overlap, label = 768, 192, "📗 Средний документ"
    else:
        chunk_size, overlap, label = 512, 128, "📙 Малый документ"
    log(f"{label}: {os.path.basename(pdf_path)} → CHUNK={chunk_size}, OVERLAP={overlap}")
    return chunk_size, overlap

# === Очистка текста ===
def clean_text(text: str) -> str:
    text = re.sub(r"КонсультантПлюс.*?(?=\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.consultant\.ru", "", text)
    text = re.sub(r"страница\s*\d+\s*(из)?\s*\d*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"надежная правовая поддержка", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Разделение текста ===
def smart_split(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = clean_text(text)
    pattern = r"(?=(?:^|\s)(?:\d+\.\d+\.?\d*|[A-ZА-Я]\.[\d\.]+|[a-zа-я]\)|[a-f]\)|[A-F]\))\s)"
    chunks = re.split(pattern, text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > MIN_LEN]
    result, buf = [], ""
    for chunk in chunks:
        if len(buf) + len(chunk) < chunk_size:
            buf += " " + chunk
        else:
            if len(buf) > MIN_LEN:
                result.append(buf.strip())
            buf = chunk
    if buf:
        result.append(buf.strip())
    return [c for c in result if len(c) <= MAX_LEN]

# === Чтение файлов ===
def read_file(path):
    text = ""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".docx":
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif ext == ".pdf":
            pdf = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        log(f"Ошибка чтения {os.path.basename(path)}: {e}", "⚠️")
    return clean_text(text)

# === инициализация и прогрев модели ===
# === Индексация ===
def ingest_docs(rebuild=True):
    log(f"{'Полная переиндексация' if rebuild else 'Добавление новых документов'} GOST1k...", "🧱")
    log(f"Устройство: {DEVICE}", "⚙️")

    # === инициализация и прогрев модели ===
    gpu_name = "CPU"
    vram_gb = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(props.total_memory / 1024**3, 1)
    log(f"🧠 GPU: {gpu_name} ({vram_gb} ГБ VRAM) | FP16: {'Yes' if torch.cuda.is_available() else 'No'} | BGE-M3 hybrid", "⚙️")

    t0 = time.time()
    embedder = BGEM3FlagModel(MODEL_NAME, use_fp16=True, device=DEVICE)
    try:
        _ = embedder.encode(["warmup"], batch_size=1, max_length=128)
        warm = time.time() - t0
        log(f"🔥 Модель загружена и прогрета за {warm:.1f} с. Веса закэшированы в HuggingFace.", "✅")
    except Exception as e:
        log(f"⚠️ Прогрев не выполнен: {e}", "⚠️")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            log("Старая коллекция удалена (логически)", "🗑️")
            for d in os.listdir(CHROMA_DIR):
                path = os.path.join(CHROMA_DIR, d)
                if os.path.isdir(path) and re.match(r"^[0-9a-f\-]{36}$", d):
                    shutil.rmtree(path, ignore_errors=True)
                    log(f"Удалена папка {d}", "🧹")
        except Exception as e:
            log(f"Коллекция отсутствовала — создаём заново ({e})", "⚠️")

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith((".txt", ".md", ".docx", ".pdf"))]
    log(f"Найдено файлов: {len(files)}", "📄")

    total_chunks = 0
    start_time = time.time()

    for file in tqdm(files, desc="🔍 Индексация", ncols=100):
        path = os.path.join(DOCS_DIR, file)
        if path.lower().endswith(".pdf"):
            chunk_size, overlap = adapt_chunk_params(path)
        else:
            chunk_size, overlap = CHUNK_SIZE, CHUNK_OVERLAP

        text = read_file(path)
        if not text or len(text) < MIN_LEN:
            continue

        chunks = smart_split(text, chunk_size, overlap)
        if not chunks:
            continue

        embeddings = []
        cur_batch = BATCH_SIZE

        for i in range(0, len(chunks), cur_batch):
            batch = chunks[i:i + cur_batch]
            while True:
                try:
                    res = embedder.encode(batch, batch_size=cur_batch, max_length=2048)
                    dense_vecs = res["dense_vecs"]
                    embeddings.extend(dense_vecs)
                    break
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        torch.cuda.empty_cache()
                        gc.collect()
                        cur_batch = max(1, cur_batch // 2)
                        log(f"⚠️ OOM — уменьшаю batch до {cur_batch}", "⚠️")
                        time.sleep(1)
                    else:
                        raise

        ids = [f"{file}_{i}" for i in range(len(chunks))]
        metas = [{"source": file, "chunk": i} for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metas)

        total_chunks += len(chunks)
        vram_now = torch.cuda.memory_allocated() / 1024**2
        log(f"[✅] {file}: {len(chunks)} фрагментов | VRAM: {vram_now:.0f} МБ", "📦")

        torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - start_time
    log(f"Всего проиндексировано: {total_chunks} фрагментов за {elapsed:.1f} сек", "📦")
    log("Индексация завершена", "✅")

if __name__ == "__main__":
    ingest_docs(rebuild=True)
    input("Для выхода нажмите Enter...")
