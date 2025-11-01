# 🧱 GOST1k — локальный RAG для регуляторных документов и стандартов

**GOST1k** — автономная офлайн-система для **поиска, анализа и сопоставления регуляторных документов**,  
включая **ГОСТ, СТО, РД, ФСТЭК, Банк России, ISO, НПА и ТК 362**.

Система построена на принципах hybrid retrieval + fusion, reranking, dense+lexical embeddings, offline on-prem архитектуру.
Есть возможности роста: улучшение разбиения документа (semantic chunking), усиление обработки запросов (query transformation), извлечение релевантных сегментов из документов, обратная связь/адаптация, объясняемость.
В README ты можешь отметить: “использованы техники hybrid retrieval, fusion, reranking” и “внедрена on-prem архитектура без зависимостей от LangChain”.
> 🔒 **Фокус:** достоверное извлечение информации из источников, без "галлюцинаций" и генерации несуществующих данных.
|  №  | Техника из RAG_Techniques                               | Статус в GOST1k | Комментарий                                                              |
| :-: | ------------------------------------------------------- | :-------------: | ------------------------------------------------------------------------ |
|  1  | **Basic Retrieval** (retrieval from vector DB)          |        ✅        | Используется ChromaDB с локальными эмбеддингами                          |
|  2  | **Context Re-Retrieval** (multi-stage retrieval)        |        ✅        | Двухступенчатый Rerank: 40→15→5                                          |
|  3  | **Context Window Management**                           |   ⚙️ Частично   | LLM (Qwen 2.5-7B) получает оптимизированный контекст из top-5 фрагментов |
|  4  | **Embedding Normalization**                             |        ✅        | Включена через BGEM3FlagModel                                            |
|  5  | **Chunking Optimization** (fixed-size / overlapping)    |        ✅        | Реализовано: CHUNK_SIZE=512, OVERLAP=128                                 |
|  6  | **Semantic / Dynamic Chunking**                         |   ⚙️ Частично   | Пока фиксированные чанки; можно добавить семантическое деление           |
|  7  | **Query Transformation** (reformulation, decomposition) |        ❌        | Запросы не переформулируются автоматически                               |
|  8  | **Multi-Query Expansion**                               |        ❌        | Пока одна формулировка запроса                                           |
|  9  | **Hybrid Search (Dense + Sparse)**                      |        ✅        | Полноценный гибрид через BGE-M3 (dense + sparse + lexical)               |
|  10 | **Reciprocal Rank Fusion (RRF)**                        |        ✅        | Используется при объединении dense и sparse результатов                  |
|  11 | **Cross-Encoder Reranking**                             |        ✅        | `BAAI/bge-reranker-base` уточняет выдачу после fusion                    |
|  12 | **Multi-Hop Retrieval**                                 |        ❌        | Нет цепочки уточняющих запросов                                          |
|  13 | **Contextual Merging / Aggregation**                    |   ⚙️ Частично   | Формирование итогового контекста перед LLM                               |
|  14 | **Answer Verification**                                 |        ❌        | Ответ не проходит верификацию отдельным этапом                           |
|  15 | **Source Attribution / Citation**                       |   ⚙️ Частично   | В логах сохраняются источники (можно вывести в UI)                       |
|  16 | **Retrieval Fusion** (multi-model / multi-db)           |        ✅        | Dense + Sparse + RRF = полноценный fusion retrieval                      |
|  17 | **On-Prem Architecture**                                |        ✅        | Полностью локально: без LangChain, API и облаков                         |
|  18 | **RAG with Feedback Loop / Adaptive Retrieval**         |        ❌        | Пока без пользовательской коррекции                                      |
|  19 | **Model Ensemble** (several retrievers/LLMs)            |        ❌        | Используется одна модель для каждого этапа                               |
|  20 | **Caching / Warmup Optimization**                       |        ✅        | Прогрев GPU при старте (`warmup` блок)                                   |
|  21 | **Low VRAM Optimization (FP16)**                        |        ✅        | FP16 включён в BGE-M3                                                    |
|  22 | **Explainable Retrieval (Why selected)**                |   ⚙️ Частично   | Можно вывести топ-фрагменты в UI                                         |
|  23 | **Context Compression**                                 |   ⚙️ Частично   | Топ-5 лучших фрагментов сокращают ввод для LLM                           |
|  24 | **RAG Evaluation Metrics (Recall@K, MRR)**              |        ❌        | Не реализовано (пока нет оценки качества)                                |
|  25 | **RAG Guardrails / Safety Filters**                     |        ❌        | Нет семантической фильтрации, т.к. документы нейтральны                  |
|  26 | **Retriever Fusion with Sparse Model**                  |        ✅        | Встроено в BGE-M3 (dense + BM25-like sparse)                             |
|  27 | **Async Pipeline / Parallel Retrieval**                 |        ✅        | Асинхронная логика через asyncio и httpx                                 |
|  28 | **LLM Response Postprocessing**                         |   ⚙️ Частично   | Ответ просто логируется, без дополнительного редактирования              |
|  29 | **Fully Offline Execution**                             |        ✅        | Всё выполняется локально (Ollama + ChromaDB + torch)                     |
|  30 | **Explainable UI**                                      |   ⚙️ Частично   | Streamlit UI с логами, можно добавить вывод источников                   |

---

## ⚙️ Основные возможности

- 💬 Семантический поиск по любой регуляторке (ГОСТ, СТО, ISO, ФСТЭК, НПА, Банк России)  
- 🧠 Автоматическое определение типа запроса (summary / analysis / compare)  
- 🔁 Многоступенчатая обработка: dense → sparse → RRF → rerank → LLM  
- 🚀 Полностью офлайн: Ollama + локальные модели + локальная ChromaDB  
- 🧩 Гибридный поиск (плотные, разреженные и лексические вектора)  
- 🧱 Оптимизация под GPU (CUDA, FP16) и контроль VRAM  
- 🌑 Тёмная тема Streamlit-интерфейса с логами запросов  

---



<img width="1837" height="729" alt="image" src="https://github.com/user-attachments/assets/2314abf8-e169-4d55-bf00-08fa6ae3caff" />
🧪 Примеры боевых 
(https://github.com/user-attachments/files/23286418/QA.pdf)
от простых к сложным.


---
     ┌──────────────────────────────────────────┐
     │                UI (Streamlit)             │
     │       query → rag_main.answer()           │
     └──────────────────────────────────────────┘
                      │
                      ▼
       ┌────────────────────────────┐
       │        RAG Pipeline         │
       │ 1. BGEM3FlagModel (dense)   │
       │ 2. Sparse retrieval (BM25)  │
       │ 3. RRF Fusion (dense+BM25)  │
       │ 4. CrossEncoder rerank 40→15→5 │
       │ 5. Ollama LLM (answer gen)  │
       └────────────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────────────┐
     │        ChromaDB локальная база            │
     │     • persist/chroma.sqlite3              │
     │     • docs/*.pdf, *.docx (разбитые чанки) │
     └──────────────────────────────────────────┘



---
         ┌──────────────────────────────────────────┐
         │                UI (Streamlit)             │
         │       query → rag_main.answer()           │
         └──────────────────────────────────────────┘
                          │
                          ▼
           ┌────────────────────────────┐
           │        RAG Pipeline         │
           │ 1. BGEM3FlagModel (dense)   │
           │ 2. Sparse retrieval (BM25)  │
           │ 3. RRF Fusion (dense+BM25)  │
           │ 4. CrossEncoder rerank 40→15→5 │
           │ 5. Ollama LLM (answer gen)  │
           └────────────────────────────┘
                          │
                          ▼
         ┌──────────────────────────────────────────┐
         │        ChromaDB локальная база            │
         │     • persist/chroma.sqlite3              │
         │     • docs/*.pdf, *.docx (разбитые чанки) │
         └──────────────────────────────────────────┘

---

## 🧠 Модели

| Компонент | Модель | Назначение |
|------------|---------|------------|
| **Dense** | `BAAI/bge-m3` | Многоязычные плотные вектора (dense + sparse + lexical) |
| **Reranker** | `BAAI/bge-reranker-base` | Кросс-энкодер для уточнения top-K |
| **LLM (локально)** | через `Ollama` (`mistral`, `llama3`, `phi3`, `qwen2`) | Генерация ответов |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Объединение dense и sparse результатов |

---

## 📊 Статус компонентов RAG

| Компонент | Статус | Комментарий |
|------------|:------:|-------------|
| **BGEM3FlagModel** | ✅ | Используется `BAAI/bge-m3` (dense + sparse + lexical) |
| **normalize_embeddings** | ✅ | Нормализация векторов включена |
| **sparse retrieval (BM25)** | ✅ | Активен в гибридном поиске |
| **fusion (RRF)** | ✅ | Комбинация dense и sparse |
| **reranker (CrossEncoder)** | ✅ | Двухступенчатый rerank 40→15→5 |
| **Ollama async** | ✅ | Локальный LLM с асинхронной генерацией |
| **GPU warmup** | ✅ | Прогрев CUDA при старте |
| **логирование** | ✅ | Все запросы и ответы пишутся в `logs/gost1k.log` |
query → dense + sparse retrieve → fusion (RRF) → rerank-1 → rerank-2 → generate → log
---

## 🚀 Быстрый старт

| Шаг | Команда | Описание |
|-----|----------|-----------|
| **1. Клонировать проект** | ```bash<br>git clone https://github.com/kam1k88/GOST1k.git<br>cd GOST1k``` | Клонирование репозитория и переход в директорию проекта |
| **2. Создать окружение и установить зависимости** | ```bash<br>python -m venv venv<br>venv\Scripts\activate<br>pip install -r requirements.txt``` | Создание виртуального окружения и установка пакетов |
| **3. Индексация документов из папки `docs/`** | ```bash<br>python ingest.py --rebuild``` | Построение базы ChromaDB с эмбеддингами |
| **4. Запустить интерфейс** | ```bash<br>streamlit run ui.py``` | Запуск веб-интерфейса Streamlit |
| **5. Открыть в браузере** | [http://localhost:8501](http://localhost:8501) | Интерфейс доступен локально |

---


💡 **Примечание:**  
Все действия выполняются **локально**, без интернета и облаков.  
ChromaDB хранит эмбеддинги и документы on-prem, полностью автономно.
| Компонент                   | Модель                       | Фреймворк / библиотека                  | Назначение                                              | Примечание                                                 |
| --------------------------- | ---------------------------- | --------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------- |
| **Embedder**                | `BAAI/bge-m3`                | **FlagEmbedding (BGEM3FlagModel)**      | Генерация гибридных векторов (dense + sparse + lexical) | Многоязычная, идеально работает с русскими ГОСТами         |
| **Reranker**                | `BAAI/bge-reranker-base`     | **SentenceTransformers (CrossEncoder)** | Точное переупорядочивание top-K документов              | Лёгкий и надёжный для GPU ≤8 GB                            |
| **LLM (генератор ответов)** | `qwen2.5-7b-instruct-q4_K_M` | **Ollama (16k tokens)**                 | Генерация связных и корректных текстов                  | Поддерживает русский, оптимизирован под Q4_K_M квантование |
<img width="701" height="180" alt="image" src="https://github.com/user-attachments/assets/b94a6b7a-644b-4e8d-aabd-17fa162db6d8" />


---
🔒 Приватность

Всё выполняется локально, без внешних API и облаков

База, документы и модели хранятся на диске пользователя

Логи (logs/gost1k.log) содержат только запросы и ответы
---
📦 Структура проекта
| Файл / папка       | Назначение                                                                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ingest.py`        | Индексация документов из `docs/` в базу ChromaDB. Читает PDF и DOCX, разбивает текст на чанки и создаёт эмбеддинги через `FlagModel (E5-M3)` |
| `rag_main.py`      | Основная логика RAG: поиск по базе (`dense_query`), ранжирование (`CrossEncoder bge-reranker-base`), генерация ответа через Ollama                        |
| `rag_api.py`       | REST API (FastAPI)                                                                                                                                        |
| `ui.py`            | Веб-интерфейс (Streamlit)                                                                                                                                 |
| `check_ollama.py`  | Проверка доступности Ollama                                                                                                                               |
| `requirements.txt` | Зависимости проекта                                                                                                                                       |
| `docs/`            | Папка с исходными документами (`.pdf`, `.docx`)                                                                                                           |
| `chroma_db/`       | Локальная база эмбеддингов                                                                                                                                |
| `logs/`            | Логи запросов и генерации                                                                                                                                 |
---
📊 Технические параметры
| Компонент     | Значение               |
| ------------- | ---------------------- |
| CHUNK_SIZE    | 512                    |
| CHUNK_OVERLAP | 128                    |
| BATCH_SIZE    | 32                     |
| DEVICE        | CUDA (автоопределение) |
| RERANK        | 40 → 15 → 5            |
| RRF K         | 75                     |
---
💾 Объем установки ~22 ГБ на диске
| Компонент                                     | Назначение                                      | Вес      |
| --------------------------------------------- | ----------------------------------------------- | -------- |
| **venv (Python, torch, streamlit, chromadb)** | Рабочее окружение                               | ~6 ГБ    |
| **docs/**                                     | Исходные документы (PDF, DOCX)                  | —        |
| **chroma_db/**                                | Векторная база эмбеддингов (≈2× от объёма docs) | —        |
| **CUDA Toolkit 13.0**                         | GPU-ускорение (опционально)                     | ~4 ГБ    |
| **WSL (Ubuntu 22.04)**                        | Среда для Python и Ollama                       | ~6 ГБ    |
| **Ollama**                                    | Локальный сервер моделей                        | ~0.3 ГБ  |
| **Qwen 2.5-7B-Instruct-Q4_K_M**               | Основная LLM-модель                             | ~4.5 ГБ  |
| **E5-M3**                                     | FlagModel-модель эмбеддингов                    | ~0.35 ГБ |
| **BGE-reranker-base**                         | CrossEncoder-reranker                           | ~0.5 ГБ  |
---
💾 Зависимости
torch
chromadb
FlagEmbedding
sentence-transformers
python-dotenv
httpx
fastapi
streamlit
---
🚫 Почему GOST1k не использует LangChain / LangGraph

GOST1k решает задачу локального on-prem RAG, а не построения облачных агентных систем.

LangChain и LangGraph предназначены для сложных мультиагентных пайплайнов с внешними API.
GOST1k — это минималистичная, автономная и полностью воспроизводимая система, где всё выполняется напрямую и локально:
Отличия:

❌ никаких SDK и внешних API

💻 только локальные модели (Ollama, E5-M3, BGE-reranker-base)

🔒 полностью офлайн-режим

⚙️ прозрачная логика и контроль над каждым этапом

Такой подход быстрее, надёжнее и идеально подходит для on-prem сред, где важна безопасность и воспроизводимость.
---

📘 Автор Аркадий Максимов (kam1k88)
Создатель проекта GOST1k — локального RAG-поисковика по любой регуляторке и стандартам ИБ.
GitHub: https://github.com/kam1k88/GOST1k

📜 Лицензия
MIT License © 2025 Аркадий Максимов
