# 🧠 GOST1k — применение SentenceTransformers для локального RAG

**GOST1k** — это локальный стек **Retrieval-Augmented Generation (RAG)** для поиска по документам **ГОСТ**.  
Он сочетает надёжность локальной базы знаний (**ChromaDB**) с мощью современных языковых моделей (**E5 + BGE + Qwen/Ollama**).

---

## 📚 Цель

Повысить качество поиска и индексации документов, следуя официальным рекомендациям  
фреймворка **SentenceTransformers (SBERT)**.

---
📍 Структура проекта


GOST1k/
├─ docker-compose.yml
│
├─ api/
│   ├─ Dockerfile
│   └─ rag_api.py
│
├─ embedder/
│   ├─ Dockerfile
│   └─ embedder_server.py
│
├─ reranker/
│   ├─ Dockerfile
│   └─ reranker_server.py
│
├─ chroma_db/
│   └─ (векторная база)
│
└─ docs/
    └─ (документы для индексации)


---

## 🧩 Этап 1. Индексация документов (E5-small)

Используется **bi-encoder** модель `intfloat/e5-small`.  
Она кодирует тексты независимо, создавая векторные представления.

**Основные практики:**

- **Префиксы:**
  ```python
  passage_text = "passage: " + segment
  query_text = "query: " + user_query
  ```

- **Нормализация:**
  ```python
  embeddings = model.encode(texts, normalize_embeddings=True)
  ```

- **Разбиение текста:**
  > ГОСТ разбивается на фрагменты (пункты, параграфы) не длиннее ~512 токенов.

**Пример запуска:**
```bash
python main.py ingest
```

---

## 🔍 Этап 2. Семантический поиск

Модуль `search.py` принимает запрос пользователя и находит близкие по смыслу фрагменты:

```python
query_text = "query: " + user_query
q_emb = model.encode(query_text, normalize_embeddings=True)
results = chroma.query(query_embeddings=q_emb, n_results=10)
```

- Используется **асимметричный поиск** (`query/passage`)
- Метрика — **косинусное сходство**

---

## ⚖️ Этап 3. Реранжирование результатов (BGE-reranker-base)

Для повышения точности используется **кросс-энкодер** `BAAI/bge-reranker-base`.  
Он оценивает пары *(запрос, кандидат)* и сортирует результаты по релевантности.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")
scores = reranker.predict([(query, passage), ...])
```

- Работает только на этапе поиска (не индексируется)
- Не требует префиксов
- Оптимально: **top-10 → top-3** финальных ответов

---

## 🚀 Этап 4. Генерация ответа (LLM)

Лучшие фрагменты передаются в **LLM (Ollama / Qwen2.5-7B-Instruct)** для генерации итогового ответа:

```bash
python main.py search "Как проверить соответствие ГОСТ Р 34.11-2012?"
```

В веб-интерфейсе **Streamlit** отображается ответ + цитаты.

---

## 🧮 Оптимизация

- **Параллельное кодирование:**
  ```python
  pool = model.start_multi_process_pool(target_devices=["cuda:0","cuda:1"])
  embeddings = model.encode(texts, pool=pool)
  ```

- **Совместимые версии:**
  ```text
  sentence-transformers>=2.2.2,<6.0
  ```

- **Локальные модели:**
  ```
  models/intfloat/e5-small/
  models/BAAI/bge-reranker-base/
  ```

- **ChromaDB:**
  Хранение векторов и поиск по косинусному сходству:
  ```python
  collection.query(query_embeddings=q_emb, n_results=10)
  ```

---

## 🧠 Подытожим

✅ Индексация bi-энкодером E5 с правильными префиксами и нормализацией  
✅ Семантический поиск по ChromaDB  
✅ Реранжирование с BGE CrossEncoder  
✅ Интеграция с LLM для генерации итогового ответа  
✅ Полностью локальный стек без зависимости от облаков  

---

## 📄 Лицензия

MIT © 2025 — проект **GOST1k**
