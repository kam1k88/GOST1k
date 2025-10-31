# GOST1k

Локальный RAG по документам ТК 362 "Защита информации".  
Работает офлайн, без облаков. Используется для поиска, анализа и генерации ответов в контексте ГОСТов и других нормативных документов.

---

## Структура проекта

| Файл / папка | Назначение |
|---------------|-------------|
| `ingest.py` | Индексация документов из `docs/` в базу ChromaDB. Читает PDF и DOCX, разбивает текст на фрагменты и создаёт эмбеддинги через `SentenceTransformer (E5-small)` |
| `rag_main.py` | Основная логика RAG. Выполняет поиск по векторной базе (`dense_query`), ранжирование результатов (`CrossEncoder bge-reranker-base`) и генерацию ответа через Ollama (модель Qwen или LLaMA). |
| `rag_api.py` | REST API на FastAPI. Принимает POST-запросы с вопросом и возвращает сгенерированный ответ и список найденных фрагментов. |
| `ui.py` | Интерфейс на Streamlit. Отправляет запросы к API и показывает результат с текстом ответа и топом найденных фрагментов. |
| `check_ollama.py` | Проверка доступности локального сервера Ollama и модели. |
| `requirements.txt` | Список зависимостей для развёртывания. |
| `docs/` | Папка с исходными документами ГОСТ (PDF/DOCX). |
| `chroma_db/` | Локальная база эмбеддингов. |
| `logs/` | Логи запросов и генерации. |

---

## Быстрый запуск

```bash
git clone https://github.com/kam1k88/GOST1k.git
cd GOST1k

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python ingest.py --rebuild
python rag_api.py


UI можно запустить отдельно:
streamlit run ui.py

Пример запроса через API
curl -X POST "http://127.0.0.1:8000/api/query" ^
     -H "Content-Type: application/json" ^
     -d "{\"query\": \"аудит информационной безопасности\"}"

Зависимости

Python 3.11+

PyMuPDF, ChromaDB, SentenceTransformers, FastAPI, Streamlit

Ollama (локальный сервер) с моделью Qwen2.5 или аналогичной

Лицензия
MIT License © 2025