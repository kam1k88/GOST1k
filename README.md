# GOST1k

Локальный RAG по документам ТК 362 "Защита информации".  
Работает офлайн, без облаков. Используется для поиска, анализа и генерации ответов в контексте ГОСТов и других нормативных документов.

---

## Структура

| Файл | Назначение |
|------|-------------|
| `ingest.py` | Индексация документов из `docs/` в базу ChromaDB |
| `rag_main.py` | Основная логика поиска, rerank и генерации |
| `rag_api.py` | REST API (FastAPI) |
| `ui.py` | Интерфейс Streamlit |
| `check_ollama.py` | Проверка Ollama |
| `requirements.txt` | зависимости проекта |
| `docs/`, `chroma_db/`, `logs/` | данные, база и логи |

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


Пример запроса
curl -X POST "http://127.0.0.1:8000/api/query" ^
     -H "Content-Type: application/json" ^
     -d "{\"query\": \"аудит информационной безопасности\"}"

Зависимости
Python 3.11+
PyMuPDF, ChromaDB, SentenceTransformers, FastAPI, Streamlit
Ollama с моделью Qwen2.5 или аналогичной

Лицензия
MIT License © 2025