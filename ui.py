import streamlit as st
import requests
import json

st.set_page_config(page_title="GOST1k RAG", layout="wide")

st.title("🧩 GOST1k — Анализ ГОСТов ТК 362")

# === Панель параметров ===
with st.sidebar:
    st.header("⚙️ Настройки")
    api_url = st.text_input("API URL:", "http://127.0.0.1:8000/api/query")
    model = st.selectbox(
        "Модель Ollama:",
        ["qwen2.5:7b-instruct-q4_K_M", "llama2:7b", "gemma2:2b", "qwen2.5:3b-instruct-q4_K_M"],
        index=0
    )
    mode = st.selectbox(
        "Режим ответа:",
        ["structured", "reflective", "deep_analytic"],
        index=0
    )

st.markdown("---")

# === Основная область ===
query = st.text_area("Введите запрос:", "аудит информационной безопасности и регистрация событий")

if st.button("🔍 Запросить"):
    if not query.strip():
        st.warning("Введите текст запроса.")
    else:
        with st.spinner("Обработка запроса..."):
            try:
                payload = {"query": query, "mode": mode}
                response = requests.post(api_url, json=payload, timeout=600)
                if response.status_code == 200:
                    data = response.json()
                    result = data.get("result", "")
                    st.success("✅ Ответ сгенерирован:")
                    st.markdown(result)

                    # если в ответе содержатся фрагменты (опционально, для расширенного API)
                    if "docs" in data:
                        st.markdown("### 🏆 Топ-5 фрагментов из базы:")
                        for i, d in enumerate(data["docs"], 1):
                            st.markdown(f"**{i}.** {d['text'][:250]} ...")
                            st.caption(d.get("source", ""))

                else:
                    st.error(f"Ошибка {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"⚠️ Ошибка запроса: {e}")
