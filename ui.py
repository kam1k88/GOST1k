import os
import streamlit as st
import asyncio
from rag_main import answer

# === Отключаем системные логи Streamlit ===
os.environ["STREAMLIT_SERVER_ENABLE_LOGGING"] = "false"
os.environ["STREAMLIT_LOG_LEVEL"] = "error"

# === Настройки страницы ===
st.set_page_config(
    page_title="🔍 GOST1k — поиск по регуляторным документам",
    layout="wide",  # widescreen
    initial_sidebar_state="collapsed"
)

# === Заголовок ===
st.markdown("<h1 style='color:#fff;'>GOST1k</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#aaa;'",
    unsafe_allow_html=True,
)

# === Поле ввода ===
query = st.text_area(
    "Введите запрос:",
    placeholder="""Например:
- Пример письма руководителю ТК 260
- Какие требования предъявляются к защите КИИ?
- Перечисли стандарты по управлению уязвимостями
- Сравни ГОСТ Р 57580 и СТО БР ИББС
- Какие документы ФСТЭК регулируют категорирование?
""",
    height=200
)

# === Обработка ===
if query.strip():
    with st.spinner("Обработка запроса..."):
        ans = asyncio.run(answer(query))
    st.markdown(ans)
