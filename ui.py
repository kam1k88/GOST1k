import streamlit as st
import asyncio
from rag_main import answer

# === Настройки страницы ===
st.set_page_config(
    page_title="GOST1k",
    layout="wide",  # widescreen
    initial_sidebar_state="collapsed"
)

st.title("GOST1k")

# === Поле ввода ===
query = st.text_area(
    "Введите запрос:",
    placeholder="""Например:
- Пример письма руководителю ТК 260
- Какие требования предъявляются к защите КИИ?
- Перечисли стандарты по управлению уязвимостями
- Сравни ГОСТ Р 57580 и СТО БР ИББС
- Какие документы ФСТЭК регулируют категорирование?
- Как ФЗ-152 связан с ГОСТ Р 56939?
""",
    height=200
)

# === Обработка ===
if query.strip():
    ans = asyncio.run(answer(query))
    st.markdown(ans)
