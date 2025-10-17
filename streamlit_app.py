import streamlit as st
from search import search

st.set_page_config(page_title="GOST1k", layout="wide")
st.title("📚 GOST1k: Поиск по ГОСТ-документам")

query = st.text_input("Введите вопрос по нормативным требованиям:", key="query")

if query:
    st.write("\nРезультаты:")
    search(query, streamlit_output=True)
