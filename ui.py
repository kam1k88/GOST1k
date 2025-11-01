import streamlit as st
import asyncio
from rag_main import answer

st.title("GOST1k")

query = st.text_input("Введите запрос:")
if query:
    ans = asyncio.run(answer(query))
    st.markdown(ans)