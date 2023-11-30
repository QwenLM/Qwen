import streamlit as st
from streamlit_chat import message
from fastllm_pytools import llm
import sys

st.set_page_config(
    page_title="fastllm web demo",
    page_icon=":robot:"
)

@st.cache_resource
def get_model():
    model = llm.model(sys.argv[1])
    return model

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, (prompt, response) in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)

if prompt := st.chat_input("请开始对话"):
    model = get_model()
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in model.stream_response(prompt, st.session_state.messages, one_by_one = True):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append((prompt, full_response))
