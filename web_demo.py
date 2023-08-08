#  streamlit run web_demo.py

import json
import torch
import streamlit as st
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

st.set_page_config(page_title="Qwen-7B-Chat")
st.title("Qwen-7B-Chat")

@st.cache_resource
def init_model():
    model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto", revision = 'v1.0.1',trust_remote_code=True, bf16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("qwen/Qwen-7B-Chat", revision = 'v1.0.1',trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B", revision = 'v1.0.1', trust_remote_code=True)
    return model, tokenizer

def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是通义千问，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model, tokenizer = init_model()
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            for response in model.chat(tokenizer, prompt, history=None, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
