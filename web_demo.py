#!/usr/bin/env python3

from transformers import AutoTokenizer
import gradio as gr
import mdtex2html
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from argparse import ArgumentParser
import sys

print("Call args:" + str(sys.argv))
parser = ArgumentParser()
parser.add_argument("--share", action="store_true", default=False)
parser.add_argument("--inbrowser", action="store_true", default=False)
parser.add_argument("--server_port", type=int, default=80)
parser.add_argument("--server_name", type=str, default="0.0.0.0")
parser.add_argument("--exit", action="store_true", default=False)
parser.add_argument("--model_revision", type=str, default="")
args = parser.parse_args(sys.argv[1:])
print("Args:" + str(args))

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-7B-Chat", trust_remote_code=True, resume_download=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    device_map="auto",
    trust_remote_code=True,
    resume_download=True,
    **{"revision": args.model_revision}
    if args.model_revision is not None
    and args.model_revision != ""
    and args.model_revision != "None"
    else {},
).eval()

model.generation_config = GenerationConfig.from_pretrained(
    "Qwen/Qwen-7B-Chat", trust_remote_code=True, resume_download=True
)

if "exit" in args:
    if args.exit:
        sys.exit(0)
    else:
        del args.exit

if "model_revision" in args:
    del args.model_revision


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


task_history = []


def predict(query, chatbot):
    print("User: " + parse_text(query))
    chatbot.append((parse_text(query), ""))
    fullResponse = ""

    for response in model.chat_stream(tokenizer, query, history=task_history):
        chatbot[-1] = (parse_text(query), parse_text(response))

        yield chatbot
        fullResponse = parse_text(response)

    task_history.append((query, fullResponse))
    print("Qwen-7B-Chat: " + parse_text(fullResponse))


def regenerate(chatbot):
    if not task_history:
        yield chatbot
        return
    item = task_history.pop(-1)
    chatbot.pop(-1)
    yield from predict(item[0], chatbot)


def reset_user_input():
    return gr.update(value="")


def reset_state():
    task_history.clear()
    return []


with gr.Blocks() as demo:
    gr.Markdown("""<p align="center"><img src="https://modelscope.cn/api/v1/models/qwen/Qwen-7B-Chat/repo?Revision=master&FilePath=assets/logo.jpeg&View=true" style="height: 80px"/><p>""")
    gr.Markdown("""<center><font size=8>Qwen-7B-Chat Bot</center>""")
    gr.Markdown(
        """<center><font size=3>This WebUI is based on Qwen-7B-Chat, developed by Alibaba Cloud. (本WebUI基于Qwen-7B-Chat打造，实现聊天机器人功能。)</center>"""
    )
    gr.Markdown(
        """<center><font size=4>Qwen-7B <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖 <a> | <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>&nbsp ｜ Qwen-7B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖 <a>| <a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>&nbsp ｜ &nbsp<a href="https://github.com/QwenLM/Qwen-7B">Github</a></center>"""
    )

    chatbot = gr.Chatbot(lines=10, label='Qwen-7B-Chat', elem_classes="control-height")
    query = gr.Textbox(lines=2, label='Input')

    with gr.Row():
        emptyBtn = gr.Button("🧹 Clear History (清除历史)")
        submitBtn = gr.Button("🚀 Submit (发送)")
        regenBtn = gr.Button("🤔️ Regenerate (重试)")

    submitBtn.click(predict, [query, chatbot], [chatbot], show_progress=True)
    submitBtn.click(reset_user_input, [], [query])
    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)
    regenBtn.click(regenerate, [chatbot], [chatbot], show_progress=True)

    gr.Markdown(
        """<font size=2>Note: This demo is governed by the original license of Qwen-7B. We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc. (注：本演示受Qwen-7B的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)"""
    )

if len(sys.argv) > 1:
    demo.queue().launch(**vars(args))
else:
    demo.queue().launch()