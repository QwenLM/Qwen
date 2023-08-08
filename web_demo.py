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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True, resume_download=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    device_map="auto",
    trust_remote_code=True,
    resume_download=True,
    **{"revision": args.model_revision} if args.model_revision is not None and args.model_revision != "" and args.model_revision != "None" else {},
).eval()

model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True, resume_download=True)

if 'exit' in args:
    if args.exit:
        sys.exit(0)
    else:
        del args.exit

if 'model_revision' in args:
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
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
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


def predict(input, chatbot):
    print('Q: ' + parse_text(input))
    chatbot.append((parse_text(input), ""))
    fullResponse = ""

    for response in model.chat_stream(tokenizer, input, history=task_history):
        chatbot[-1] = (parse_text(input), parse_text(response))

        yield chatbot
        fullResponse = parse_text(response)

    task_history.append((input, fullResponse))
    print("A: " + parse_text(fullResponse))


def reset_user_input():
    return gr.update(value='')


def reset_state():
    task_history = []
    return []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">通义千问 - QwenLM/Qwen-7B</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                query = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")

    submitBtn.click(predict, [query, chatbot], [chatbot], show_progress=True)
    submitBtn.click(reset_user_input, [], [query])
    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)

if len(sys.argv) > 1:
    demo.queue().launch(**vars(args))
else:
    demo.queue().launch()