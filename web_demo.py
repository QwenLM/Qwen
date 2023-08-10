# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser

import gradio as gr
import mdtex2html
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

DEFAULT_CKPT_PATH = 'QWen/QWen-7B-Chat'


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _parse_text(text):
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
                    line = line.replace("`", r"\`")
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


def _launch_demo(args, model, tokenizer):
    task_history = []

    def predict(_query, _chatbot):
        print("User: " + _parse_text(_query))
        _chatbot.append((_parse_text(_query), ""))
        full_response = ""

        for response in model.chat_stream(tokenizer, _query, history=task_history):
            _chatbot[-1] = (_parse_text(_query), _parse_text(response))

            yield _chatbot
            full_response = _parse_text(response)

        task_history.append((_query, full_response))
        print("Qwen-7B-Chat: " + _parse_text(full_response))

    def regenerate(_chatbot):
        if not task_history:
            yield _chatbot
            return
        item = task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot)

    def reset_user_input():
        return gr.update(value="")

    def reset_state():
        task_history.clear()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://modelscope.cn/api/v1/models/qwen/Qwen-7B-Chat/repo?
Revision=master&FilePath=assets/logo.jpeg&View=true" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>Qwen-7B-Chat Bot</center>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen-7B-Chat, developed by Alibaba Cloud. \
(本WebUI基于Qwen-7B-Chat打造，实现聊天机器人功能。)</center>""")
        gr.Markdown("""\
<center><font size=4>Qwen-7B <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖 </a> 
| <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>&nbsp ｜ 
Qwen-7B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>&nbsp ｜ 
&nbsp<a href="https://github.com/QwenLM/Qwen-7B">Github</a></center>""")

        chatbot = gr.Chatbot(label='Qwen-7B-Chat', elem_classes="control-height")
        query = gr.Textbox(lines=2, label='Input')

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(predict, [query, chatbot], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot], [chatbot], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen-7B. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen-7B的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer)


if __name__ == '__main__':
    main()
