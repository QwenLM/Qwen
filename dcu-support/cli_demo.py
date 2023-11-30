# coding=utf-8
import argparse
from fastllm_pytools import llm

def args_parser():
    parser = argparse.ArgumentParser(description = 'qwen_chat_demo')
    parser.add_argument('-p', '--path', type = str, required = True, default = '', help = '模型文件的路径')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    model = llm.model(args.path)

    history = []
    print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("AI:", end = "")
        curResponse = ""
        for response in model.stream_response(query, history = history, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0):
            curResponse += response
            print(response, flush = True, end = "")
        history.append((query, curResponse))