import os
import platform
import signal
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
# We recommend checking the support of BF16 first. Run the command below:
# import torch
# torch.cuda.is_bf16_supported()
# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# use fp32
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat",
                                                           trust_remote_code=True)  # 可指定不同的生成长度、top_p等相关超参

stop_stream = False


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def print_history(history):
    for pair in history:
        print(f"\nUser：{pair[0]}\nQwen-7B：{pair[1]}")


def main():
    history, response = [], ''
    global stop_stream
    clear_screen()
    print("欢迎使用 Qwen-7B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\nUser：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            clear_screen()
            print("欢迎使用 Qwen-7B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        for response in model.chat(tokenizer, query, history=history, stream=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                clear_screen()
                print_history(history)
                print(f"\nUser: {query}")
                print("\nQwen-7B：", end="")
                print(response)

        history.append((query, response))


if __name__ == "__main__":
    main()
