import argparse
from fastllm_pytools import llm
import time

def args_parser():
    parser = argparse.ArgumentParser(description = 'fastllm_chat_demo')
    parser.add_argument('-p', '--path', type = str, required = True, default = '', help = '模型文件的路径')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()

    model_path = args.path

    prompts = ["深圳有什么好玩的", "上海有什么好玩的", "晚上睡不着怎么办", "南京有什么好吃的"] * 2
    print(prompts)

    responses, historys = [], []
    
    model = llm.model(model_path)
    
    t0 = time.time()
    responses, historys = model.response_batch(prompts)        
    t1 = time.time()

    token_output_count = 0
    word_len = 0
    for i, res in enumerate(responses):
        tokens = model.tokenizer_encode_string(res)
        token_output_count += len(tokens)
        word_len += len(res)

        print("batch index: ", i)
        print(res)
        print("")

    print("\ntoken/s: {:.2f}, character/s: {:.2f}".format(token_output_count/(t1-t0), word_len/(t1-t0)))

