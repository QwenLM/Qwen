# 运行方式：python auto_comments.py --path 'path of file or folder'
# 脚本功能：使用QWen-7B-Chat为提供的代码文件自动生成注释。(详见auto_comments.md)


import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MaxLine = 50 # 限制单次处理最大代码行数
SplitKey = ["\ndef "] # 自定义的切分代码标识
CodeFileType = ["py"] # 目前仅测试过对python文件生成注释

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='Qwen-7B/eval/evaluate_ceval.py')
    parser.add_argument('--regenerate', action='store_true', default=False) #如果已经生成过注释，默认不会重新生成
    args = parser.parse_args()
    return args

class QWenChat():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

        # use bf16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
        # use auto mode, automatically select precision based on the device.
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
        
        # Specify hyperparameters for generation
        self.model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
        self.history = None
        
    def chat(self, query, system = ""):

        # use history
        # response, history = self.model.chat(self.tokenizer, query, history=self.history)

        # 默认不使用history
        response, history = self.model.chat(self.tokenizer, query, history=None)
        self.history = history

        return response
# 生成注释
def gen_code_comments(context, model = None, **kwargs):
    prompt = "\n为以上代码生成细致的中文注释，注意使用合适的语法。要求必须在每个函数开头生成一段统一的函数功能注释。\n除了注释，请保证原始代码内容不变。不要返回除了注释和代码以外的其余信息，不要生成额外代码。\n"
    return model.chat(context + prompt)

def read_file(path):
    f = open(path, "r",encoding='utf-8')
    lines = f.readlines()
    return "".join(lines)

def write_file(path, context):
    with open(path,'w') as f:
        f.write(context)

# 如果代码文件过长，可以简单按照最大行数切分代码
def split_context_by_maxline(text):
    lines = text.split("\n")
    lines_len = len(lines)
    res = []
    for i in range(MaxLine, lines_len, MaxLine):
        res.append("\n".join(lines[i-MaxLine:i]))

    if i < lines_len:
        res.append("\n".join(lines[i:]))
    return res

# 如果代码文件过长，可以简单按照函数切分代码
def split_context_by_splitkey(text):
    blocks = text.split(SplitKey[0])
    return [blocks[0]] + [SplitKey[0]+x for x in blocks[1:]]

# merge原始代码和生成的注释，目的是保证原始代码不被更改。这部分可以使用各种不同的策略处理。
def merge_code_and_comments(original_file, comments_path):
    res = []
    ori_f = open(original_file, "r",encoding='utf-8')
    ori_lines = ori_f.readlines()

    com_f = open(comments_path, "r",encoding='utf-8')
    com_lines = com_f.readlines()
    len_com_lines = len(com_lines)
    p = 0
    j = 0
    for i, line in enumerate(ori_lines):
        if line.isspace():
            continue
        if line.strip()[0] == '#':
            res.append(line)
            continue
        while j < len_com_lines and line[:-1] not in com_lines[j]:
            j += 1
        if j < len_com_lines:
            p = j - 1
            up_comments = []
            triple_dot_flag = 0
            while p < j:
                if p < 0 or (res and res[-1] and com_lines[p] == res[-1]):
                    break
                if com_lines[p].strip() and (len(com_lines[p].strip())>3 and com_lines[p].strip()[-3:] == '"""' and com_lines[p].strip()[:3] == '"""') or (len(com_lines[p].strip())>3 and com_lines[p].strip()[-3:] == "'''" and com_lines[p].strip()[:3] == "'''"):
                    up_comments.append(com_lines[p])
                    p -= 1
                    continue
                if com_lines[p].strip() and (com_lines[p].strip()[-3:] == '"""' or com_lines[p].strip()[:3] == '"""' or com_lines[p].strip()[-3:] == "'''" or com_lines[p].strip()[:3] == "'''"):
                    triple_dot_flag = (triple_dot_flag + 1)%2
                    up_comments.append(com_lines[p])
                    p -= 1
                    continue
                if triple_dot_flag:
                    up_comments.append(com_lines[p])
                    p -= 1
                    continue
                if (com_lines[p].strip()=="") or (com_lines[p].strip() and com_lines[p].strip()[0] == '#' and "省略部分内容" not in com_lines[p]):
                    up_comments.append(com_lines[p])
                else:
                    break
                p -= 1
            if up_comments:
                res.extend(reversed(up_comments))
            if "#" in com_lines[j] and "#" not in line:
                in_line_comments = "  #" + com_lines[j].split("#")[-1]
                res.append(line[:-1]+in_line_comments)
            else:
                res.append(line)
            p = j+1
        else:
            res.append(line)
            j = p

    write_file(comments_path, "".join(res))

# 处理单个文件
def deal_one_file(model, path, args):
    context = read_file(path)

    fname = path.split("/")[-1]
    fpath = "/".join(path.split("/")[:-1])
    outfname = fname.split(".")[0]+"_comments."+fname.split(".")[-1]

    comments_path = os.path.join(fpath, outfname)
    if (not args.regenerate) and os.path.exists(comments_path):
        print("use cache: ", comments_path)
        return

    context_line = len(context.split("\n"))
    if context_line < MaxLine:
        res = gen_code_comments(context, model = model)
    elif SplitKey[0] not in context:
        context_list = split_context_by_maxline(context)
        res = "\n".join([gen_code_comments(context_block, model = model) for context_block in context_list])
    else:
        context_list = split_context_by_splitkey(context)
        res = "\n".join([gen_code_comments(context_block, model = model) for context_block in context_list])

    write_file(comments_path, res)
    merge_code_and_comments(path, comments_path)

# 处理文件夹
def deal_folder(model, path, args):
    for fl in os.listdir(path):
        now_path = os.path.join(path, fl)
        if os.path.isfile(now_path):
            if (now_path.split(".")[-1] in CodeFileType) and ("_comments" not in now_path):
                deal_one_file(model, now_path, args)
        elif os.path.isdir(now_path):
            deal_folder(model, now_path, args)
        else:
            print("Please specify a correct path!")

def transfer(args):
    model = QWenChat()

    if os.path.isfile(args.path):
        if (args.path.split(".")[-1] in CodeFileType) and ("_comments" not in args.path):
            deal_one_file(model, args.path, args)
    elif os.path.isdir(args.path):
        deal_folder(model, args.path, args)
    else:
        print("Please specify a correct path!")

if __name__ == '__main__':
    args = parse_args()
    print(args)
    transfer(args)
