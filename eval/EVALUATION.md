## 评测复现

- CEVAL

```Shell
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir data/ceval
mv ceval-exam.zip data/ceval
cd data/ceval; unzip ceval-exam.zip
cd ../../

# Qwen-7B
python evaluate_ceval.py -d data/ceval/

# Qwen-7B-Chat (We only provide 0-shot reproduction scripts. 5-shot results are obtained by OpenCompass (https://github.com/InternLM/opencompass).)
pip install thefuzz
python evaluate_chat_ceval.py -d data/ceval/
```

- MMLU

```Shell
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

# Qwen-7B
python evaluate_mmlu.py -d data/mmlu/data/

# Qwen-7B-Chat (We only provide 0-shot reproduction scripts. 5-shot results are obtained by OpenCompass (https://github.com/InternLM/opencompass).)
pip install thefuzz
python evaluate_chat_mmlu.py -d data/mmlu/data/
```

- CMMLU

```Shell
wget https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip
mkdir data/cmmlu
mv cmmlu_v1_0_1.zip data/cmmlu
cd data/cmmlu; unzip cmmlu_v1_0_1.zip
cd ../../

# Qwen-7B
python evaluate_cmmlu.py -d data/cmmlu/
```

- HumanEval

Get the HumanEval.jsonl file from [here](https://github.com/openai/human-eval/tree/master/data)

```Shell
git clone https://github.com/openai/human-eval
pip install -e human-eval

# Qwen-7B
python evaluate_humaneval.py -f HumanEval.jsonl -o HumanEval_res.jsonl
evaluate_functional_correctness HumanEval_res.jsonl
# Qwen-7B-Chat
python evaluate_chat_humaneval.py -f HumanEval.jsonl -o HumanEval_res_chat.jsonl
evaluate_functional_correctness HumanEval_res_chat.jsonl
```

When installing package human-eval, please note its following disclaimer:

This program exists to run untrusted model-generated code. Users are strongly encouraged not to do so outside of a robust security sandbox. The execution call in execution.py is deliberately commented out to ensure users read this disclaimer before running code in a potentially unsafe manner. See the comment in execution.py for more information and instructions.

- GSM8K

```Shell
# Qwen-7B
python evaluate_gsm8k.py

# Qwen-7B-Chat (We only provide 0-shot reproduction scripts. 5-shot results are obtained by OpenCompass (https://github.com/InternLM/opencompass).)
python evaluate_chat_gsm8k.py # zeroshot
```

- PLUGIN

This script is used to reproduce the results of the ReAct and Hugging Face Agent in the Tool Usage section of the README document.

```Shell
# Qwen-7B-Chat
mkdir data;
cd data;
## Old Evaluation Dataset (Version 20230803)
# wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/opensource_data/exam_plugin_v1/exam_plugin_v1_react_positive.jsonl;
# wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/opensource_data/exam_plugin_v1/exam_plugin_v1_react_negative.jsonl;
## New Evaluation Dataset (Version 20231206)
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/opensource_data/exam_plugin_v20231206/exam_plugin_v20231206_react_positive.jsonl;
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/opensource_data/exam_plugin_v20231206/exam_plugin_v20231206_react_negative.jsonl;cd ..;
pip install json5;
pip install jsonlines;
pip install rouge_score;
python evaluate_plugin.py --eval-react-positive --eval-react-negative --eval-hfagent
```
