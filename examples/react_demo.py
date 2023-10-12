#
# 相关材料：
#   ReAct Prompting 原理简要介绍，不包含代码实现：
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_prompt.md
#   基于 model.chat 接口（对话模式）的 ReAct Prompting 实现（含接入 LangChain 的工具实现）：
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/langchain_tooluse.ipynb
#   基于 model.generate 接口（续写模式）的 ReAct Prompting 实现，比 chat 模式的实现更复杂些：
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_demo.py（本文件）
#

import json
import os

import json5
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

for _ in range(10):  # 网络不稳定，多试几次
    try:
        name = 'Qwen/Qwen-7B-Chat'
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, device_map="auto", trust_remote_code=True
        ).eval()
        model.generation_config = generation_config
        model.generation_config.top_k = 1
        break
    except Exception as e:
        print(e)

# 将一个插件的关键信息拼接成一段文本的模版。
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

# ReAct prompting 的 instruction 模版，将包含插件的详细信息。
PROMPT_REACT = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""


#
# 本示例代码的入口函数。
#
# 输入：
#   prompt: 用户的最新一个问题。
#   history: 用户与模型的对话历史，是一个 list，
#       list 中的每个元素为 {"user": "用户输入", "bot": "模型输出"} 的一轮对话。
#       最新的一轮对话放 list 末尾。不包含最新一个问题。
#   list_of_plugin_info: 候选插件列表，是一个 list，list 中的每个元素为一个插件的关键信息。
#       比如 list_of_plugin_info = [plugin_info_0, plugin_info_1, plugin_info_2]，
#       其中 plugin_info_0, plugin_info_1, plugin_info_2 这几个样例见本文档前文。
#
# 输出：
#   模型对用户最新一个问题的回答。
#
def llm_with_plugin(prompt: str, history, list_of_plugin_info=()):
    chat_history = [(x['user'], x['bot']) for x in history] + [(prompt, '')]

    # 需要让模型进行续写的初始文本
    planning_prompt = build_input_text(chat_history, list_of_plugin_info)

    text = ''
    while True:
        output = text_completion(planning_prompt + text, stop_words=['Observation:', 'Observation:\n'])
        action, action_input, output = parse_latest_plugin_call(output)
        if action:  # 需要调用插件
            # action、action_input 分别为需要调用的插件代号、输入参数
            # observation是插件返回的结果，为字符串
            observation = call_plugin(action, action_input)
            output += f'\nObservation: {observation}\nThought:'
            text += output
        else:  # 生成结束，并且不再需要调用插件
            text += output
            break

    new_history = []
    new_history.extend(history)
    new_history.append({'user': prompt, 'bot': text})
    return text, new_history


# 将对话历史、插件信息聚合成一段初始文本
def build_input_text(chat_history, list_of_plugin_info) -> str:
    # 候选插件的详细信息
    tools_text = []
    for plugin_info in list_of_plugin_info:
        tool = TOOL_DESC.format(
            name_for_model=plugin_info["name_for_model"],
            name_for_human=plugin_info["name_for_human"],
            description_for_model=plugin_info["description_for_model"],
            parameters=json.dumps(plugin_info["parameters"], ensure_ascii=False),
        )
        if plugin_info.get('args_format', 'json') == 'json':
            tool += " Format the arguments as a JSON object."
        elif plugin_info['args_format'] == 'code':
            tool += ' Enclose the code within triple backticks (`) at the beginning and end of the code.'
        else:
            raise NotImplementedError
        tools_text.append(tool)
    tools_text = '\n\n'.join(tools_text)

    # 候选插件的代号
    tools_name_text = ', '.join([plugin_info["name_for_model"] for plugin_info in list_of_plugin_info])

    im_start = '<|im_start|>'
    im_end = '<|im_end|>'
    prompt = f'{im_start}system\nYou are a helpful assistant.{im_end}'
    for i, (query, response) in enumerate(chat_history):
        if list_of_plugin_info:  # 如果有候选插件
            # 倒数第一轮或倒数第二轮对话填入详细的插件信息，但具体什么位置填可以自行判断
            if (len(chat_history) == 1) or (i == len(chat_history) - 2):
                query = PROMPT_REACT.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                    query=query,
                )
        query = query.lstrip('\n').rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        response = response.lstrip('\n').rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        # 使用续写模式（text completion）时，需要用如下格式区分用户和AI：
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"

    assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
    prompt = prompt[: -len(f'{im_end}')]
    return prompt


def text_completion(input_text: str, stop_words) -> str:  # 作为一个文本续写模型来使用
    im_end = '<|im_end|>'
    if im_end not in stop_words:
        stop_words = stop_words + [im_end]
    stop_words_ids = [tokenizer.encode(w) for w in stop_words]

    # TODO: 增加流式输出的样例实现
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to(model.device)
    output = model.generate(input_ids, stop_words_ids=stop_words_ids)
    output = output.tolist()[0]
    output = tokenizer.decode(output, errors="ignore")
    assert output.startswith(input_text)
    output = output[len(input_text) :].replace('<|endoftext|>', '').replace(im_end, '')

    for stop_str in stop_words:
        idx = output.find(stop_str)
        if idx != -1:
            output = output[: idx + len(stop_str)]
    return output  # 续写 input_text 的结果，不包含 input_text 的内容


def parse_latest_plugin_call(text):
    plugin_name, plugin_args = '', ''
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
        k = text.rfind('\nObservation:')
        plugin_name = text[i + len('\nAction:') : j].strip()
        plugin_args = text[j + len('\nAction Input:') : k].strip()
        text = text[:k]
    return plugin_name, plugin_args, text


#
# 输入：
#   plugin_name: 需要调用的插件代号，对应 name_for_model。
#   plugin_args：插件的输入参数，是一个 dict，dict 的 key、value 分别为参数名、参数值。
# 输出：
#   插件的返回结果，需要是字符串。
#   即使原本是 JSON 输出，也请 json.dumps(..., ensure_ascii=False) 成字符串。
#
def call_plugin(plugin_name: str, plugin_args: str) -> str:
    #
    # 请开发者自行完善这部分内容。这里的参考实现仅是 demo 用途，非生产用途。
    #
    if plugin_name == 'google_search':
        # 使用 SerpAPI 需要在这里填入您的 SERPAPI_API_KEY！
        os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY", default='')
        from langchain import SerpAPIWrapper

        return SerpAPIWrapper().run(json5.loads(plugin_args)['search_query'])
    elif plugin_name == 'image_gen':
        import urllib.parse

        prompt = json5.loads(plugin_args)["prompt"]
        prompt = urllib.parse.quote(prompt)
        return json.dumps({'image_url': f'https://image.pollinations.ai/prompt/{prompt}'}, ensure_ascii=False)
    else:
        raise NotImplementedError


def test():
    tools = [
        {
            'name_for_human': '谷歌搜索',
            'name_for_model': 'google_search',
            'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
            'parameters': [
                {
                    'name': 'search_query',
                    'description': '搜索关键词或短语',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '文生图',
            'name_for_model': 'image_gen',
            'description_for_model': '文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL',
            'parameters': [
                {
                    'name': 'prompt',
                    'description': '英文关键词，描述了希望图像具有什么内容',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
    ]
    history = []
    for query in ['你好', '搜索一下谁是周杰伦', '再搜下他老婆是谁', '给我画个可爱的小猫吧，最好是黑猫']:
        print(f"User's Query:\n{query}\n")
        response, history = llm_with_plugin(prompt=query, history=history, list_of_plugin_info=tools)
        print(f"Qwen's Response:\n{response}\n")


if __name__ == "__main__":
    test()

"""如果执行成功，在终端下应当能看到如下输出：
User's Query:
你好

Qwen's Response:
Thought: 提供的工具对回答该问题帮助较小，我将不使用工具直接作答。
Final Answer: 你好！很高兴见到你。有什么我可以帮忙的吗？

User's Query:
搜索一下谁是周杰伦

Qwen's Response:
Thought: 我应该使用Google搜索查找相关信息。
Action: google_search
Action Input: {"search_query": "周杰伦"}
Observation: Jay Chou is a Taiwanese singer, songwriter, record producer, rapper, actor, television personality, and businessman.
Thought: I now know the final answer.
Final Answer: 周杰伦（Jay Chou）是一位来自台湾的歌手、词曲创作人、音乐制作人、说唱歌手、演员、电视节目主持人和企业家。他以其独特的音乐风格和才华在华语乐坛享有很高的声誉。

User's Query:
再搜下他老婆是谁

Qwen's Response:
Thought: 我应该使用Google搜索查找相关信息。
Action: google_search
Action Input: {"search_query": "周杰伦 老婆"}
Observation: Hannah Quinlivan
Thought: I now know the final answer.
Final Answer: 周杰伦的老婆是Hannah Quinlivan，她是一位澳大利亚籍的模特和演员。两人于2015年结婚，并育有一子。

User's Query:
给我画个可爱的小猫吧，最好是黑猫

Qwen's Response:
Thought: 我应该使用文生图API来生成一张可爱的小猫图片。
Action: image_gen
Action Input: {"prompt": "cute black cat"}
Observation: {"image_url": "https://image.pollinations.ai/prompt/cute%20black%20cat"}
Thought: I now know the final answer.
Final Answer: 生成的可爱小猫图片的URL为https://image.pollinations.ai/prompt/cute%20black%20cat。你可以点击这个链接查看图片。
"""
