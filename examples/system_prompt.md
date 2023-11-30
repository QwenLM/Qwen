# 系统指令 (System Prompts)

## 什么是系统指令? (What is the System Prompts?)

系统指令设定了AI助手的行为模式，例如人物设定、语言风格、任务模式、甚至针对具体问题的具体行为。

System Propmts set the behavior mode of the AI assistant, such as character settings, language styles, task modes, and even specific behaviors for specific tasks.

系统指令可以是一个广泛的人物设定，如“You are a helpful assistant”；也可以是一个十分详细的要求，如“拒绝回答所有代码相关的问题”。

The System Prompts can be a broad character setting, such as "You are a helpful assistant"; or it can be a very detailed request, such as "Refuse to answer all code-related questions."

系统指令为用户提供了一个易组织、上下文稳定的控制AI助手行为的方式，可以从多种角度定制属于你自己的AI助手。

System Prompts provide users with an easy-to-organize, context-stable way to control the behavior of the AI assistant. You can customize your own AI assistant from multiple perspectives.

系统指令需要在多轮对话中稳定，例如角色扮演类系统指令被设定后AI助手不应该在多轮对话中跳脱自身的设定。

System Prompts need to be stable across multiple rounds of dialogue. For example, after a role-playing system prompt is set, the AI assistant should not escape its own settings in multiple rounds of dialogue.

同时，模型也需要具有基于系统指令中对自身行为进行推理的能力。这两者都是为模型赋予跟随系统指令能力时需要克服的难点。

At the same time, the model also needs to have the ability to reason about its own behavior based on system prompts. Both of these are difficulties that need to be overcome when giving the model the ability to follow system prompts.

Qwen-1.8B-Chat 和 Qwen-72B-Chat在多样且存在多轮复杂交互的系统指令上进行了充分训练，使模型可以跟随多样的系统指令，实现上下文(in-context)中的模型定制化，进一步提升了通义千问的可扩展性。

Qwen-1.8-Chat and Qwen-72B-Chat have been fully trained on diverse system prompts with multiple rounds of complex interactions, so that they can follow a variety of system prompts and realize model customization in context, further improving the scalability of Qwen-chat.

## 系统指令能做什么？ (What can System Prompts do?)

### 角色扮演 Role Play

在系统指令中告诉千问你需要它扮演的角色，即可沉浸式和该角色对话交流

Tell Qwen-Chat the role you want it to play in the System Prompt, and you can have an immersive conversation with that role.


![](../assets/system_prompt_role_play.png)

![](../assets/system_prompt_role_play_en.png)

### 语言风格 Language Style


简单调整千问的语言风格

Simple adjustment of the Qwen-Chat's language style

![](../assets/system_prompt_language_style.png)

![](../assets/system_prompt_language_style_en.png)

### 任务设定 Task Setting

指定具体任务，打造处理专项任务的千问模型

Setting specific tasks and creating a Qwen-Chat model to handle special tasks

![](../assets/system_prompt_task_setting.png)

![](../assets/system_prompt_task_setting_en.png)

### 行为设定 Behavior Setting

设定千问对具体任务的行为模式

Set behavior patterns of Qwen-Chat for specific tasks

![](../assets/system_prompt_behavior_setting.png)

![](../assets/system_prompt_behavior_setting_en.png)

## 代码示例 Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True)

# Only Qwen-72B-Chat and Qwen-1_8B-Chat has system prompt enhancement now.
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True).eval()
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-72B-Chat", device_map="auto", trust_remote_code=True).eval()

response, _ = model.chat(tokenizer, "你好呀", history=None, system="请用二次元可爱语气和我说话")
print(response)
# 你好啊！我是一只可爱的二次元猫咪哦，不知道你有什么问题需要我帮忙解答吗？

response, _ = model.chat(tokenizer, "My colleague works diligently", history=None, system="You will write beautiful compliments according to needs")
print(response)
# Your colleague is an outstanding worker! Their dedication and hard work are truly inspiring. They always go above and beyond to ensure that their tasks are completed on time and to the highest standard. I am lucky to have them as a colleague, and I know I can count on them to handle any challenge that comes their way.
```