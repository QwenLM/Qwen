#
# # Fine-tuning Script:
# Please start by reading the Fine-tuning section of README.md.
#
# # Fine-tuning Data Preparation:
# Then, if you would like to see some examples of how to prepare training samples for function calling,
# which is actually ReAct prompting under the hood, please read this file.

# # Inference Script:
# If you are interested in implementing function calling via ReAct prompting for inference,
# please refer to openai_api.py in our repository.
#
# If you have any questions, please raise an issue.
#

import json


def format_train_sample(messages):
    #
    # You do not need the `function` role, as Qwen's function calling is actually implemented via ReAct,
    # not by adding a `function` role or `function_call` message. See openai_api.py for details.
    #
    # If you need the `system` role, you might need to modify `finetune.py` accordingly.
    #
    assert set(m["role"] for m in messages) == {"user", "assistant"}

    sample = {
        "conversations": [
            {
                "from": m["role"],
                "value": m["content"],
            }
            for m in messages
        ]
    }
    return sample


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

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

Begin!"""


def build_react_instruction(functions):
    tools_text = []
    tools_name_text = []
    for func_info in functions:
        name = func_info.get("name", "")
        name_m = func_info.get("name_for_model", name)
        name_h = func_info.get("name_for_human", name)
        desc = func_info.get("description", "")
        desc_m = func_info.get("description_for_model", desc)
        tool = TOOL_DESC.format(
            name_for_model=name_m,
            name_for_human=name_h,
            description_for_model=desc_m,
            parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
        )
        tools_text.append(tool)
        tools_name_text.append(name_m)
    tools_text = "\n\n".join(tools_text)
    tools_name_text = ", ".join(tools_name_text)
    instruction = REACT_INSTRUCTION.format(
        tools_text=tools_text,
        tools_name_text=tools_name_text,
    )
    return instruction


def main():
    example_train_samples = []

    example_functions = [
        {
            "name_for_human": "Google Search",
            "name_for_model": "google_search",
            "description_for_model": "Google Search is a general search engine that can be used to access the internet,"
            + " query encyclopedia knowledge, and stay informed about current events."
            + " Format the arguments as a JSON object.",  # If you expect `Action Input` to be a JSON.
            "parameters": [
                {
                    "name": "search_query",
                    "description": "Search keywords or phrases",
                    "required": True,  # Set to False if it is an optional parameter.
                    "schema": {"type": "string"},
                },
                # You can add more parameters to this `parameters` list if you wish.
            ],
        },
        {
            "name_for_human": "Code Interpreter",
            "name_for_model": "code_interpreter",
            "description_for_model": "Code interpreter that can execute Python code."
            + "Enclose the code within triple backticks (`)"
            + " at the beginning and end of the code.",  # If you expect `Action Input` to be a Markdown code block.
            "parameters": [
                {
                    "name": "code",
                    "description": "Code to be executed",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
        },
    ]
    example_instruction = build_react_instruction(example_functions)

    # This example has multiple actions in one single response.
    example_train_samples.append(
        [
            {
                "role": "user",
                "content": f"{example_instruction}\n\nQuestion: Who is jay chou? And who is his wife?",
            },
            {
                "role": "assistant",
                "content": """
Thought: I need to google Jay Chou.
Action: google_search
Action Input: {"search_query": "Jay Chou"}
Observation: Jay Chou is ...
Thought: I need to google Jay's wife.
Action: google_search
Action Input: {"search_query": "Jay Chou's wife"}
Observation: Jay Chou's wife is ...
Thought: I now know the final answer
Final Answer: Jay Chou is ... His wife is ...
                """.strip(),
            },
        ]
    )

    # This example involves multiple rounds of conversation.
    example_train_samples.append(
        [
            # Round #1
            {
                "role": "user",
                "content": f"{example_instruction}\n\nQuestion: 123+456=?",
            },
            {
                "role": "assistant",
                "content": """
Thought: I need to compute the result using Code Interpreter.
Action: code_interpreter
Action Input:
```py
123 + 456
```
Observation: 579
Thought: I now know the final answer
Final Answer: 579
                """.strip(),
            },
            # Round #2
            {
                "role": "user",
                "content": "Multiply the result by 2.",
            },
            {
                "role": "assistant",
                "content": """
Thought: Code Interpreter is helpful for answering this question.
Action: code_interpreter
Action Input:
```py
579 * 2
```
Observation: 1158
Thought: I now know the final answer
Final Answer: 1158
                        """.strip(),
            },
            # Round #3
            {
                "role": "user",
                "content": "You are so smart, Qwen.",  # No action is needed for this question.
            },
            {
                "role": "assistant",
                "content": """
Thought: I now know the final answer
Final Answer: Thank you.
                        """.strip(),
            },
            # Round #4
            {
                "role": "user",
                "content": "Please re-execute the code for computing my first question again.",
            },
            {
                "role": "assistant",
                "content": """
Thought: I need to re-compute the result.
Action: code_interpreter
Action Input:
```py
123 + 456
```
Observation: 579
Thought: I now know the final answer
Final Answer: 579
                        """.strip(),
            },
        ]
    )

    example_train_samples = [format_train_sample(x) for x in example_train_samples]
    with open(
        "example_func_call_train_samples.json", "w"
    ) as fout:  # data for fine-tuning
        fout.write(json.dumps(example_train_samples, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
