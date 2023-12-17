
import pydantic
import json
import re

from copy import deepcopy
from enum import Enum, IntEnum
from pydantic import BaseModel
from loguru import logger
from typing import (
    Dict, 
    List, 
    Any, 
    Literal, 
    Optional, 
    Union,
    cast,
    Type,
    Tuple
)

from transformers import PreTrainedTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastapi import HTTPException

from openai.types.model import Model
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.create_embedding_response import Usage


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

_TEXT_COMPLETION_CMD = object()

# --------------- Pydantic v2 compatibility ---------------

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")


def model_json(model: pydantic.BaseModel, **kwargs) -> str:
    if PYDANTIC_V2:
        return model.model_dump_json(**kwargs)
    return model.json(**kwargs)  # type: ignore


def model_dump(model: pydantic.BaseModel, **kwargs) -> Dict[str, Any]:
    if PYDANTIC_V2:
        return model.model_dump(**kwargs)
    return cast(
        "dict[str, Any]",
        model.dict(**kwargs),
    )


def model_parse(model: Type[pydantic.BaseModel], data: Any) -> pydantic.BaseModel:
    if PYDANTIC_V2:
        return model.model_validate(data)
    return model.parse_obj(data)  # pyright: ignore[reportDeprecated]


def disable_warnings(model: Type[pydantic.BaseModel]):
    # Disable warning for model_name settings
    if PYDANTIC_V2:
        model.model_config["protected_namespaces"] = ()

def parse_messages(
    messages: List[ChatCompletionMessageParam], split_role="user"
) -> Tuple[str, List[List[ChatCompletionMessageParam]]]:
    """
    Parse a list of chat completion messages into system and rounds.

    Args:
        messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
        split_role: The role at which to split the rounds. Defaults to Role.USER.

    Returns:
        Tuple[str, List[List[ChatCompletionMessageParam]]]: A tuple containing the system message and a list of rounds.
    """
    system, rounds = "", []
    r = []
    for i, message in enumerate(messages):
        if message["role"] == "system":
            system = message["content"]
            continue
        if message["role"] == split_role and r:
            rounds.append(r)
            r = []
        r.append(message)
    if r:
        rounds.append(r)
    return system, rounds


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model] = []


class ChatCompletionCreateParams(BaseModel):
    messages: List[ChatCompletionMessageParam]
    """A list of messages comprising the conversation so far.

    [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).
    """

    model: str
    """ID of the model to use.

    See the
    [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility)
    table for details on which models work with the Chat API.
    """

    frequency_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on their existing frequency in the
    text so far, decreasing the model's likelihood to repeat the same line verbatim.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    function_call: Optional[FunctionCall] = None
    """Deprecated in favor of `tool_choice`.

    Controls which (if any) function is called by the model. `none` means the model
    will not call a function and instead generates a message. `auto` means the model
    can pick between generating a message or calling a function. Specifying a
    particular function via `{"name": "my_function"}` forces the model to call that
    function.

    `none` is the default when no functions are present. `auto`` is the default if
    functions are present.
    """

    functions: Optional[List] = None
    """Deprecated in favor of `tools`.

    A list of functions the model may generate JSON inputs for.
    """

    logit_bias: Optional[Dict[str, int]] = None
    """Modify the likelihood of specified tokens appearing in the completion.

    Accepts a JSON object that maps tokens (specified by their token ID in the
    tokenizer) to an associated bias value from -100 to 100. Mathematically, the
    bias is added to the logits generated by the model prior to sampling. The exact
    effect will vary per model, but values between -1 and 1 should decrease or
    increase likelihood of selection; values like -100 or 100 should result in a ban
    or exclusive selection of the relevant token.
    """

    max_tokens: Optional[int] = None
    """The maximum number of [tokens](/tokenizer) to generate in the chat completion.

    The total length of input tokens and generated tokens is limited by the model's
    context length.
    [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
    for counting tokens.
    """

    n: Optional[int] = 1
    """How many chat completion choices to generate for each input message."""

    presence_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on whether they appear in the text so
    far, increasing the model's likelihood to talk about new topics.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    response_format: Optional[ResponseFormat] = None
    """An object specifying the format that the model must output.

    Used to enable JSON mode.
    """

    seed: Optional[int] = None
    """This feature is in Beta.

    If specified, our system will make a best effort to sample deterministically,
    such that repeated requests with the same `seed` and parameters should return
    the same result. Determinism is not guaranteed, and you should refer to the
    `system_fingerprint` response parameter to monitor changes in the backend.
    """

    stop: Optional[Union[str, List[str]]] = None
    """Up to 4 sequences where the API will stop generating further tokens."""

    temperature: Optional[float] = 0.9
    """What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic.

    We generally recommend altering this or `top_p` but not both.
    """

    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    """
    Controls which (if any) function is called by the model. `none` means the model
    will not call a function and instead generates a message. `auto` means the model
    can pick between generating a message or calling a function. Specifying a
    particular function via
    `{"type: "function", "function": {"name": "my_function"}}` forces the model to
    call that function.

    `none` is the default when no functions are present. `auto` is the default if
    functions are present.
    """

    tools: Optional[List] = None
    """A list of tools the model may call.

    Currently, only functions are supported as a tool. Use this to provide a list of
    functions the model may generate JSON inputs for.
    """

    top_p: Optional[float] = 1.0
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """

    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor
    and detect abuse.
    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    """

    stream: Optional[bool] = False
    """If set, partial message deltas will be sent, like in ChatGPT.

    Tokens will be sent as data-only
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
    as they become available, with the stream terminated by a `data: [DONE]`
    message.
    [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
    """

    # Addictional parameters
    repetition_penalty: Optional[float] = 1.03
    """The parameter for repetition penalty. 1.0 means no penalty.
    See[this paper](https://arxiv.org / pdf / 1909.05858.pdf) for more details.
    """

    typical_p: Optional[float] = None
    """Typical Decoding mass.
    See[Typical Decoding for Natural Language Generation](https://arxiv.org / abs / 2202.00666) for more information
    """

    watermark: Optional[bool] = False
    """Watermarking with [A Watermark for Large Language Models](https://arxiv.org / abs / 2301.10226)
    """

    best_of: Optional[int] = 1

    ignore_eos: Optional[bool] = False

    use_beam_search: Optional[bool] = False

    stop_token_ids: Optional[List[int]] = None

    skip_special_tokens: Optional[bool] = True

    spaces_between_special_tokens: Optional[bool] = True

    min_p: Optional[float] = 0.0


class CompletionCreateParams(BaseModel):
    model: str
    """ID of the model to use.

    You can use the
    [List models](https://platform.openai.com/docs/api-reference/models/list) API to
    see all of your available models, or see our
    [Model overview](https://platform.openai.com/docs/models/overview) for
    descriptions of them.
    """

    prompt: Union[str, List[str], List[int], List[List[int]], None]
    """
    The prompt(s) to generate completions for, encoded as a string, array of
    strings, array of tokens, or array of token arrays.

    Note that <|endoftext|> is the document separator that the model sees during
    training, so if a prompt is not specified the model will generate as if from the
    beginning of a new document.
    """

    best_of: Optional[int] = 1
    """
    Generates `best_of` completions server-side and returns the "best" (the one with
    the highest log probability per token). Results cannot be streamed.

    When used with `n`, `best_of` controls the number of candidate completions and
    `n` specifies how many to return – `best_of` must be greater than `n`.

    **Note:** Because this parameter generates many completions, it can quickly
    consume your token quota. Use carefully and ensure that you have reasonable
    settings for `max_tokens` and `stop`.
    """

    echo: Optional[bool] = False
    """Echo back the prompt in addition to the completion"""

    frequency_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on their existing frequency in the
    text so far, decreasing the model's likelihood to repeat the same line verbatim.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    logit_bias: Optional[Dict[str, int]] = None
    """Modify the likelihood of specified tokens appearing in the completion.

    Accepts a JSON object that maps tokens (specified by their token ID in the GPT
    tokenizer) to an associated bias value from -100 to 100. You can use this
    [tokenizer tool](/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to
    convert text to token IDs. Mathematically, the bias is added to the logits
    generated by the model prior to sampling. The exact effect will vary per model,
    but values between -1 and 1 should decrease or increase likelihood of selection;
    values like -100 or 100 should result in a ban or exclusive selection of the
    relevant token.

    As an example, you can pass `{"50256": -100}` to prevent the <|endoftext|> token
    from being generated.
    """

    logprobs: Optional[int] = None
    """
    Include the log probabilities on the `logprobs` most likely tokens, as well the
    chosen tokens. For example, if `logprobs` is 5, the API will return a list of
    the 5 most likely tokens. The API will always return the `logprob` of the
    sampled token, so there may be up to `logprobs+1` elements in the response.

    The maximum value for `logprobs` is 5.
    """

    max_tokens: Optional[int] = 16
    """The maximum number of [tokens](/tokenizer) to generate in the completion.

    The token count of your prompt plus `max_tokens` cannot exceed the model's
    context length.
    [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
    for counting tokens.
    """

    n: Optional[int] = 1
    """How many completions to generate for each prompt.

    **Note:** Because this parameter generates many completions, it can quickly
    consume your token quota. Use carefully and ensure that you have reasonable
    settings for `max_tokens` and `stop`.
    """

    presence_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on whether they appear in the text so
    far, increasing the model's likelihood to talk about new topics.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    seed: Optional[int] = None
    """
    If specified, our system will make a best effort to sample deterministically,
    such that repeated requests with the same `seed` and parameters should return
    the same result.

    Determinism is not guaranteed, and you should refer to the `system_fingerprint`
    response parameter to monitor changes in the backend.
    """

    stop: Optional[Union[str, List[str]]] = None
    """Up to 4 sequences where the API will stop generating further tokens.

    The returned text will not contain the stop sequence.
    """

    suffix: Optional[str] = None
    """The suffix that comes after a completion of inserted text."""

    temperature: Optional[float] = 1.
    """What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic.

    We generally recommend altering this or `top_p` but not both.
    """

    top_p: Optional[float] = 1.
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """

    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor
    and detect abuse.
    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    """

    stream: Optional[bool] = False
    """If set, partial message deltas will be sent, like in ChatGPT.

    Tokens will be sent as data-only
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
    as they become available, with the stream terminated by a `data: [DONE]`
    message.
    [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
    """

    # Addictional parameters
    repetition_penalty: Optional[float] = 1.03
    """The parameter for repetition penalty. 1.0 means no penalty.
    See[this paper](https://arxiv.org / pdf / 1909.05858.pdf) for more details.
    """

    typical_p: Optional[float] = None
    """Typical Decoding mass.
    See[Typical Decoding for Natural Language Generation](https://arxiv.org / abs / 2202.00666) for more information
    """

    watermark: Optional[bool] = False
    """Watermarking with [A Watermark for Large Language Models](https://arxiv.org / abs / 2301.10226)
    """

    ignore_eos: Optional[bool] = False

    use_beam_search: Optional[bool] = False

    stop_token_ids: Optional[List[int]] = None

    skip_special_tokens: Optional[bool] = True

    spaces_between_special_tokens: Optional[bool] = True

    min_p: Optional[float] = 0.0


class EmbeddingCreateParams(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    """Input text to embed, encoded as a string or array of tokens.

    To embed multiple inputs in a single request, pass an array of strings or array
    of token arrays. The input must not exceed the max input tokens for the model
    (8192 tokens for `text-embedding-ada-002`) and cannot be an empty string.
    [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
    for counting tokens.
    """

    model: str
    """ID of the model to use.

    You can use the
    [List models](https://platform.openai.com/docs/api-reference/models/list) API to
    see all of your available models, or see our
    [Model overview](https://platform.openai.com/docs/models/overview) for
    descriptions of them.
    """

    encoding_format: Literal["float", "base64"] = "float"
    """The format to return the embeddings in.

    Can be either `float` or [`base64`](https://pypi.org/project/pybase64/).
    """

    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor
    and detect abuse.
    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    """


class Embedding(BaseModel):
    embedding: Any
    """The embedding vector, which is a list of floats.

    The length of vector depends on the model as listed in the
    [embedding guide](https://platform.openai.com/docs/guides/embeddings).
    """

    index: int
    """The index of the embedding in the list of embeddings."""

    object: Literal["embedding"]
    """The object type, which is always "embedding"."""


class CreateEmbeddingResponse(BaseModel):
    data: List[Embedding]
    """The list of embeddings generated by the model."""

    model: str
    """The name of the model used to generate the embedding."""

    object: Literal["list"]
    """The object type, which is always "list"."""

    usage: Usage
    """The usage information for the request."""


def build_qwen_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatCompletionMessageParam],
    context_len: int = 8192,
    max_new_tokens: int = 256,
    functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """
    Builds the input tokens for Qwen chat generation.

    Refs:
        https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py

    Args:
        tokenizer: The tokenizer used to encode the input tokens.
        messages: The list of chat messages.
        context_len: The maximum length of the context.
        max_new_tokens: The maximum number of new tokens to add.
        functions: Optional dictionary or list of dictionaries representing the functions.
        tools: Optional list of dictionaries representing the tools.

    Returns:
        The list of input tokens.
    """
    query, history = process_qwen_messages(messages, functions, tools)
    if query is _TEXT_COMPLETION_CMD:
        return build_last_message_input(tokenizer, history)

    messages = []
    for q, r in history:
        messages.extend(
            [
                ChatCompletionUserMessageParam(role="user", content=q),
                ChatCompletionAssistantMessageParam(role="assistant", content=r)
            ]
        )
    messages.append(ChatCompletionUserMessageParam(role="user", content=query))

    max_input_tokens = context_len - max_new_tokens
    system, rounds = parse_messages(messages)
    system = f"You are a helpful assistant.{system}"

    im_start_tokens, im_end_tokens = [tokenizer.im_start_id], [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        return tokenizer.encode(
            role, allowed_special=set()
        ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

    system_tokens_part = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for r in rounds[::-1]:
        round_tokens = []
        for message in r:
            if round_tokens:
                round_tokens += nl_tokens

            if message["role"] == Role.USER:
                content_tokens = im_start_tokens + _tokenize_str("user", message["content"]) + im_end_tokens
            else:
                content_tokens = im_start_tokens + _tokenize_str("assistant", message["content"]) + im_end_tokens

            round_tokens.extend(content_tokens)

        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            if history_tokens:
                history_tokens = nl_tokens + history_tokens

            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + nl_tokens + history_tokens
    if messages[-1]["role"] != Role.ASSISTANT:
        input_tokens += nl_tokens + im_start_tokens + tokenizer.encode("assistant") + nl_tokens
    return input_tokens[-max_input_tokens:]  # truncate left



def process_qwen_messages(
    messages: List[ChatCompletionMessageParam],
    functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[List[str]]]:
    """
    Process the Qwen messages and generate a query and history.

    Args:
        messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
        functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]): The functions to be used.
        tools (Optional[List[Dict[str, Any]]]): The tools to be used.

    Returns:
        Tuple[str, List[List[str]]]: The generated query and history.
    """
    if all(m["role"] != Role.USER for m in messages):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting at least one user message.",
        )

    messages = deepcopy(messages)
    default_system = "You are a helpful assistant."
    system = ""
    if messages[0]["role"] == Role.SYSTEM:
        system = messages.pop(0)["content"].lstrip("\n").rstrip()
        if system == default_system:
            system = ""

    if tools:
        functions = [t["function"] for t in tools]

    if functions:
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
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            )

            tools_text.append(tool)
            tools_name_text.append(name_m)

        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        system += "\n\n" + REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        )
        system = system.lstrip("\n").rstrip()

    dummy_thought = {
        "en": "\nThought: I now know the final answer.\nFinal answer: ",
        "zh": "\nThought: 我会作答了。\nFinal answer: ",
    }

    _messages = messages
    messages = []
    for m_idx, m in enumerate(_messages):
        role, content = m["role"], m["content"]
        func_call, tools_call = m.get("function_call", None), m.get("tools_call", None)
        if content:
            content = content.lstrip("\n").rstrip()
        if role in [Role.FUNCTION, Role.TOOL]:
            if (len(messages) == 0) or (messages[-1]["role"] != Role.ASSISTANT):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role assistant before role function.",
                )
            messages[-1]["content"] += f"\nObservation: {content}"
            if m_idx == len(_messages) - 1:
                messages[-1]["content"] += "\nThought:"
        elif role == Role.ASSISTANT:
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role user before role assistant.",
                )
            last_msg = messages[-1]["content"]
            last_msg_has_zh = len(re.findall(r"[\u4e00-\u9fff]+", last_msg)) > 0

            if func_call is None and tools_call is None:
                if functions or tools_call:
                    content = dummy_thought["zh" if last_msg_has_zh else "en"] + content
            else:
                if func_call:
                    f_name, f_args = func_call.get("name"), func_call.get("arguments")
                else:
                    f_name, f_args = tools_call[0]["function"]["name"], tools_call[0]["function"]["arguments"]
                if not content:
                    if last_msg_has_zh:
                        content = f"Thought: 我可以使用 {f_name} API。"
                    else:
                        content = f"Thought: I can use {f_name}."

            if messages[-1]["role"] == Role.USER:
                messages.append(
                    ChatCompletionAssistantMessageParam(role="assistant", content=content.lstrip("\n").rstrip())
                )
            else:
                messages[-1]["content"] += content
        elif role == Role.USER:
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=content.lstrip("\n").rstrip())
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid request: Incorrect role {role}."
            )

    query = _TEXT_COMPLETION_CMD
    if messages[-1]["role"] == Role.USER:
        query = messages[-1]["content"]
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise HTTPException(status_code=400, detail="Invalid request")

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i]["role"] == Role.USER and messages[i + 1]["role"] == Role.ASSISTANT:
            usr_msg = messages[i]["content"].lstrip("\n").rstrip()
            bot_msg = messages[i + 1]["content"].lstrip("\n").rstrip()
            if system and (i == len(messages) - 2):
                usr_msg = f"{system}\n\nQuestion: {usr_msg}"
                system = ""
            for t in dummy_thought.values():
                t = t.lstrip("\n")
                if bot_msg.startswith(t) and ("\nAction: " in bot_msg):
                    bot_msg = bot_msg[len(t):]
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Expecting exactly one user (or function) role before every assistant role.",
            )
    if system:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{system}\n\nQuestion: {query}"
    return query, history


def build_last_message_input(tokenizer: PreTrainedTokenizer, history: list):
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
    for i, (query, response) in enumerate(history):
        query = query.lstrip("\n").rstrip()
        response = response.lstrip("\n").rstrip()
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"
    prompt = prompt[:-len(im_end)]
    logger.debug(f"==== Prompt with tools ====\n{prompt}")
    return tokenizer.encode(prompt)


def is_partial_stop(output: str, stop_str: str):
    """ Check whether the output contains a partial stop str. """
    return any(
        stop_str.startswith(output[-i:])
        for i in range(0, min(len(output), len(stop_str)))
    )


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    """
    Prepare a list of logits processors based on the provided parameters.

    Args:
        temperature (float): The temperature value for temperature warping.
        repetition_penalty (float): The repetition penalty value.
        top_p (float): The top-p value for top-p warping.
        top_k (int): The top-k value for top-k warping.

    Returns:
        LogitsProcessorList: A list of logits processors.
    """
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op, so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list
