import gc
import traceback
import torch
import uvicorn
import time
import uuid
import anyio
import json
from anyio.streams.memory import MemoryObjectSendStream

from abc import ABC
from threading import Lock
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from functools import partial
from typing import Dict, List, Any, Literal, Optional, Union, Tuple, Iterator, Iterable, AsyncIterator
from loguru import logger
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from sse_starlette import EventSourceResponse
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openai.types.model import Model
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
)
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
)

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

from openai_utils import (
    Role, 
    ModelList, 
    ChatCompletionCreateParams,
    CompletionCreateParams,
    ErrorCode,
    ErrorResponse,
    model_dump,
    model_parse,
    model_json,
    build_qwen_chat_input,
    is_partial_stop,
    prepare_logits_processor)


llama_outer_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def list_models():
    return ModelList(
    data=[
        Model(
            id="qwen",
            object="model",
            created=int(time.time()),
            owned_by="open"
        )
    ]
)


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request
):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await handle_request(request, template.stop)
    request.max_tokens = request.max_tokens or 1024

    params = model_dump(request)
    params.update(dict(echo=False))
    logger.debug(f"==== request ====\n{params}")

    iterator_or_completion = await run_in_threadpool(_create_chat_completion, params)

    if isinstance(iterator_or_completion, Iterator):
        # It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid, and we can use it to stream the response.
        def iterator() -> Iterator:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
        )
    else:
        return iterator_or_completion


def _create_chat_completion(
    params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[Iterator, ChatCompletion]:
    params = params or {}
    params.update(kwargs)
    return (
        _create_chat_completion_stream(params)
        if params.get("stream", False)
        else _create_chat_completion_non_stream(params)
    )


def _create_chat_completion_stream(params: Dict[str, Any]) -> Iterator:
    """
    Creates a chat completion stream.

    Args:
        params (Dict[str, Any]): The parameters for generating the chat completion.

    Yields:
        Dict[str, Any]: The output of the chat completion stream.
    """
    _id, _created, _model = None, None, None
    has_function_call = False
    for i, output in enumerate(_generate(params)):
        if output["error_code"] != 0:
            yield output
            return

        _id, _created, _model = output["id"], output["created"], output["model"]
        if i == 0:
            choice = ChunkChoice(
                index=0,
                delta=ChoiceDelta(role="assistant", content=""),
                finish_reason=None,
            )
            yield ChatCompletionChunk(
                id=f"chat{_id}",
                choices=[choice],
                created=_created,
                model=_model,
                object="chat.completion.chunk",
            )

        finish_reason = output["finish_reason"]
        if len(output["delta"]) == 0 and finish_reason != "function_call":
            continue

        function_call = None
        if finish_reason == "function_call":
            try:
                _, function_call = template.parse_assistant_response(
                    output["text"], params.get("functions"), params.get("tools"),
                )
            except Exception as e:
                traceback.print_exc()
                logger.warning("Failed to parse tool call")

        if isinstance(function_call, dict) and "arguments" in function_call:
            has_function_call = True
            function_call = ChoiceDeltaFunctionCall(**function_call)
            delta = ChoiceDelta(
                content=output["delta"],
                function_call=function_call
            )
        elif isinstance(function_call, dict) and "function" in function_call:
            has_function_call = True
            finish_reason = "tool_calls"
            function_call["index"] = 0
            tool_calls = [model_parse(ChoiceDeltaToolCall, function_call)]
            delta = ChoiceDelta(
                content=output["delta"],
                tool_calls=tool_calls,
            )
        else:
            delta = ChoiceDelta(content=output["delta"])

        choice = ChunkChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        yield ChatCompletionChunk(
            id=f"chat{_id}",
            choices=[choice],
            created=_created,
            model=_model,
            object="chat.completion.chunk",
        )

    if not has_function_call:
        choice = ChunkChoice(
            index=0,
            delta=ChoiceDelta(),
            finish_reason="stop"
        )
        yield ChatCompletionChunk(
            id=f"chat{_id}",
            choices=[choice],
            created=_created,
            model=_model,
            object="chat.completion.chunk",
        )


def _create_chat_completion_non_stream(params: Dict[str, Any]) -> Union[ChatCompletion, JSONResponse]:
    """
    Creates a chat completion based on the given parameters.

    Args:
        params (Dict[str, Any]): The parameters for generating the chat completion.

    Returns:
        ChatCompletion: The generated chat completion.
    """
    last_output = None
    for output in _generate(params):
        last_output = output

    if last_output["error_code"] != 0:
        return create_error_response(last_output["error_code"], last_output["text"])

    function_call, finish_reason = None, "stop"
    if params.get("functions") or params.get("tools"):
        try:
            res, function_call = template.parse_assistant_response(
                last_output["text"], params.get("functions"), params.get("tools"),
            )
            last_output["text"] = res
        except Exception as e:
            traceback.print_exc()
            logger.warning("Failed to parse tool call")

    if isinstance(function_call, dict) and "arguments" in function_call:
        finish_reason = "function_call"
        function_call = FunctionCall(**function_call)
        message = ChatCompletionMessage(
            role="assistant",
            content=last_output["text"],
            function_call=function_call,
        )
    elif isinstance(function_call, dict) and "function" in function_call:
        finish_reason = "tool_calls"
        tool_calls = [model_parse(ChatCompletionMessageToolCall, function_call)]
        message = ChatCompletionMessage(
            role="assistant",
            content=last_output["text"],
            tool_calls=tool_calls,
        )
    else:
        message = ChatCompletionMessage(
            role="assistant",
            content=last_output["text"].strip(),
        )

    choice = Choice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    usage = model_parse(CompletionUsage, last_output["usage"])
    return ChatCompletion(
        id=f"chat{last_output['id']}",
        choices=[choice],
        created=last_output["created"],
        model=last_output["model"],
        object="chat.completion",
        usage=usage,
    )


def _generate(params: Dict[str, Any]) -> Iterator:
    """
    Generates text based on the given parameters.

    Args:
        params (Dict[str, Any]): A dictionary containing the parameters for text generation.

    Yields:
        Iterator: A dictionary containing the generated text and error code.
    """
    messages = params.get("messages")
    inputs, prompt = _apply_chat_template(
        messages,
        max_new_tokens=params.get("max_tokens", 256),
        functions=params.get("functions"),
        tools=params.get("tools"),
    )

    params.update(dict(inputs=inputs, prompt=prompt))

    try:
        for output in _generate_stream_func(params):
            output["error_code"] = 0
            yield output

    except (ValueError, RuntimeError) as e:
        traceback.print_exc()
        yield {
            "text": f"{e}",
            "error_code": ErrorCode.INTERNAL_ERROR,
        }


def _apply_chat_template(
    messages: List[ChatCompletionMessageParam],
    max_new_tokens: Optional[int] = 256,
    functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Union[List[int], Dict[str, Any]], Optional[str]]:
    """
    Apply chat template to generate model inputs and prompt.

    Args:
        messages (List[ChatCompletionMessageParam]): List of chat completion message parameters.
        max_new_tokens (Optional[int], optional): Maximum number of new tokens to generate. Defaults to 256.
        functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Functions to apply to the messages. Defaults to None.
        tools (Optional[List[Dict[str, Any]]], optional): Tools to apply to the messages. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[Union[List[int], Dict[str, Any]], Union[str, None]]: Tuple containing the generated inputs and prompt.
    """
    if template.function_call_available:
        messages = template.postprocess_messages(
            messages, functions, tools=tools,
        )
        if functions or tools:
            logger.debug(f"==== Messages with tools ====\n{messages}")

    inputs = build_qwen_chat_input(
        tokenizer, messages, context_len, max_new_tokens, functions, tools
    )
    return inputs, None


@torch.inference_mode()
def _generate_stream_func(
    params: Dict[str, Any],
):
    # Read parameters
    input_ids = params.get("inputs")
    prompt = params.get("prompt")
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_tokens", 256))
    logprobs = params.get("logprobs")
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop")

    stop_token_ids = params.get("stop_token_ids") or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    device = model.device
    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
    else:
        start_ids = torch.as_tensor([input_ids], device=device)

    past_key_values, sent_interrupt = None, False
    token_logprobs = [None]  # The first token has no logprobs.
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    previous_text = ""
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values

            if logprobs is not None:
                # Prefull logprobs for the prompt.
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                    shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])

        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [output_ids if sent_interrupt else [token]], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=None if sent_interrupt else past_key_values,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [output_ids if sent_interrupt else [token]], device=device
                    ),
                    use_cache=True,
                    past_key_values=None if sent_interrupt else past_key_values,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]

        token = tokens[0]
        output_ids.append(token)

        if logprobs is not None:
            # Cannot use last_token_logits because logprobs is based on raw logits.
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % 2 == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len(prompt)
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=False,  # fix for qwen react
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs if echo else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}] * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            partially_stopped, finish_reason = False, None
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            if each_stop == "Observation:":
                                finish_reason = "function_call"
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if (not partially_stopped) and output and output[-1] != "�":
                delta_text = output[len(previous_text):]
                previous_text = output

                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "delta": delta_text,
                    "text": output,
                    "logprobs": ret_logprobs,
                    "finish_reason": finish_reason,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                }

        if stopped:
            break

    yield {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "delta": "",
        "text": output,
        "logprobs": ret_logprobs,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


class QwenTemplate(ABC):

    name = "qwen"
    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    allow_models = ["qwen"]
    stop = {
        "token_ids": [151643, 151644, 151645],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        "strings": ["<|endoftext|>", "<|im_end|>"],
    }
    function_call_available = True

    @property
    def template(self) -> str:
        """ This template formats inputs in the standard ChatML format. See
        https://github.com/openai/openai-python/blob/main/chatml.md
        """
        return (
            "{{ system_prompt }}"
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

    def postprocess_messages(
        self,
        messages: List[ChatCompletionMessageParam],
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return messages
    
    def parse_assistant_response(
        self,
        output: str,
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[Union[str, Dict[str, Any]]]]:
        func_name, func_args = "", ""
        i = output.rfind("\nAction:")
        j = output.rfind("\nAction Input:")
        k = output.rfind("\nObservation:")

        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is omitted by the LLM,
                # because the output text may have discarded the stop word.
                output = output.rstrip() + "\nObservation:"  # Add it back.
            k = output.rfind("\nObservation:")
            func_name = output[i + len("\nAction:"): j].strip()
            func_args = output[j + len("\nAction Input:"): k].strip()

        if func_name:
            if functions:
                function_call = {
                    "name": func_name,
                    "arguments": func_args
                }
            else:
                function_call = {
                    "function": {
                        "name": func_name,
                        "arguments": func_args
                    },
                    "id": func_name,
                    "type": "function",
                }
            return output[:k], function_call

        z = output.rfind("\nFinal Answer: ")
        if z >= 0:
            output = output[z + len("\nFinal Answer: "):]
        return output, None


async def handle_request(
        request: Union[CompletionCreateParams, ChatCompletionCreateParams],
        stop: Dict[str, Any] = None
) -> Union[Union[CompletionCreateParams, ChatCompletionCreateParams], JSONResponse]:
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        raise error_check_ret
    
    # stop settings
    _stop, _stop_token_ids = [], []
    if stop is not None:
        _stop_token_ids = stop.get("token_ids", [])
        _stop = stop.get("strings", [])

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    if request.functions:
        request.stop.append("Observation:")

    request.stop = list(set(_stop + request.stop))
    request.stop_token_ids = request.stop_token_ids or []
    request.stop_token_ids = list(set(_stop_token_ids + request.stop_token_ids))

    return request


def check_requests(request: Union[CompletionCreateParams, ChatCompletionCreateParams]) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is None or isinstance(request.stop, (str, list)):
        return None
    else:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(model_dump(ErrorResponse(message=message, code=code)), status_code=500)
    

async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Union[Iterator, AsyncIterator],
):
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                if isinstance(chunk, BaseModel):
                    chunk = model_json(chunk)
                elif isinstance(chunk, dict):
                    chunk = json.dumps(chunk, ensure_ascii=False)

                await inner_send_chan.send(dict(data=chunk))

                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()

                if llama_outer_lock.locked():
                    await inner_send_chan.send(dict(data="[DONE]"))
                    raise anyio.get_cancelled_exc_class()()
        except anyio.get_cancelled_exc_class() as e:
            logger.info("disconnected")
            with anyio.move_on_after(1, shield=True):
                logger.info(f"Disconnected from client (via refresh/close) {request.client}")
                raise e
            

def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="Qwen/Qwen-7B-Chat",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Demo server name. Default: 127.0.0.1, which is only visible from the local computer."
        " If you want other computers to access your server, use 0.0.0.0 instead.",
    )
    parser.add_argument(
        "--context_len", type=int, default=None, help="Context length for generating completions."
    )
    parser.add_argument("--disable-gc", action="store_true",
                        help="Disable GC after each response generated.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    )

    if args.cpu_only:
        device = "cpu"
    else:
        device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    ).to(device).eval()

    # Multi-GPU support, use the following two lines instead of the above line, num gpus to your actual number of graphics cards
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus(args.checkpoint_path, num_gpus=2)
    
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    )

    context_len = 8192 if args.context_len is None else args.context_len
    template = QwenTemplate()

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)