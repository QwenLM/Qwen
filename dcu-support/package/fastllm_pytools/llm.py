import ctypes;
import math
import os;
import threading
from typing import Optional, Tuple, Union, List, Callable, Dict, Any;
from copy import deepcopy

import platform
if platform.system() == 'Windows':
    fastllm_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], "fastllm_tools.dll"))
else:
    fastllm_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], "libfastllm_tools.so"))

fastllm_lib.create_llm_model.argtypes = [ctypes.c_char_p]
fastllm_lib.create_llm_model.restype = ctypes.c_int

fastllm_lib.token_decode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
fastllm_lib.token_decode.restype = ctypes.c_int

fastllm_lib.token_encode_string.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
fastllm_lib.token_encode_string.restype = ctypes.c_int

fastllm_lib.launch_response_llm_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                                  ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                                  ctypes.c_float, ctypes.c_float, ctypes.c_bool]
fastllm_lib.launch_response_llm_model.restype = ctypes.c_int

fastllm_lib.fetch_response_llm_model.argtypes = [ctypes.c_int, ctypes.c_int]
fastllm_lib.fetch_response_llm_model.restype = ctypes.c_int

fastllm_lib.fetch_response_logits_llm_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
fastllm_lib.fetch_response_logits_llm_model.restype = ctypes.c_int

fastllm_lib.response_str_llm_model.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                               ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                               ctypes.c_float, ctypes.c_float, ctypes.c_bool]
fastllm_lib.response_str_llm_model.restype = ctypes.c_char_p

fastllm_lib.launch_response_str_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p,
                                                     ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                                     ctypes.c_float, ctypes.c_float, ctypes.c_bool]
fastllm_lib.launch_response_str_llm_model.restype = ctypes.c_int

fastllm_lib.fetch_response_str_llm_model.argtypes = [ctypes.c_int, ctypes.c_int]
fastllm_lib.fetch_response_str_llm_model.restype = ctypes.c_char_p

fastllm_lib.make_history_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
fastllm_lib.make_history_llm_model.restype = ctypes.c_char_p

fastllm_lib.make_input_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
fastllm_lib.make_input_llm_model.restype = ctypes.c_char_p

fastllm_lib.add_tokenizer_word_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_float, ctypes.c_int]

fastllm_lib.set_device_map.argtype = [ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

fastllm_lib.get_llm_model_type.argtype = [ctypes.c_int]
fastllm_lib.get_llm_model_type.restype = ctypes.c_char_p

fastllm_lib.response_batch_str_llm_model.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
                                                     ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                                     ctypes.c_float, ctypes.c_float, ctypes.c_bool]
fastllm_lib.response_batch_str_llm_model.restype = ctypes.POINTER(ctypes.c_char_p)

fastllm_lib.response_batch_tokens_llm_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                                        ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                                        ctypes.c_float, ctypes.c_float, ctypes.c_bool]
fastllm_lib.response_batch_tokens_llm_model.restype = ctypes.POINTER(ctypes.c_char_p)

def set_cpu_threads(threads: int):
    fastllm_lib.set_cpu_threads(threads);

def get_cpu_threads() -> int:
    return fastllm_lib.get_cpu_threads();

def print_ins_info():
    fastllm_lib.print_cpu_ins();

def set_cpu_kvcache(cpu_kvcache):
    fastllm_lib.set_kvcache_in_cpu(ctypes.c_bool(cpu_kvcache));

def get_cpu_kvcache():
    return fastllm_lib.get_kvcache_in_cpu();

def set_cpu_low_mem(low_mem):
    fastllm_lib.set_cpu_low_mem(ctypes.c_bool(low_mem));

def get_cpu_low_mem():
    return fastllm_lib.get_cpu_low_mem();

def set_device_map(device_map):
    devices = [];
    values = [];
    if (isinstance(device_map, str)):
        devices.append(device_map);
        values.append(1);
    elif (isinstance(device_map, list)):
        devices = [str(x) for x in device_map];
        values = [1 for x in device_map];
    elif (isinstance(device_map, dict)):
        devices = [str(x) for x in device_map.keys()];
        values = [int(device_map[x]) for x in device_map.keys()];
    else:
        print("set_device_map error.");
        return;
    device_str = ''.join(devices);
    device_len = [len(x) for x in devices];
    fastllm_lib.set_device_map(len(device_len),
                               (ctypes.c_int * len(device_len))(*device_len),
                               device_str.encode(),
                               (ctypes.c_int * len(values))(*values));
def from_hf(model,
            tokenizer = None,
            dtype = "float16"):
    from fastllm_pytools import hf_model;
    return hf_model.create(model, tokenizer, dtype = dtype);

class model:
    def __init__ (self, path : str,
                  id : int = -99999):
        if (id != -99999):
            self.model = id;
        else:
            self.model = fastllm_lib.create_llm_model(path.encode());
        self.direct_query = False;

        # 为了减少重复申请释放buffer对象而使用的线程局部存储区对象池
        self.thread_local_obj = threading.local()
        self.thread_local_obj.tokenizer_encode_string__output_buffer = None
        self.thread_local_obj.tokenizer_decode_token__output_buffer = None

        # tokenizer_decode_token 输出结果的静态缓存，手工触发构建
        # 由于token数量有限且不太多，所以缓存该结果来减少调用较为适合。
        # 不做成自动缓存是为了避免在多线程调用的时候对缓存dict加锁，同时也为不同场景提供选择空间
        self.tokenizer_decode_token_cache = None

        self.model_type = fastllm_lib.get_llm_model_type(self.model).decode()
        # print("model_type:", self.model_type)

    def get_prompt(self,
                   query: str,
                   history: List[Tuple[str, str]] = None) -> str:
        if (not(history)):
            history = [];
        prompt = "";
        for i, (old_query, response) in enumerate(history):
            prompt = fastllm_lib.make_history_llm_model(self.model, prompt.encode(), i, old_query.encode(), response.encode()).decode();
        prompt = fastllm_lib.make_input_llm_model(self.model, prompt.encode(), len(history), query.encode()).decode();
        return prompt;

    def save(self, path : str):
        fastllm_lib.save_llm_model(self.model, path.encode());

    def eval(self):
        pass;

    def build_tokenizer_decode_token_cache(self):
        if self.tokenizer_decode_token_cache is not None:
            return

        cache_dict = dict()
        vocab_size = fastllm_lib.get_tokenizer_vocab_size(self.model)
        for token_id in range(vocab_size):
            cache_dict[token_id] = self.tokenizer_decode_token(token_id)

        self.tokenizer_decode_token_cache = cache_dict

    def tokenizer_encode_string(self, content: str) -> List[int]:
        output_buffer_init_len = 1024
        if self.thread_local_obj.tokenizer_encode_string__output_buffer is None:
            self.thread_local_obj.tokenizer_encode_string__output_buffer = (ctypes.c_int * output_buffer_init_len)()

        buffer = self.thread_local_obj.tokenizer_encode_string__output_buffer
        buffer_len = len(buffer)
        result_len = fastllm_lib.token_encode_string(self.model, content.encode(), buffer_len, buffer)
        if result_len > buffer_len:
            if result_len > 10240:
                # 要处理的数据过长，使用一次性的buffer
                temp_buffer = (ctypes.c_int * result_len)()
                ret = fastllm_lib.token_encode_string(self.model, content.encode(), result_len, temp_buffer)
                return [i for i in temp_buffer]
            else:
                # 扩展buffer大小
                new_buffer_len = round(math.ceil(result_len / 1024.0)) * 1024
                buffer = (ctypes.c_int * new_buffer_len)()
                self.thread_local_obj.tokenizer_encode_string__output_buffer = buffer
                result_len = fastllm_lib.token_encode_string(self.model, content.encode(), new_buffer_len, buffer)

        return [buffer[i] for i in range(result_len)]

    def tokenizer_decode_token(self, token_id: int) -> bytes:
        if self.tokenizer_decode_token_cache is not None:
            cache_result = self.tokenizer_decode_token_cache.get(token_id)
            if cache_result is not None:
                return cache_result

        output_buffer_init_len = 256
        if self.thread_local_obj.tokenizer_decode_token__output_buffer is None:
            self.thread_local_obj.tokenizer_decode_token__output_buffer = ctypes.create_string_buffer(output_buffer_init_len)

        buffer = self.thread_local_obj.tokenizer_decode_token__output_buffer
        ret = fastllm_lib.token_decode(self.model, token_id, len(buffer), buffer)
        if ret > 0:
            # buffer长度不够，扩展buffer大小
            new_buffer_len = round(math.ceil(ret / 16.0)) * 16
            buffer = ctypes.create_string_buffer(new_buffer_len)
            self.thread_local_obj.tokenizer_decode_token__output_buffer = buffer
            ret = fastllm_lib.token_decode(self.model, token_id, len(buffer), buffer)
            assert ret == 0

        buffer_bytes = buffer.raw
        result_len = len(buffer_bytes)
        for i in range(len(buffer_bytes)):
            if buffer_bytes[i] == 0:
                result_len = i
                break
        return buffer_bytes[:result_len]

    def response_logits(self,
                        query: str,
                        history: List[Tuple[str, str]] = None,
                        tokenizer = None) -> str:
        prompt = query if self.direct_query else self.get_prompt(query, history);
        if (tokenizer == None):
            handle = fastllm_lib.launch_response_str_llm_model(self.model, prompt.encode(),
                                                               ctypes.c_int(1), ctypes.c_bool(False), ctypes.c_float(1), ctypes.c_int(1),
                                                               ctypes.c_float(1), ctypes.c_float(1), ctypes.c_bool(True));
        else:
            input = tokenizer.encode(prompt);
            handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                           1, False, 1, 1, 1, 1, True);
        vocab_size = fastllm_lib.get_tokenizer_vocab_size(self.model);
        logits = list(range(vocab_size))
        array = (ctypes.c_float * (vocab_size * 4))(*logits);
        ret = fastllm_lib.fetch_response_logits_llm_model(self.model, handle, array);
        out = list(array)[:vocab_size];
        while (ret != -1):
            ret = fastllm_lib.fetch_response_logits_llm_model(self.model, handle, array);
        return out;

    def response(self,
                 query: str,
                 history: List[Tuple[str, str]] = None,
                 max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0) -> str:
        ret = "";
        for i in self.stream_response(query = query,
                                      history = history,
                                      max_length = max_length,
                                      do_sample = do_sample,
                                      top_p = top_p, top_k = top_k,
                                      temperature = temperature,
                                      repeat_penalty = repeat_penalty,
                                      one_by_one = True):
            ret += i;
        return ret;

    def stream_response(self,
                        query: str,
                        history: List[Tuple[str, str]] = None,
                        max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                        one_by_one = True):
        prompt = query if self.direct_query else self.get_prompt(query, history);
        handle = fastllm_lib.launch_response_str_llm_model(self.model, prompt.encode(),
                                                           ctypes.c_int(max_length), ctypes.c_bool(do_sample), ctypes.c_float(top_p), ctypes.c_int(top_k),
                                                           ctypes.c_float(temperature), ctypes.c_float(repeat_penalty), ctypes.c_bool(False));
        res = "";
        ret = b'';
        fail_cnt = 0;
        while True:
            ret += fastllm_lib.fetch_response_str_llm_model(self.model, handle);
            cur = "";
            try:
                cur = ret.decode();
                ret = b'';
            except:
                fail_cnt += 1;
                if (fail_cnt == 20):
                    break;
                else:
                    continue;
            fail_cnt = 0;
            if (cur == "<flmeos>"):
                break;
            if one_by_one:
                yield cur;
            else:
                res += cur;
                yield res;

    def stream_response_raw(self,
                            input_tokens: List[int],
                            max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                            one_by_one = True
                            ):
        handle = fastllm_lib.launch_response_llm_model(self.model, len(input_tokens),
                                                       (ctypes.c_int * len(input_tokens))(*input_tokens),
                                                       ctypes.c_int(max_length), ctypes.c_bool(do_sample), ctypes.c_float(top_p), ctypes.c_int(top_k),
                                                       ctypes.c_float(temperature), ctypes.c_float(repeat_penalty), ctypes.c_bool(False))

        # 可能遇到长尾char需要多个token才能够生成，所以只返回bytes，string.decode策略交给外部
        # 方便统计输出token数量，和控制不完整utf8时候解码的逻辑

        total_bytes = b''
        while True:
            cur_token = fastllm_lib.fetch_response_llm_model(self.model, handle)
            if cur_token == -1:
                break

            cur_bytes = self.tokenizer_decode_token(cur_token)

            if one_by_one:
                yield cur_bytes
            else:
                total_bytes += cur_bytes
                yield total_bytes

    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192,
             do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0, **kwargs):
        if self.model_type  != "chatglm3":
            if (not(history)):
                history = [];
            prompt = query if self.direct_query else self.get_prompt(query, history);
            input = tokenizer.encode(prompt);
            handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                           max_length, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                           False);

            result = [];
            while True:
                cur = fastllm_lib.fetch_response_llm_model(self.model, handle);
                if (cur == -1):
                    break;
                result.append(cur);
            response = tokenizer.decode(result);
            history = history + [(query, response)];
            return response, history;
        else:
            if history is None:
                history = []
            role = "user"
            input = self.build_chatglm3_input(tokenizer, query, history=history, role=role)
            history.append({"role": role, "content": query})

            handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                           max_length, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                           False);
            tokens = [];
            while True:
                cur = fastllm_lib.fetch_response_llm_model(self.model, handle);
                if (cur == -1):
                    break;
                tokens.append(cur);
            response = tokenizer.decode(tokens);
            if response and response[-1] != "�":
                response, new_history = self.process_chatglm3_response(response, history)
                return response, new_history

    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values = None,
                    max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                    return_past_key_values = False, **kwargs) -> str:
        if self.model_type  != "chatglm3":
            if (not(history)):
                history = [];
            prompt = query if self.direct_query else self.get_prompt(query, history);
            input = tokenizer.encode(prompt);
            handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                           max_length, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                           False);
            tokens = [];
            while True:
                cur = fastllm_lib.fetch_response_llm_model(self.model, handle);
                if (cur == -1):
                    break;
                tokens.append(cur);
                response = tokenizer.decode(tokens);
                new_history = history + [(query, response)];
                if return_past_key_values:
                    yield response, new_history, None;
                else:
                    yield response, new_history;
        else:
            if history is None:
                history = []
            role = "user"
            input = self.build_chatglm3_input(tokenizer, query, history=history, role=role)
            history.append({"role": role, "content": query})

            handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                           max_length, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                           False);
            tokens = [];
            while True:
                cur = fastllm_lib.fetch_response_llm_model(self.model, handle);
                if (cur == -1):
                    break;
                tokens.append(cur);
                response = tokenizer.decode(tokens);
                if response and response[-1] != "�":
                    response, new_history = self.process_chatglm3_response(response, history)
                    if return_past_key_values:
                        yield response, new_history, past_key_values
                    else:
                        yield response, new_history


    def set_adapter(self, name: str):
        fastllm_lib.set_adapter(self.model, str(name).encode())

    def disable_adapter(self):
        fastllm_lib.disable_adapter(self.model)

    def process_chatglm3_response(self, output, history):
        content = ""
        history = deepcopy(history)
        for response in output.split("<|assistant|>"):
            metadata, content = response.split("\n", maxsplit=1)
            if not metadata.strip():
                content = content.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])
                    def tool_call(**kwargs):
                        return kwargs
                    parameters = eval(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history

    def build_chatglm3_input(self, tokenizer, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(tokenizer.build_single_message(item["role"], item.get("metadata", ""), content))
        input_ids.extend(tokenizer.build_single_message(role, "", query))
        input_ids.extend([tokenizer.get_command("<|assistant|>")])
        return input_ids

    def response_batch(self, querys: List[str],
                       historys: List[List[Tuple[str, str]]] = None,
                       max_length: int = 1024, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                       **kwargs) -> List[str]:
        query_size = len(querys)
        if (not(historys)):
            historys = [[] for _ in range(query_size)]
        inputs = (ctypes.c_char_p * query_size)()
        for i, query in enumerate(querys):
            prompt = query if self.direct_query else self.get_prompt(query, historys[i])
            inputs[i] = ctypes.c_char_p(prompt.encode())

        outputs = fastllm_lib.response_batch_str_llm_model(self.model, inputs, query_size,
                                                           max_length, do_sample, top_p, top_k, temperature, repeat_penalty, False)

        responses = []
        for i in range(query_size):
            response = ctypes.string_at(outputs[i]).decode()
            responses.append(response)
            historys[i] = historys[i] + [(querys[i], response)]
        return responses, historys

    def chat_batch(self, tokenizer, querys: List[str], historys: List[List[Tuple[str, str]]] = None, max_length: int = 1024,
                   do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0, **kwargs):
        query_size = len(querys)
        if (not(historys)):
            historys = [[] for _ in range(query_size)]

            inputs = []
            inputs_len = []
            for i, query in enumerate(querys):
                prompt = query if self.direct_query else self.get_prompt(query, historys[i])
                input = tokenizer.encode(prompt);
                inputs.extend(input)
                inputs_len.append(len(input))

            outputs = fastllm_lib.response_batch_tokens_llm_model(self.model, query_size,
                                                                  (ctypes.c_int * len(inputs_len))(*inputs_len),
                                                                  (ctypes.c_int * len(inputs))(*inputs),
                                                                  max_length, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                                  False)

            responses = []
            for i in range(query_size):
                response = ctypes.string_at(outputs[i]).decode()
                responses.append(response)
                historys[i] = historys[i] + [(querys[i], response)]
            return responses, historys


