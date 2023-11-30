from fastllm_pytools import llm;
import torch;
import ctypes;
import numpy as np;

fastllm_data_type_dict = {
    "int4": 8,
    "int8": 3,
    "float16": 7
}
fastllm_weight_type_dict = {
    "linear": 1,
    "embedding": 2,
    "QuantizedLinear": 111
}

def create(model,
           tokenizer = None,
           pre_prompt = None,
           user_role = None,
           bot_role = None,
           history_sep = None,
           dtype = "float16"):
    if (dtype not in fastllm_data_type_dict):
        print("dtype should in ", list(fastllm_data_type_dict.keys()));
        exit(0);

    # 0.1 model info
    if model.config.model_type == "chatglm" and model.config.transformers_version == "4.30.2":
        model.config.model_type = "chatglm3"
    modelInfo = model.config.__dict__
    if model.generation_config is not None:
        modelInfo.update(model.generation_config.__dict__)
    if (pre_prompt):
        modelInfo["pre_prompt"] = pre_prompt;
    if (user_role):
        modelInfo["user_role"] = user_role;
    if (bot_role):
        modelInfo["bot_role"] = bot_role;
    if (history_sep):
        modelInfo["history_sep"] = history_sep;
    if (modelInfo["model_type"] == "baichuan" and hasattr(model, "model") and hasattr(model.model, "get_alibi_mask")):
        # Baichuan 2代
        modelInfo["use_alibi"] = "1";
        modelInfo["pre_prompt"] = "";
        modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.user_token_id) + "> ") if hasattr(model.generation_config, "user_token_id") else "";
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.assistant_token_id) + ">") if hasattr(model.generation_config, "assistant_token_id") else "";
        modelInfo["history_sep"] = "";
    if (modelInfo["model_type"] == "qwen"):
        if modelInfo["chat_format"] == "chatml":
            modelInfo["im_end_id"] = tokenizer.im_end_id
            modelInfo["im_start_id"] = tokenizer.im_start_id


    weight_type_dict = {};
    module_dict = {};
    weight_bits = {};
    for key, m in model.named_modules():
        if (str(type(m)).find("QuantizedLinear") != -1):
            weight_type_dict[key + ".weight"] = "QuantizedLinear";
            weight_bits[key + ".weight"] = m.weight_bit_width;
        if (isinstance(m, torch.nn.Linear)):
            weight_type_dict[key + ".weight"] = "linear";
            module_dict[key + ".weight"] = m;
        if (isinstance(m, torch.nn.Embedding)):
            weight_type_dict[key] = "embedding";

    peft_config = {}
    active_adapter = ""
    if hasattr(model, "peft_config"):
        peft_config = model.peft_config
    if hasattr(model, "active_adapter") and isinstance(model.active_adapter, str):
        # in transformers >= 4.33.0, active_adapter is a funtion in model, ignore it now
        active_adapter = model.active_adapter

    model = model.cpu();
    dict = model.state_dict();
    model_type = model.config.__dict__["model_type"];
    model = llm.fastllm_lib.create_empty_llm_model(model_type.encode());
    for it in modelInfo.keys():
        llm.fastllm_lib.add_dict_llm_model(model, str(it).encode(), str(modelInfo[it]).encode());

    for adapter_name in peft_config.keys():
        adapter_dict = peft_config[adapter_name].__dict__
        for it in adapter_dict.keys():
            llm.fastllm_lib.add_adapter_dict_llm_model(model, str(adapter_name).encode(), str(it).encode(), str(adapter_dict[it]).encode())
    if len(active_adapter) != 0:
        llm.fastllm_lib.set_adapter(model, str(active_adapter).encode())

    # 1. vocab
    if (tokenizer):
        if (hasattr(tokenizer, "tokenizer")):
            if modelInfo["model_type"] == "qwen":
                pass
            else:
                tokenizer = tokenizer.tokenizer;
        if (hasattr(tokenizer, "sp_model")):
            piece_size = tokenizer.sp_model.piece_size();
            for i in range(piece_size):
                llm.fastllm_lib.add_tokenizer_word_llm_model(model, tokenizer.sp_model.id_to_piece(i).encode(),
                                                             i, ctypes.c_float(tokenizer.sp_model.get_score(i)));
        else:
            vocab = tokenizer.get_vocab();
            for v in vocab.keys():
                if (modelInfo["model_type"] == "moss"):
                    vv = [(ord(c) if c not in tokenizer.byte_decoder else tokenizer.byte_decoder[c]) for c in v];
                    llm.fastllm_lib.add_tokenizer_word_llm_model(model, vv, vocab[v], ctypes.c_float(1.0));
                elif (modelInfo["model_type"] == "qwen"):
                    llm.fastllm_lib.add_tokenizer_word_llm_model(model, v, vocab[v], ctypes.c_float(1.0));
                else:
                    llm.fastllm_lib.add_tokenizer_word_llm_model(model, v.encode(), vocab[v], ctypes.c_float(1.0));
    tot = 0;
    for key in dict:
        ori_data_type = 0;
        ori_np_data_type = np.float32;
        cur_weight_type = 0;
        if (key in weight_type_dict and weight_type_dict[key] in fastllm_weight_type_dict):
            cur_weight_type = fastllm_weight_type_dict[weight_type_dict[key]];
        to_data_type = 0;

        if (cur_weight_type == 1):
            to_data_type = fastllm_data_type_dict[dtype];
            if (to_data_type == 7):
                ori_data_type = 7;
                ori_np_data_type = np.float16;
        elif (cur_weight_type == 2):
            # TODO bfloat
            to_data_type = 0;

        weight_name = key
        if peft_config is not None:
            weight_name = weight_name.replace('base_model.model.', '')
        if (cur_weight_type == 111):
            llm.fastllm_lib.add_qlinear_weight_llm_model(model, weight_name.encode(),
                                                 len(dict[key].shape),
                                                 (ctypes.c_int * len(dict[key].shape))(*list(dict[key].shape)),
                                                 weight_bits[key],
                                                 dict[key + "_scale"].numpy().astype(np.float32).ctypes.data_as(ctypes.c_void_p),
                                                 dict[key].numpy().ctypes.data_as(ctypes.c_void_p));
        else:
            llm.fastllm_lib.add_weight_llm_model(model, weight_name.encode(),
                                             len(dict[key].shape),
                                             (ctypes.c_int * len(dict[key].shape))(*list(dict[key].shape)),
                                             to_data_type, cur_weight_type, ori_data_type,
                                             dict[key].numpy().astype(ori_np_data_type).ctypes.data_as(ctypes.c_void_p));
        tot += 1;
        print("convert (", tot, "/", len(dict), end = " )\r");

    print("");
    llm.fastllm_lib.init_params_llm_model(model);
    llm.fastllm_lib.warmup_llm_model(model);
    ret = llm.model("", id = model);
    return ret;

