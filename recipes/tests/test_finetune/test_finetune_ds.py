import os
import sys
import pytest
import shutil
from itertools import product
import torch
from modelscope.hub.snapshot_download import snapshot_download

sys.path.append(os.path.dirname(__file__) + "/..")
from utils import run_in_subprocess
from ut_config import (
    MODEL_TYPE,
    DOCKER_VERSION_CU114,
    DOCKER_VERSION_CU117,
    DOCKER_VERSION_CU121,
    DOCKER_MOUNT_DIR,
    DOCKER_TEST_DIR,
    DATA_DIR,
    DS_CONFIG_ZERO2_DIR,
    DS_CONFIG_ZERO3_DIR,
)

is_chat = ["chat", "base"]
docker_version = [DOCKER_VERSION_CU114, DOCKER_VERSION_CU117, DOCKER_VERSION_CU121]
# ZeRO3 is incompatible with LoRA when finetuning on base model.
# FSDP or ZeRO3 are incompatible with QLoRA.
parametrize_list_none_ds = list(
    product(*[[1], ["full", "lora"], is_chat, docker_version, [None]])
)
parametrize_list_ds_zero2 = list(
    product(*[[2], ["full", "lora"], is_chat, docker_version, [DS_CONFIG_ZERO2_DIR]])
)
parametrize_list_ds_zero3 = list(
    product(*[[2], ["full"], is_chat, docker_version, [DS_CONFIG_ZERO3_DIR]])
) + list(product(*[[2], ["lora"], ["chat"], docker_version, [DS_CONFIG_ZERO3_DIR]]))
parametrize_list_qlora = list(
    product(*[[1, 2], ["qlora"], ["chat"], docker_version, [None, DS_CONFIG_ZERO2_DIR]])
)
parametrize_list = (
    parametrize_list_none_ds
    + parametrize_list_ds_zero2
    + parametrize_list_ds_zero3
    + parametrize_list_qlora
)


@pytest.mark.parametrize(
    "num_gpus,train_type,is_chat,docker_version,deepspeed", parametrize_list
)
def test_finetune(num_gpus, train_type, is_chat, docker_version, deepspeed):
    cmd_docker = f"docker run --gpus all --ipc=host --network=host --rm -v {os.getcwd()}/../../../Qwen:{DOCKER_MOUNT_DIR} {docker_version} /bin/bash -c "
    cmd = ""
    # for GPUs SM < 80
    is_ampere = torch.cuda.get_device_capability()[0] >= 8
    if not is_ampere:
        cmd = f"pip uninstall -y flash-attn && "

    model_type = f"{MODEL_TYPE}-Chat" if is_chat == "chat" else MODEL_TYPE
    model_type = f"{model_type}-Int4" if train_type == "qlora" else model_type
    cmd += f"""torchrun --nproc_per_node {num_gpus} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 12345 {DOCKER_MOUNT_DIR}/finetune.py \
    --model_name_or_path "{DOCKER_TEST_DIR}/{model_type}/" \
    --data_path  {DATA_DIR} \
    --output_dir "{DOCKER_TEST_DIR}/output_qwen" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 512"""
    if deepspeed:
        cmd += f" --deepspeed {deepspeed}"
    if train_type == "lora":
        cmd += " --use_lora"
    elif train_type == "qlora":
        cmd += " --use_lora --q_lora"
    # for SM < 80
    if (
        (not is_ampere)
        and train_type == "lora"
        and (deepspeed and "zero2" in deepspeed)
        and is_chat == "base"
    ):
        cmd += " --fp16 True"
    snapshot_download(model_type, cache_dir=".", revision="master")
    run_in_subprocess(cmd_docker + f'"{cmd}"')
    if train_type == "full":
        assert os.path.exists("output_qwen/config.json")
    else:
        assert os.path.exists("output_qwen/adapter_config.json")
    shutil.rmtree("output_qwen")
