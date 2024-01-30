import os
import sys
import time
import pytest
import subprocess
import torch
from modelscope.hub.snapshot_download import snapshot_download

sys.path.append(os.path.dirname(__file__) + "/..")
from utils import run_in_subprocess, simple_openai_api, TelnetPort
from ut_config import (
    MODEL_TYPE,
    DOCKER_VERSION_CU121,
    DOCKER_MOUNT_DIR,
    DOCKER_TEST_DIR,
)


@pytest.mark.parametrize(
    "num_gpus,use_int4",
    [
        (1, False),
        (1, True),
        (2, False),
        # ValueError: The input size is not aligned with the quantized weight shape. This can be caused by too large tensor parallel size.
        # (2, True)
    ],
)
def test_inference_vllm_fschat(num_gpus, use_int4):
    model_type = f"{MODEL_TYPE}-Chat-Int4" if use_int4 else f"{MODEL_TYPE}-Chat"
    container_name = "test_inference_vllm_fschat"
    cmd_docker = f'docker run --gpus all --ipc=host --network=host --rm --name="{container_name}" -p 8000:8000 -v {os.getcwd()}/../../../Qwen:{DOCKER_MOUNT_DIR} {DOCKER_VERSION_CU121} /bin/bash -c '
    cmd = ""

    cmd += f"""nohup python -m fastchat.serve.controller > /dev/null 2>&1 \
    & python -m fastchat.serve.openai_api_server --host localhost --port 8000 > /dev/null 2>&1 \
    & python -m fastchat.serve.vllm_worker --model-path {DOCKER_TEST_DIR}/{model_type} --tensor-parallel-size {num_gpus} --trust-remote-code"""

    # for GPUS SM < 80 and use_int==True
    is_ampere = torch.cuda.get_device_capability()[0] >= 8
    if not is_ampere or use_int4:
        cmd += " --dtype half"

    snapshot_download(model_type, cache_dir=".", revision="master")
    # start model server
    run_in_subprocess(
        f'docker rm -f {container_name} 2>/dev/null || echo "The container does not exist."'
    )
    print(cmd_docker + f'"{cmd}"')
    run_in_subprocess("nohup " + cmd_docker + f'"{cmd}"' + " > tmp.log 2>&1 &")

    while not TelnetPort("localhost", 21002):
        print("Wait for the model service start.")
        time.sleep(0.5)

        if (
            subprocess.run(
                f"docker inspect {container_name}",
                shell=True,
                stdout=subprocess.DEVNULL,
            ).returncode
            != 0
        ):
            break

    try:
        simple_openai_api(model_type.split("/")[-1])
    except Exception as e:
        time.sleep(1)
        with open("tmp.log") as f:
            raise Exception(f"{e} \n {f.read()}")

    run_in_subprocess(f"docker rm -f {container_name}")
