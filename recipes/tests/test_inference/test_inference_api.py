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
    DOCKER_VERSION_CU114,
    DOCKER_VERSION_CU117,
    DOCKER_VERSION_CU121,
    DOCKER_MOUNT_DIR,
    DOCKER_TEST_DIR,
)


# use_cpu=True,use_int=False RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
# use_cpu=True,use_int4=True ValueError: Found modules on cpu/disk. Using Exllama or Exllamav2 backend requires all the modules to be on GPU.You can deactivate exllama backend by setting `disable_exllama=True` in the quantization config object
@pytest.mark.parametrize(
    "docker_version,use_cpu,use_int4",
    [
        (DOCKER_VERSION_CU114, False, False),
        (DOCKER_VERSION_CU114, False, True),
        (DOCKER_VERSION_CU117, False, False),
        (DOCKER_VERSION_CU117, False, True),
        (DOCKER_VERSION_CU121, False, False),
        (DOCKER_VERSION_CU121, False, True),
    ],
)
def test_inference_api(docker_version, use_cpu, use_int4):
    container_name = "test_inference_api"
    model_type = f"{MODEL_TYPE}-Chat-Int4" if use_int4 else f"{MODEL_TYPE}-Chat"
    cmd_docker = f'docker run --gpus all --ipc=host --network=host --rm --name="{container_name}" -p 8000:8000 -v {os.getcwd()}/../../../Qwen:{DOCKER_MOUNT_DIR} {docker_version} /bin/bash -c '
    cmd = ""
    # for GPUs SM < 80
    is_ampere = torch.cuda.get_device_capability()[0] >= 8
    if not is_ampere:
        cmd += f"pip uninstall -y flash-attn && "

    cmd += f"""python {DOCKER_MOUNT_DIR}/openai_api.py -c {DOCKER_TEST_DIR}/{model_type}"""

    if use_cpu:
        cmd += " --cpu-only"

    snapshot_download(model_type, cache_dir=".", revision="master")
    # start model server
    print(cmd_docker + f'"{cmd}"')
    run_in_subprocess(
        f'docker rm -f {container_name} 2>/dev/null || echo "The container does not exist."'
    )
    run_in_subprocess("nohup " + cmd_docker + f'"{cmd}"' + " > tmp.log 2>&1 &")

    while not TelnetPort("localhost", 8000):
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
        # while load int4 model such as Qwen-1_8B-Chat-Int4, the model name is Qwen-1_8B-Chat
        simple_openai_api(f"{MODEL_TYPE}-Chat".split("/")[-1])
    except Exception as e:
        time.sleep(1)
        with open("tmp.log") as f:
            raise Exception(f"{e} \n {f.read()}")

    run_in_subprocess(f"docker rm -f {container_name}")
