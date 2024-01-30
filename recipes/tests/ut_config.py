import os

# common
MODEL_TYPE = "Qwen/Qwen-1_8B"
DOCKER_VERSION_CU114 = "qwenllm/qwen:cu114"
DOCKER_VERSION_CU117 = "qwenllm/qwen:cu117"
DOCKER_VERSION_CU121 = "qwenllm/qwen:cu121"
DOCKER_MOUNT_DIR = "/qwen-recipes"
DOCKER_TEST_DIR = os.path.join(DOCKER_MOUNT_DIR, "recipes/tests")

# finetune
DATA_DIR = os.path.join(DOCKER_MOUNT_DIR, "recipes/tests/assets/test_sampled_qwen.json")
DS_CONFIG_ZERO2_DIR = os.path.join(
    DOCKER_MOUNT_DIR, "finetune/ds_config_zero2.json"
)
DS_CONFIG_ZERO3_DIR = os.path.join(
    DOCKER_MOUNT_DIR, "finetune/ds_config_zero3.json"
)
