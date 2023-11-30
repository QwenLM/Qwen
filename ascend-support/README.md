# 昇腾910架构基于mindformers推理Qwen-7B-Chat模型

## 环境要求

- 硬件：Ascend 910A/B

## 运行步骤

首先参考Qwen README下载官方模型到`/path/to/Qwen-7B-Chat`。

### 下载并启动镜像

```bash
docker pull qwenllm/qwen-mindspore:latest

cd /path/to/Qwen/ascend-support

# 下载模型到此处
CHECKPOINT_PATH=/path/to/Qwen-7B-Chat

cd ascend-support

# 启动docker容器
bash docker_qwen.sh -c ${CHECKPOINT_PATH}
```

### 执行权重转换

在容器内执行下面的命令，将Qwen模型转换为适配`mindformers`的格式：

```bash
python3 /data/qwen/mindformers/research/qwen/convert_weight.py
```

转换后模型的输出位置为`${CHECKPOINT_PATH}/qwen-7b-chat.ckpt`。

### 执行推理

在容器内执行下面的命令，进行推理：

```bash
cd /data/qwen/mindformers/research/qwen
export PYTHONPATH=/data/qwen/mindformers:$PYTHONPATH
python3 infer_qwen.py
```
