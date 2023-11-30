# DCU 架构基于 fastllm 推理 Qwen 模型


## 环境配置

### 环境准备

```
docker pull image.sourcefind.cn:5000/dcu/admin/base/pytorch:1.13.1-centos7.6-dtk-23.04-py38-latest
```

### 容器启动

根据如下命令启动推理容器，其中需自定义一个容器名<container_name>，<project_path>即为本目录的路径：
```
# <container_name> 自定义容器名
# <project_path> 当前工程所在路径
docker run -it --name=<container_name> -v <project_path>:/work --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --shm-size=16G --group-add 39 image.sourcefind.cn:5000/dcu/admin/base/pytorch:1.13.1-centos7.6-dtk-23.04-py38-latest /bin/bash
```

### 加载环境

进入容器后执行如下命令，加载运行环境变量

```
source /opt/dtk-23.04/cuda/env.sh
```

### 安装方法

```
#进入本工程目录
cd package
python setup.py install
```

## 推理

### 模型转换

首先参考Qwen README下载官方模型，并通过如下方式将模型转换为 fastllm 用于推理的形式：

- 通过`pip install -r requirements.txt`安装模型转换所需依赖

- 如果使用已经下载完成的模型或者自己finetune的模型需要修改qwen2flm.py文件中创建tokenizer, model时的模型存放路径

```
# 在本工程目录下执行：
python3 qwen2flm.py qwen-7b-fp16.bin float16 # 导出fp16模型，参数为导出的模型路径
```


### 模型推理

```
# 命令行聊天程序，使用了模型创建以及流式对话效果
python cli_demo.py -p qwen-7b-fp16.bin

# batch推理程序
python cli_demo_batch.py -p qwen-7b-fp16.bin

# 简易webui，需要先安装streamlit-chat
streamlit run web_demo.py qwen-7b-fp16.bin 
```
