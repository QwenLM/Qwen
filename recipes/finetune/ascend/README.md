# Fine-tuning Qwen by Ascend NPU
Below, we provide a simple example to show how to finetune Qwen by Ascend NPU. Currently, fine-tuning and inference are supported for Qwen 7B and 14B models. You can also refer to the official [mindformers](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen/qwen.md) for detailed usage.

## Environment Requirement

- Hardware: Ascend 910A/B

## Quickstart

1. Launch Docker Image

```bash
ImageID=pai-image-manage-registry.cn-wulanchabu.cr.aliyuncs.com/pai/llm-inference:qwen_v23.0.rc3
docker run -it -u root --ipc=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /var/log/npu/:/usr/slog \
-v /etc/hccn.conf:/etc/hccn.conf \
${ImageID} /bin/bash
```

2. Download and Convert model

- download model by modelscope

```bash
cd mindformers
python3 -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('Qwen/Qwen-7B-Chat', cache_dir='.', revision='master')"
```

- convert hf model weights to ckpt weights

```bash
python research/qwen/convert_weight.py \
    --torch_ckpt_dir Qwen/Qwen-7B-Chat \
    --mindspore_ckpt_path qwen-7b-chat.ckpt

mkdir -vp load_checkpoint/rank_0
mv qwen-7b-chat.ckpt load_checkpoint/rank_0/
```

3. Prepare training data

- download demo data

```bash
wget -c https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/alpaca_data_min.json
```

- Converts the raw data to the specified format

```bash
python research/qwen/alpaca_converter.py \
    --data_path alpaca_data_min.json \
    --output_path alpaca-data-conversation_min.json
```

- Generate Mindrecord data

```bash
python research/qwen/qwen_preprocess.py \
    --input_glob alpaca-data-conversation_min.json \
    --model_file Qwen/Qwen-7B-Chat/qwen.tiktoken \
    --seq_length 1024 \
    --output_file alpaca_min.mindrecord
```

4. Prepare RANK_TABLE_FILE

```bash
# generate RANK_TABLE_FILE with 8 npu
python mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

5. Fine-tune

You need to replace RANK_TABLE_FILE with the file generated in step 5.

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE
bash research/run_singlenode.sh "python3 research/qwen/run_qwen.py \
--config research/qwen/run_qwen_7b.yaml \
--load_checkpoint /mindformers/research/qwen/load_checkpoint \
--vocab_file Qwen/Qwen-7B-Chat/qwen.tiktoken \
--use_parallel True \
--run_mode finetune \
--auto_trans_ckpt True \
--train_data alpaca_min.mindrecord" \
RANK_TABLE_FILE [0,8] 8
```

6. Merge model weights

- Rename model weights

```bash
cd output/checkpoint_network
mv rank_0/qwen_rank_0-network.ckpt rank_0/checkpoint_0.ckpt
mv rank_1/qwen_rank_1-network.ckpt rank_1/checkpoint_1.ckpt
mv rank_2/qwen_rank_2-network.ckpt rank_2/checkpoint_2.ckpt
mv rank_3/qwen_rank_3-network.ckpt rank_3/checkpoint_3.ckpt
mv rank_4/qwen_rank_4-network.ckpt rank_4/checkpoint_4.ckpt
mv rank_5/qwen_rank_5-network.ckpt rank_5/checkpoint_5.ckpt
mv rank_6/qwen_rank_6-network.ckpt rank_6/checkpoint_6.ckpt
mv rank_7/qwen_rank_7-network.ckpt rank_7/checkpoint_7.ckpt
cd ../..
```

- Merge model weights

```bash
python mindformers/tools/transform_ckpt.py \
    --src_ckpt_strategy output/strategy  \
    --src_ckpt_dir output/checkpoint_network \
    --dst_ckpt_dir output/merged_model
```

7. Inference fine-tuned model

```bash
python research/qwen/run_qwen.py \
    --config research/qwen/run_qwen_7b.yaml \
    --predict_data '比较适合深度学习入门的书籍有' \
    --run_mode predict \
    --load_checkpoint output/merged_model/rank_0/checkpoint_0.ckpt \
    --vocab_file Qwen/Qwen-7B-Chat/qwen.tiktoken \
    --auto_trans_ckpt False \
    --device_id 0
```