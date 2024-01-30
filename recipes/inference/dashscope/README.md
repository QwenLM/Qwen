# Inference Qwen Using DashScope

The most simple way to use Qwen through APIs is DashScope API service through Alibaba Cloud. We give an introduction to the usage. Additionally, we provide a script for you to deploy an OpenAI-style API on your own servers.

DashScope is the large language model API service provided by Alibaba Cloud, which now supports Qwen. Note that the models behind DashScope are in-house versions temporarily without details provided. The services include `qwen-turbo` and `qwen-plus`, where the former one runs faster and the latter achieves better performance. For more information, visit the documentation [here](https://dashscope.aliyun.com).

Please head to the official website [link](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.6c2774fahtfXdn) to create a DashScope account and obtain the API key (AK). We recommend setting the AK with an environment variable:
```bash
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
```
Then please install the packages and click [here](https://help.aliyun.com/zh/dashscope/developer-reference/install-dashscope-sdk) for the documentation. If you use Python, you can install DashScope with pip:
```bash
pip install dashscope
```
If you use JAVA SDK, you can install it in this way:
```xml
<!-- https://mvnrepository.com/artifact/com.alibaba/dashscope-sdk-java -->
<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>dashscope-sdk-java</artifactId>
    <version>the-latest-version</version>
</dependency>
```
The simplest way to use DashScope is the usage with messages, which is similar to OpenAI API. The example is demonstrated below:
```python
import random
from http import HTTPStatus
from dashscope import Generation


def call_with_messages():
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '如何做西红柿鸡蛋？'}]
    gen = Generation()
    response = gen.call(
        Generation.Models.qwen_turbo,
        messages=messages,
        seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message" format.
    )
    return response


if __name__ == '__main__':
    response = call_with_messages()
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
```
For more usages, please visit the official website for more details.
<br><br>

