from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1/",
)


# List models API
models = client.models.list()
print(models.model_dump())


# Chat completion API
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "你好，请问你是谁？",
        }
    ],
    model="qwen",
)
print(chat_completion)


# Stream
stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "你好，请问你是谁？",
        }
    ],
    model="qwen",
    stream=True,
)
for part in stream:
    print(part.choices[0].delta.content or "", end="", flush=True)

