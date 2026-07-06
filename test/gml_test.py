from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    # api_key="sk-ws-H.EMMIRPP.F9Hw.MEQCICAkZLpOf1Y8Y-YOoiEF0t819Evp5YBQbbBUo0UeKcdFAiBgMjkVdBfKEUi2OoksBa6KDR1FJrlS8AF_qskUllECpA",
    # base_url="https://ws-v2voxi8y4z0jsid8.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key="sk-85aed58c0df049a6945bd066090efd4b",
    base_url="https://llm-hsek2qqo6rwkdvwz.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
messages = [{"role": "user", "content": "你是谁"}]
completion = client.chat.completions.create(
    model="glm-5.2",
    messages=messages,
    stream=True
)
is_answering = False  # 是否进入回复阶段
print("\n" + "=" * 20 + "思考过程" + "=" * 20)
for chunk in completion:
    if chunk.choices:
        delta = chunk.choices[0].delta
        # 只收集思考内容
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            if not is_answering:
                print(delta.reasoning_content, end="", flush=True)
        # 收到content，开始进行回复
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                is_answering = True
            print(delta.content, end="", flush=True)