"""
AI 智能体核心模块 —— 使用 OpenAI SDK 连接 DeepSeek API，支持 Function Calling。
支持多轮对话记忆：调用者维护 messages 列表，每次传入并接收更新后的列表。
"""

import json
import os
import sys

# 解决 Windows 终端 GBK 编码问题，强制使用 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化 OpenAI 客户端，指向 DeepSeek 的兼容端点
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 系统提示词
SYSTEM_PROMPT = "你是一个乐于助人的AI助手。当用户询问天气时，请调用 get_weather 工具获取信息后，用自然语言告知用户。"


def get_weather(city: str) -> str:
    """模拟天气查询，返回固定的天气信息。"""
    weather_data = {
        "北京": "【模拟】北京今天晴，22度，微风",
        "上海": "【模拟】上海今天多云，25度，东南风3级",
        "广州": "【模拟】广州今天阵雨，28度，南风2级",
        "深圳": "【模拟】深圳今天晴，29度，西南风3级",
        "杭州": "【模拟】杭州今天阴，20度，东北风2级",
    }
    return weather_data.get(city, f"【模拟】{city}今天多云，23度，微风")


# ----------------------------------------------------------------
# 工具定义（OpenAI Function Calling 格式）
# ----------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

# 工具名称 → 实际可调用函数的映射
TOOL_MAP = {
    "get_weather": get_weather,
}


def chat_with_agent(user_query: str, messages: list | None = None) -> tuple[str, list]:
    """
    多轮对话：让大模型根据用户输入决定是否调用工具，并返回最终回复。

    参数:
        user_query:  用户当前输入
        messages:    历史对话列表（OpenAI 格式）。传 None 则自动创建新会话。

    返回:
        (reply, updated_messages): 助手回复 + 更新后的完整对话历史。
        调用者保存 updated_messages，下次继续传入即可延续对话。
    """
    # ---------- 初始化或延续对话 ----------
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 追加当前用户消息
    messages.append({"role": "user", "content": user_query})

    # ---------- 第一次请求：让模型决定是否调用工具 ----------
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,
        temperature=0.7,
    )

    choice = response.choices[0]
    message = choice.message

    # ---------- 如果模型要调用工具，就在这里执行 ----------
    if message.tool_calls:
        # 将模型的工具调用请求加入对话历史
        messages.append(message.model_dump())

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"[Agent] 调用工具: {func_name}({func_args})")

            # 执行对应的 Python 函数
            func = TOOL_MAP.get(func_name)
            if func:
                result = func(**func_args)
            else:
                result = f"未知工具: {func_name}"

            # 将工具执行结果以 tool 角色加入对话
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # ---------- 第二次请求：让模型基于工具结果生成最终回复 ----------
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
        )
        reply = final_response.choices[0].message.content
    else:
        # ---------- 不需要工具，直接返回模型回复 ----------
        reply = message.content

    # 将助手回复加入对话历史
    messages.append({"role": "assistant", "content": reply})

    return reply, messages


# ----------------------------------------------------------------
# 本地测试 —— 多轮对话 demo
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("AI 智能体 - 多轮对话测试")
    print("=" * 50)

    # 一个连续对话：模型需要记住上下文
    conversation = [
        "查一下北京的天气",
        "那上海呢？",             # ← 依赖前文，模型应理解"那…呢"指天气
        "帮我总结一下刚才查过的两个城市分别什么天气",
    ]

    messages = None  # 首次传 None，自动创建新会话

    for query in conversation:
        print(f"\n[User]: {query}")
        reply, messages = chat_with_agent(query, messages)
        print(f"[Agent]: {reply}")
        print("-" * 50)

    print(f"\n对话结束，共 {len(messages)} 条消息（含 system prompt）")
