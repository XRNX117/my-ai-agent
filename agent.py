"""
AI 智能体核心模块 —— 使用 OpenAI SDK 连接 DeepSeek API，支持 Function Calling。
支持多轮对话记忆：通过 session_id 从数据库加载历史，每次对话自动持久化。

工具函数统一在 tools.py 中管理，持久化逻辑在 database.py 中。
本模块只负责与大模型的交互逻辑。
"""

import json
import os
import sys

# 解决 Windows 终端 GBK 编码问题，强制使用 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from dotenv import load_dotenv

from tools import TOOLS, TOOL_MAP
from database import get_recent_messages, save_message
from config import HISTORY_LIMIT

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化 OpenAI 客户端，指向 DeepSeek 的兼容端点
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 系统提示词
SYSTEM_PROMPT = (
    "你是一个乐于助人的 AI 助手。"
    "当用户询问天气时，请调用 get_weather 工具获取信息后，用自然语言告知用户。"
    "当用户想看新闻、了解时事热点时，请调用 get_news 工具获取新闻列表。"
    "当用户需要算数、数学计算时，请调用 calculate 工具计算结果。"
)


def chat_with_agent(user_query: str, session_id: str) -> str:
    """
    多轮对话：从数据库加载历史，与 LLM 交互，并将本轮对话持久化。

    参数:
        user_query:  用户当前输入
        session_id:  会话唯一标识（首次由调用方生成 UUID，后续沿用）

    返回:
        reply:  AI 助手的自然语言回复
    """
    # ── 1. 从 DB（经缓存）加载最近历史 ──
    recent = get_recent_messages(session_id, limit=HISTORY_LIMIT)

    # ── 2. 构建发送给 LLM 的消息列表 ──
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent)                         # 历史上下文
    messages.append({"role": "user", "content": user_query})  # 当前问题

    # ── 3. 持久化用户消息 ──
    save_message(session_id, "user", user_query)

    # ── 4. 第一次请求：让模型决定是否调用工具 ──
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,
        temperature=0.7,
    )

    choice = response.choices[0]
    message = choice.message

    # ── 5. 如果模型要调用工具，就在这里执行 ──
    if message.tool_calls:
        # 将模型的工具调用请求加入对话
        messages.append(message.model_dump())

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"[Agent] 调用工具: {func_name}({func_args})")

            func = TOOL_MAP.get(func_name)
            if func:
                result = func(**func_args)
            else:
                result = f"未知工具: {func_name}"

            # 将工具执行结果加入对话（不持久化——工具调用是临时的）
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # 第二次请求：让模型基于工具结果生成最终回复
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
        )
        reply = final_response.choices[0].message.content
    else:
        # ── 6. 不需要工具，直接使用模型回复 ──
        reply = message.content

    # ── 7. 持久化 AI 回复 ──
    save_message(session_id, "assistant", reply)

    return reply


# ----------------------------------------------------------------
# 本地测试 —— 多轮对话 demo
# ----------------------------------------------------------------
if __name__ == "__main__":
    import uuid

    print("=" * 50)
    print("AI 智能体 - 多轮对话测试（数据库持久化）")
    print("=" * 50)

    # 使用固定 session_id 模拟一个连续对话
    test_sid = str(uuid.uuid4())
    print(f"测试会话 ID: {test_sid[:8]}...")

    conversation = [
        "查一下北京的天气",
        "那上海呢？",                     # ← 依赖前文，模型应理解"那…呢"指天气
        "今天有什么热门新闻？",            # ← 触发 get_news
        "帮我算一下 1234 * 5678 等于多少", # ← 触发 calculate
        "帮我总结一下刚才查过的两个城市分别什么天气",
    ]

    for query in conversation:
        print(f"\n[User]: {query}")
        reply = chat_with_agent(query, test_sid)
        print(f"[Agent]: {reply}")
        print("-" * 50)

    # 验证持久化：再次读取历史
    all_history = get_recent_messages(test_sid, limit=100)
    user_count = sum(1 for m in all_history if m["role"] == "user")
    assistant_count = sum(1 for m in all_history if m["role"] == "assistant")
    print(f"\n持久化验证：数据库中共 {len(all_history)} 条消息 "
          f"({user_count} user, {assistant_count} assistant)")
