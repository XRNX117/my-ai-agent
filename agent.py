"""
AI 智能体核心模块 —— ReAct Agent，使用 LangChain create_react_agent 实现
自主规划与工具调用循环（Thought → Action → Observation → ... → Final Answer）。

支持多轮对话记忆：通过 session_id 从数据库加载历史，每次对话自动持久化。
工具函数统一在 tools.py 中管理，本模块负责 Agent 的构建与执行。
"""

import os
import sys

# Windows 终端 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate

from tools import get_weather, get_news, calculate
from database import get_recent_messages, save_message
from config import HISTORY_LIMIT

load_dotenv()

# ════════════════════════════════════════════════════════
# LLM 初始化（DeepSeek，OpenAI 兼容）
# ════════════════════════════════════════════════════════

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.7,
    max_tokens=2048,
)

# ════════════════════════════════════════════════════════
# LangChain 工具（由 tools.py 的函数包装而来）
# ════════════════════════════════════════════════════════

langchain_tools = [
    StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description="查询指定城市的实时天气信息。参数 city 为城市名称，例如：北京、上海、广州、深圳。",
    ),
    StructuredTool.from_function(
        func=get_news,
        name="get_news",
        description="获取国内最新热门新闻列表。无需参数，直接调用即可获得 8 条热点新闻。",
    ),
    StructuredTool.from_function(
        func=calculate,
        name="calculate",
        description=(
            "执行安全的数学计算。参数 expression 为数学表达式，"
            "支持加减乘除、幂运算、取余和括号。例如：'1234 * 5678'、'2 ** 10'、'(100 + 200) / 3'。"
        ),
    ),
]

# ════════════════════════════════════════════════════════
# ReAct 提示词模板
# ════════════════════════════════════════════════════════

REACT_PROMPT = PromptTemplate.from_template("""你是一个乐于助人的中文 AI 助手，具备自主推理和工具调用能力。

你可以使用以下工具来回答用户的问题：

{tools}

严格遵循以下格式进行推理（不要跳过 Thought 步骤）：

Question: 用户提出的问题
Thought: 分析问题，思考需要采取什么行动
Action: 要使用的工具名称（必须是 [{tool_names}] 之一）
Action Input: 工具的输入参数
Observation: 工具执行后返回的结果
...（上述 Thought / Action / Action Input / Observation 可重复多次）
Thought: 我现在已经掌握了足够的信息，可以给出最终答案
Final Answer: 用中文对用户问题给出的最终回答（直接回复，不要使用工具名称或 JSON）

---- 对话历史 ----
{chat_history}
---- 以上是历史，以下是当前问题 ----

Question: {input}
Thought: {agent_scratchpad}""")

# ════════════════════════════════════════════════════════
# 创建 Agent Executor
# ════════════════════════════════════════════════════════

agent = create_react_agent(llm, langchain_tools, REACT_PROMPT)

agent_executor = AgentExecutor(
    agent=agent,
    tools=langchain_tools,
    return_intermediate_steps=True,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

# ════════════════════════════════════════════════════════
# 多轮对话入口
# ════════════════════════════════════════════════════════


def chat_with_agent(user_query: str, session_id: str) -> tuple[str, list[dict]]:
    """
    ReAct Agent 多轮对话。

    流程：
    1. 从 DB（经缓存）加载最近对话历史
    2. 持久化当前用户消息
    3. AgentExecutor 执行 ReAct 循环（Thought → Action → Observation → …）
    4. 提取 intermediate_steps 作为结构化思考过程
    5. 持久化 AI 最终回复

    返回:
        (reply, thoughts): AI 最终回复 + 思考步骤列表
    """
    # ── 1. 加载对话历史 ──
    recent = get_recent_messages(session_id, limit=HISTORY_LIMIT)
    chat_history_parts = []
    for msg in recent:
        role_label = "用户" if msg["role"] == "user" else "助手"
        chat_history_parts.append(f"{role_label}: {msg['content']}")
    chat_history_str = "\n".join(chat_history_parts) if chat_history_parts else "（新对话）"

    # ── 2. 持久化用户消息 ──
    save_message(session_id, "user", user_query)

    # ── 3. 执行 ReAct Agent ──
    result = agent_executor.invoke({
        "input": user_query,
        "chat_history": chat_history_str,
    })

    reply = result["output"]

    # ── 4. 提取结构化思考过程 ──
    thoughts = []
    for action, observation in result.get("intermediate_steps", []):
        log_text = action.log if hasattr(action, "log") else ""

        # 从 log 中提取 Thought 文本
        thought_text = ""
        if "Thought:" in log_text:
            thought_text = log_text.split("Thought:", 1)[1]
            for keyword in ["\nAction:", "\nAction Input:", "\nObservation:"]:
                if keyword in thought_text:
                    thought_text = thought_text.split(keyword)[0]
            thought_text = thought_text.strip()

        thoughts.append({
            "thought": thought_text,
            "action": action.tool,
            "action_input": str(action.tool_input),
            "observation": str(observation)[:500],
        })

    # ── 5. 持久化 AI 回复 ──
    save_message(session_id, "assistant", reply)

    return reply, thoughts


# ════════════════════════════════════════════════════════
# 本地测试
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uuid

    print("=" * 60)
    print("AI 智能体 - ReAct Agent 测试（LangChain）")
    print("=" * 60)

    test_sid = str(uuid.uuid4())
    print(f"测试会话 ID: {test_sid[:8]}...")

    conversation = [
        "查一下北京的天气",
        "那上海呢？",
        "今天有什么热门新闻？",
        "帮我算一下 1234 * 5678 等于多少",
        "帮我总结一下刚才查过的两个城市分别什么天气",
    ]

    for i, query in enumerate(conversation, 1):
        print(f"\n{'='*40}")
        print(f"[第 {i} 轮] User: {query}")
        print("=" * 40)
        reply, thoughts = chat_with_agent(query, test_sid)
        print(f"\n[Final Answer]: {reply}")
        if thoughts:
            print(f"\n思考过程（{len(thoughts)} 步）：")
            for j, t in enumerate(thoughts, 1):
                print(f"  Step {j}:")
                print(f"    💭 {t['thought'][:80]}")
                print(f"    🔧 {t['action']}({t['action_input']})")
                print(f"    👁️  {t['observation'][:80]}")

    all_history = get_recent_messages(test_sid, limit=100)
    print(f"\n持久化验证：{len(all_history)} 条消息已存入 SQLite")
