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
from memory import retrieve_memories, extract_and_store
from config import HISTORY_LIMIT

load_dotenv()

# ════════════════════════════════════════════════════════
# 安全工具包装层 —— 输入校验 + 异常捕获 + 纠错提示
# ════════════════════════════════════════════════════════

# 已知城市列表（用于模糊匹配与纠错提示）
_KNOWN_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]


def _safe_get_weather(city: str) -> str:
    """
    天气查询（带输入校验与自动纠错提示）。
    如果城市名不在已知列表中，尝试模糊匹配并建议正确名称，
    以便 ReAct Agent 在下一次迭代中自动修正。
    """
    city = city.strip()

    # ── 精确匹配：直接调用原函数 ──
    if city in _KNOWN_CITIES:
        return get_weather(city)

    # ── 模糊匹配：按字符重叠率寻找最可能的正确城市 ──
    city_set = set(city)
    best_match = None
    best_score = 0.0
    for known in _KNOWN_CITIES:
        known_set = set(known)
        overlap = len(city_set & known_set)
        # 用 Jaccard 系数的近似：重叠 / 较短的集合大小
        score = overlap / min(len(city_set), len(known_set))
        if score > best_score:
            best_score = score
            best_match = known

    # ── 匹配度足够高 → 给出纠错建议 ──
    if best_match and best_score >= 0.4:
        return (
            f"❌ 错误：未找到城市「{city}」。\n"
            f"💡 猜测您想查询的是「{best_match}」，"
            f"请使用正确的城市名称「{best_match}」重新调用 get_weather 工具。"
        )

    # ── 完全不匹配 → 列出可用城市 ──
    cities_str = "、".join(_KNOWN_CITIES)
    return (
        f"❌ 错误：未找到城市「{city}」。\n"
        f"支持的城市：{cities_str}。\n"
        f"请使用上述城市之一重新调用 get_weather 工具。"
    )


def _safe_calculate(expression: str) -> str:
    """安全计算器（带异常捕获与重试提示）。"""
    try:
        return calculate(expression)
    except (ValueError, ZeroDivisionError, SyntaxError) as e:
        return (
            f"❌ 计算失败：{e}。\n"
            f"请检查表达式语法是否正确，修正后重新调用 calculate 工具。\n"
            f"支持的运算：+, -, *, /, //, %, **, ()。"
        )
    except Exception as e:
        return f"❌ 计算发生未知错误：{e}。请简化表达式后重试。"


def _safe_get_news() -> str:
    """新闻查询（带异常捕获）。"""
    try:
        return get_news()
    except Exception as e:
        return f"❌ 获取新闻失败：{e}。请稍后重试，或告知用户当前无法获取新闻。"


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
# LangChain 工具 —— 使用安全包装函数
# ════════════════════════════════════════════════════════

langchain_tools = [
    StructuredTool.from_function(
        func=_safe_get_weather,
        name="get_weather",
        description="查询指定城市的实时天气信息。参数 city 为城市名称，例如：北京、上海、广州、深圳。",
    ),
    StructuredTool.from_function(
        func=_safe_get_news,
        name="get_news",
        description="获取国内最新热门新闻列表。无需参数，直接调用即可获得 8 条热点新闻。",
    ),
    StructuredTool.from_function(
        func=_safe_calculate,
        name="calculate",
        description=(
            "执行安全的数学计算。参数 expression 为数学表达式，"
            "支持加减乘除、幂运算、取余和括号。例如：'1234 * 5678'、'2 ** 10'、'(100 + 200) / 3'。"
        ),
    ),
]

# ════════════════════════════════════════════════════════
# ReAct 提示词模板（含自我修正规则）
# ════════════════════════════════════════════════════════

REACT_PROMPT = PromptTemplate.from_template("""你是一个乐于助人的中文 AI 助手，具备自主推理、工具调用和自我修正能力。

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

---- 🔧 自我修正规则 ----
如果 Observation 以 "❌ 错误" 开头，说明工具调用失败。你必须：
1. 在下一个 Thought 中记录："上次调用失败，我将分析错误原因并修正参数后重试"。
2. 仔细阅读错误信息中的 💡 修正建议（例如建议的正确城市名）。
3. 发起一个新的 Action，使用修正后的参数再次调用工具。
4. 如果重试后仍然失败，在 Final Answer 中如实告知用户失败原因，并提供手动操作建议。
5. 最多重试 2 次。

---- 🧠 用户长期记忆（语义检索到的偏好/事实，可辅助个性化回答）----
{long_term_memory}
---- 以上是长期记忆 ----

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

    # ── 2. 检索长期记忆（语义搜索）──
    memories = retrieve_memories(user_query, n=3)
    long_term_memory_str = "\n".join(f"- {m}" for m in memories) if memories else "（暂无相关长期记忆）"

    # ── 3. 持久化用户消息 ──
    save_message(session_id, "user", user_query)

    # ── 4. 执行 ReAct Agent（注入长期记忆）──
    result = agent_executor.invoke({
        "input": user_query,
        "chat_history": chat_history_str,
        "long_term_memory": long_term_memory_str,
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

        # 判断这一轮是否为错误（observation 以 ❌ 开头）
        obs_text = str(observation)
        is_error = obs_text.startswith("❌")

        thoughts.append({
            "thought": thought_text,
            "action": action.tool,
            "action_input": str(action.tool_input),
            "observation": obs_text[:500],
            "is_error": is_error,
        })

    # ── 5. 持久化 AI 回复 ──
    save_message(session_id, "assistant", reply)

    # ── 6. 提取用户偏好/事实 → 存入长期记忆 ──
    stored = extract_and_store(user_query, reply, session_id)
    if stored > 0:
        print(f"[Memory] 从本轮对话提取并存储了 {stored} 条长期记忆")

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
        "我喜欢吃川菜，住在北京朝阳区",     # ← 偏好声明，触发长期记忆存储
        "查一下北京的天气",
        "那上海呢？",
        "查一下北就的天气",               # ← 输入错误，触发自我修正
        "推荐点好吃的呗",                  # ← 依赖长期记忆（知道爱吃川菜、住北京）
        "帮我算一下 1234 * 5678 等于多少",
        "帮我算一下 1 / 0 等于多少",      # ← 除以零，触发错误重试
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
                err_tag = " ⚠️ 错误重试" if t.get("is_error") else ""
                print(f"  Step {j}{err_tag}:")
                print(f"    💭 {t['thought'][:80]}")
                print(f"    🔧 {t['action']}({t['action_input']})")
                print(f"    👁️  {t['observation'][:80]}")

    from memory import memory_count
    all_history = get_recent_messages(test_sid, limit=100)
    print(f"\n持久化验证：{len(all_history)} 条消息已存入 SQLite")
    print(f"长期记忆：{memory_count()} 条偏好已存入 ChromaDB")
