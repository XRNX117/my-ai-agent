"""
FastAPI Web 服务 —— 提供 /chat 接口，对接 AI 智能体。
支持多轮对话记忆：通过 session_id 区分不同会话（数据持久化到 SQLite）。
内置前端页面，浏览器直接访问 http://localhost:8000 即可使用。

接口规范统一在 schemas.py 中定义。
"""

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from schemas import ChatRequest, ChatResponse
from agent import chat_with_agent
from database import get_active_session_count

app = FastAPI(title="AI 智能体", version="3.1.0")

# 允许跨域（方便前端开发调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端页面路径（每次请求时实时读取，开发期修改 HTML 无需重启）
INDEX_PATH = Path(__file__).parent / "index.html"


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    接收用户消息，调用 AI 智能体，返回模型生成的自然语言回复。

    多轮对话：
    - session_id 由客户端生成（空字符串表示新会话）。
    - 同一 session_id 的请求共享对话上下文。
    - 对话历史持久化到 SQLite 数据库，服务重启不丢失。
    """
    session_id = req.session_id or str(uuid.uuid4())

    try:
        reply, thoughts = chat_with_agent(req.message, session_id)
        return ChatResponse(
            response=reply,
            status="success",
            thoughts=[t for t in thoughts if t.get("thought") or t.get("action")],
        )
    except Exception as e:
        return ChatResponse(
            response=f"服务异常：{e}",
            status="error",
        )


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端聊天页面（每次都重新读取文件）"""
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "active_sessions": get_active_session_count(),
        "version": "3.1.0",
    }
