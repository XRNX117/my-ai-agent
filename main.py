"""
FastAPI Web 服务 —— 提供 /chat 接口，对接 AI 智能体。
支持多轮对话记忆：通过 session_id 区分不同会话（数据持久化到 SQLite）。
内置前端页面，浏览器直接访问 http://localhost:8000 即可使用。
"""

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import chat_with_agent
from database import get_active_session_count

app = FastAPI(title="AI 智能体", version="3.0.0")

# 允许跨域（方便前端开发调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 预加载前端页面
INDEX_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None   # 可选：传已有 session_id 可延续对话


class ChatResponse(BaseModel):
    reply: str
    session_id: str                  # 始终返回，客户端需保存以便后续请求使用


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    接收用户消息，调用 AI 智能体，返回模型生成的自然语言回复。

    多轮对话：
    - 首次请求无需传 session_id，服务端自动创建并返回。
    - 后续请求带上返回的 session_id 即可延续对话。
    - 对话历史持久化到 SQLite 数据库，服务重启不丢失。
    """
    session_id = req.session_id or str(uuid.uuid4())
    reply = chat_with_agent(req.message, session_id)
    return ChatResponse(reply=reply, session_id=session_id)


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端聊天页面"""
    return INDEX_HTML


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "active_sessions": get_active_session_count(),
        "version": "3.0.0",
    }
