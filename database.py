"""
数据库模块 —— SQLAlchemy + SQLite 持久化存储，附带内存缓存层。

架构：
  内存缓存（热数据） ──miss──▶ SQLite 数据库（持久化）
  读取：缓存命中 → 直接返回；缓存未命中 → 查 DB → 回填缓存
  写入：先写 DB → 再更新缓存

只持久化 user / assistant 角色消息；tool_call 和 tool 结果不写入 DB
（它们是每次请求中的临时交互，由大模型每次重新决策）。
"""

import time
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

from config import DATABASE_URL, HISTORY_LIMIT, CACHE_TTL

# ── SQLAlchemy 初始化 ──
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ════════════════════════════════════════════════════════
# ORM 模型
# ════════════════════════════════════════════════════════

class ChatSession(Base):
    """会话表 —— 每个 session_id 对应一个独立对话。"""
    __tablename__ = "chat_sessions"

    session_id  = Column(String(36), primary_key=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    messages = relationship(
        "ChatHistory",
        back_populates="session",
        order_by="ChatHistory.created_at",
        cascade="all, delete-orphan",
    )


class ChatHistory(Base):
    """对话记录表 —— 存储每条 user / assistant 消息。"""
    __tablename__ = "chat_history"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(String(36), ForeignKey("chat_sessions.session_id"),
                         nullable=False, index=True)
    role        = Column(String(20), nullable=False)   # "user" | "assistant"
    content     = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")


# ── 建表（首次启动自动创建） ──
Base.metadata.create_all(bind=engine)


# ════════════════════════════════════════════════════════
# 内存缓存层
# ════════════════════════════════════════════════════════

# 结构：{ session_id: {"messages": [...], "ts": float} }
_cache: dict[str, dict] = {}


def _cache_get(session_id: str) -> list[dict] | None:
    """从缓存读取消息列表，过期返回 None 并自动清理。"""
    entry = _cache.get(session_id)
    if entry is None:
        return None
    if time.time() - entry["ts"] < CACHE_TTL:
        return entry["messages"]
    # 过期，清理
    del _cache[session_id]
    return None


def _cache_set(session_id: str, messages: list[dict]):
    """将消息列表写入缓存。"""
    _cache[session_id] = {"messages": list(messages), "ts": time.time()}


# ════════════════════════════════════════════════════════
# CRUD 操作
# ════════════════════════════════════════════════════════

def _ensure_session(db, session_id: str):
    """确保会话记录存在，不存在则创建。"""
    if not db.query(ChatSession).filter(ChatSession.session_id == session_id).first():
        db.add(ChatSession(session_id=session_id))
        db.commit()


def get_recent_messages(session_id: str, limit: int = HISTORY_LIMIT) -> list[dict]:
    """
    获取指定会话的最近 N 条历史消息。

    策略：缓存命中 → 直接返回；缓存未命中 → 查 DB → 回填缓存。

    返回格式：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    # 1. 查内存缓存
    cached = _cache_get(session_id)
    if cached is not None:
        if len(cached) <= limit:
            return cached
        return cached[-limit:]

    # 2. 查数据库
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.created_at.asc())
            .all()
        )
        messages = [{"role": r.role, "content": r.content} for r in rows]
        # 回填缓存
        _cache_set(session_id, messages)
        return messages[-limit:] if len(messages) > limit else messages
    finally:
        db.close()


def save_message(session_id: str, role: str, content: str):
    """
    保存一条消息到数据库，同时更新内存缓存。

    参数:
        session_id: 会话 ID
        role:       "user" 或 "assistant"
        content:    消息正文
    """
    # 写数据库
    db = SessionLocal()
    try:
        _ensure_session(db, session_id)
        msg = ChatHistory(session_id=session_id, role=role, content=content)
        db.add(msg)
        db.commit()
    finally:
        db.close()

    # 更新缓存（追加到已有消息，若缓存已过期/缺失则回填）
    cached = _cache_get(session_id)
    if cached is not None:
        cached.append({"role": role, "content": content})
        _cache_set(session_id, cached)
    else:
        # 缓存未命中，从 DB 重新加载
        db = SessionLocal()
        try:
            rows = (
                db.query(ChatHistory)
                .filter(ChatHistory.session_id == session_id)
                .order_by(ChatHistory.created_at.asc())
                .all()
            )
            _cache_set(session_id, [{"role": r.role, "content": r.content} for r in rows])
        finally:
            db.close()


def get_active_session_count() -> int:
    """返回数据库中会话总数（用于健康检查）。"""
    db = SessionLocal()
    try:
        return db.query(ChatSession).count()
    finally:
        db.close()
