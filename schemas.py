"""
Pydantic 数据模型 —— 规范 API 接口的请求与响应格式。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体"""

    message: str = Field(
        ...,
        description="用户当前输入的消息",
        examples=["查一下北京的天气"],
    )
    session_id: str = Field(
        ...,
        description="会话唯一标识，客户端生成并维护。首次对话请传空字符串。",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


class ChatResponse(BaseModel):
    """聊天响应体"""

    response: str = Field(
        ...,
        description="AI 助手的自然语言回复",
    )
    status: str = Field(
        default="success",
        description="请求状态：success 表示正常，error 表示异常",
        examples=["success", "error"],
    )
