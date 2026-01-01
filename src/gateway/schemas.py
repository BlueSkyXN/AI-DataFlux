"""
API 请求/响应 Pydantic 模型定义
"""

from typing import Any, Literal
from pydantic import BaseModel, model_validator


class ChatMessage(BaseModel):
    """聊天消息结构"""
    role: str
    content: str
    name: str | None = None


class ResponseFormat(BaseModel):
    """响应格式定义"""
    type: str = "text"


class ChatCompletionRequest(BaseModel):
    """聊天补全请求体"""
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = 1.0
    n: int | None = 1
    max_tokens: int | None = None
    stream: bool | None = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = 0
    frequency_penalty: float | None = 0
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    response_format: ResponseFormat | None = None
    
    class Config:
        extra = "allow"
    
    @model_validator(mode="before")
    @classmethod
    def convert_stop_to_list(cls, values: dict[str, Any]) -> dict[str, Any]:
        """将字符串类型的 stop 转为列表"""
        if isinstance(values, dict) and "stop" in values and isinstance(values["stop"], str):
            values["stop"] = [values["stop"]]
        return values


class ChatCompletionResponseChoice(BaseModel):
    """聊天补全响应中的选项"""
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponseUsage(BaseModel):
    """Token 使用情况"""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatCompletionResponse(BaseModel):
    """非流式聊天补全响应体"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage | None = None


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    name: str | None = None
    model: str
    weight: int
    success_rate: float
    avg_response_time: float
    available: bool
    channel: str | None = None


class ModelsResponse(BaseModel):
    """/admin/models 响应体"""
    models: list[ModelInfo]
    total: int
    available: int


class HealthResponse(BaseModel):
    """/admin/health 响应体"""
    status: Literal["healthy", "degraded", "unhealthy"]
    available_models: int
    total_models: int
    uptime: float
