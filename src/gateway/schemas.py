"""
API 请求/响应 Pydantic 模型定义模块

本模块定义 API 网关的所有数据模型，使用 Pydantic 进行数据验证和序列化。
这些模型用于 OpenAI 兼容的 API 接口。

模型分类:

请求模型:
    - ChatMessage: 聊天消息结构
    - ResponseFormat: 响应格式定义
    - ChatCompletionRequest: 聊天补全请求体

响应模型:
    - ChatCompletionResponseChoice: 响应中的单个选项
    - ChatCompletionResponseUsage: Token 使用统计
    - ChatCompletionResponse: 完整的聊天补全响应

管理接口模型:
    - ModelInfo: 单个模型的详细信息
    - ModelsResponse: 模型列表响应
    - HealthResponse: 健康检查响应

OpenAI 兼容性:
    这些模型遵循 OpenAI Chat Completions API 的规范，
    确保与 OpenAI SDK 和其他兼容工具的互操作性。

类清单:

    请求模型:
        - ChatMessage(role, content, name?)
            单条聊天消息，角色 + 内容

        - ResponseFormat(type="text")
            响应格式定义，支持 "text" / "json_object"

        - ChatCompletionRequest(model, messages, temperature?, stream?, ...)
            聊天补全请求体（OpenAI 兼容）
            关键字段: model — 模型名称; messages — 消息列表; stream — 是否流式
            验证器: convert_stop_to_list — 将 stop 字符串统一转为列表

    响应模型:
        - ChatCompletionResponseChoice(index, message, finish_reason?)
            单个回复选项

        - ChatCompletionResponseUsage(prompt_tokens?, completion_tokens?, total_tokens?)
            Token 使用统计

        - ChatCompletionResponse(id, created, model, choices, usage?)
            完整聊天补全响应体（非流式）

    管理接口模型:
        - ModelInfo(id, name?, model, weight, success_rate, avg_response_time, available, channel?)
            单个模型的配置和运行时状态

        - ModelsResponse(models, total, available)
            /admin/models 响应体

        - HealthResponse(status, available_models, total_models, uptime)
            /admin/health 响应体，status ∈ {"healthy", "degraded", "unhealthy"}

依赖模块:
    - pydantic: 数据验证和序列化框架

使用示例:
    # 解析请求
    request = ChatCompletionRequest(**request_data)

    # 构建响应
    response = ChatCompletionResponse(
        id="chatcmpl-xxx",
        created=int(time.time()),
        model="gpt-4",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
    )
"""

from typing import Any, Literal
from pydantic import BaseModel, model_validator


class ChatMessage(BaseModel):
    """
    聊天消息结构

    表示对话中的单条消息，可以是用户、助手或系统的消息。

    Attributes:
        role (str): 消息角色，如 "user"、"assistant"、"system"
        content (str): 消息内容
        name (str | None): 可选的发送者名称（用于区分多个用户）
    """

    role: str
    content: str
    name: str | None = None


class ResponseFormat(BaseModel):
    """
    响应格式定义

    指定 AI 响应的格式，目前支持 "text" 和 "json_object"。

    Attributes:
        type (str): 响应格式类型，默认 "text"
    """

    type: str = "text"


class ChatCompletionRequest(BaseModel):
    """
    聊天补全请求体

    OpenAI 兼容的聊天补全 API 请求结构。

    Attributes:
        model (str): 请求的模型名称
        messages (list[ChatMessage]): 对话消息列表
        temperature (float | None): 采样温度，控制随机性
        top_p (float | None): 核采样参数
        n (int | None): 生成的回复数量
        max_tokens (int | None): 最大生成 Token 数
        stream (bool | None): 是否使用流式响应
        stop (str | list[str] | None): 停止序列
        presence_penalty (float | None): 存在惩罚
        frequency_penalty (float | None): 频率惩罚
        logit_bias (dict[str, float] | None): Token 偏置
        user (str | None): 用户标识
        response_format (ResponseFormat | None): 响应格式

    Config:
        extra = "allow": 允许额外字段，保持前向兼容
    """

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
        extra = "allow"  # 允许额外字段

    @model_validator(mode="before")
    @classmethod
    def convert_stop_to_list(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        将字符串类型的 stop 转为列表

        OpenAI API 允许 stop 参数是字符串或字符串列表，
        为了内部处理一致性，统一转换为列表。
        """
        if (
            isinstance(values, dict)
            and "stop" in values
            and isinstance(values["stop"], str)
        ):
            values["stop"] = [values["stop"]]
        return values


class ChatCompletionResponseChoice(BaseModel):
    """
    聊天补全响应中的选项

    表示 AI 生成的单个回复选项。

    Attributes:
        index (int): 选项索引（从 0 开始）
        message (ChatMessage): 生成的消息
        finish_reason (str | None): 结束原因，如 "stop"、"length"、"content_filter"
    """

    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponseUsage(BaseModel):
    """
    Token 使用情况

    记录请求和响应的 Token 消耗统计。

    Attributes:
        prompt_tokens (int | None): 输入提示的 Token 数
        completion_tokens (int | None): 生成内容的 Token 数
        total_tokens (int | None): 总 Token 数
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatCompletionResponse(BaseModel):
    """
    非流式聊天补全响应体

    OpenAI 兼容的完整响应结构。

    Attributes:
        id (str): 响应唯一标识符
        object (str): 对象类型，固定为 "chat.completion"
        created (int): 创建时间戳（Unix 时间）
        model (str): 使用的模型名称
        choices (list[ChatCompletionResponseChoice]): 生成的选项列表
        usage (ChatCompletionResponseUsage | None): Token 使用统计
    """

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage | None = None


class ModelInfo(BaseModel):
    """
    模型信息

    用于管理接口，提供单个模型的详细配置和运行时状态。

    Attributes:
        id (str): 模型内部 ID
        name (str | None): 模型显示名称
        model (str): 上游 API 的模型标识符
        weight (int): 加权选择的权重
        success_rate (float): 成功率（0-1）
        avg_response_time (float): 平均响应时间（秒）
        available (bool): 当前是否可用
        channel (str | None): 所属通道名称
    """

    id: str
    name: str | None = None
    model: str
    weight: int
    success_rate: float
    avg_response_time: float
    available: bool
    channel: str | None = None


class ModelsResponse(BaseModel):
    """
    /admin/models 响应体

    返回所有模型的详细信息。

    Attributes:
        models (list[ModelInfo]): 模型信息列表
        total (int): 模型总数
        available (int): 当前可用的模型数
    """

    models: list[ModelInfo]
    total: int
    available: int


class HealthResponse(BaseModel):
    """
    /admin/health 响应体

    返回服务健康状态。

    Attributes:
        status (str): 健康状态
            - "healthy": 所有模型可用
            - "degraded": 部分模型可用
            - "unhealthy": 无可用模型
        available_models (int): 可用模型数
        total_models (int): 模型总数
        uptime (float): 服务运行时间（秒）
    """

    status: Literal["healthy", "degraded", "unhealthy"]
    available_models: int
    total_models: int
    uptime: float
