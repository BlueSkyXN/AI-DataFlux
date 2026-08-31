"""
Gateway 服务单元测试

覆盖 src/gateway/service.py 中不依赖真实 HTTP 上游的核心行为。
"""

import asyncio
from pathlib import Path

import yaml

from src import __version__
from src.gateway.app import create_app
from src.gateway.schemas import ChatCompletionRequest
from src.gateway.service import FluxApiService


def _write_gateway_config(tmp_path: Path, models: list[dict]) -> Path:
    config = {
        "global": {"log": {"level": "error"}},
        "datasource": {
            "type": "csv",
            "engine": "pandas",
            "concurrency": {"batch_size": 1, "max_in_flight": 1},
        },
        "csv": {"input_path": "input.csv"},
        "columns_to_extract": ["input"],
        "columns_to_write": {"result": "result"},
        "prompt": {"template": "{input}"},
        "workspace": {
            "roots": {"project": "."},
            "state_dir": ".dataflux/jobs",
        },
        "gateway": {
            "max_connections": 10,
            "max_connections_per_host": 10,
        },
        "channels": {
            "openai": {
                "name": "openai",
                "base_url": "https://api.example.test",
                "endpoints": {
                    "chat_completions": "/v1/chat/completions",
                    "responses": "/v1/responses",
                },
                "timeout": 60,
                "proxy": "",
                "ssl_verify": True,
            }
        },
        "models": models,
    }
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _model_config(
    model_id: str,
    *,
    model: str | None = None,
    weight: int = 1,
    supports_json_schema: bool = True,
    capabilities: list[str] | None = None,
) -> dict:
    effective_capabilities = capabilities or [
        "chat_completions",
        "responses",
        "stream",
        "multimodal",
        "tools",
        "n",
        "logprobs",
        "previous_response_id",
    ]
    if supports_json_schema:
        effective_capabilities = [*effective_capabilities, "json_schema"]
    config = {
        "id": model_id,
        "name": model_id,
        "model": model or model_id,
        "channel_id": "openai",
        "api_key": "test-key",
        "timeout": 60,
        "weight": weight,
        "temperature": 0.3,
        "safe_rps": 100,
        "capabilities": effective_capabilities,
    }
    return config


def _chat_request(**extra) -> ChatCompletionRequest:
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(extra)
    return ChatCompletionRequest(**payload)


def test_gateway_exposes_package_version(tmp_path):
    """Gateway OpenAPI 元数据和根端点应使用应用版本事实源"""
    config_path = _write_gateway_config(tmp_path, [_model_config("model-a")])
    app = create_app(str(config_path))
    root_route = next(route for route in app.routes if route.path == "/")

    assert app.version == __version__
    assert asyncio.run(root_route.endpoint())["version"] == __version__


def test_build_upstream_payload_forwards_extra_openai_params(tmp_path):
    """ChatCompletionRequest 允许的 extra 字段应透传到上游请求体"""
    config_path = _write_gateway_config(tmp_path, [_model_config("model-a")])
    service = FluxApiService(str(config_path))
    model = service.models[0]
    request = _chat_request(
        response_format={"type": "json_object"},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="auto",
        seed=42,
        parallel_tool_calls=False,
        metadata={"trace_id": "unit-test", "empty": None},
    )

    payload = service._build_upstream_payload(model, request)

    assert payload["model"] == "model-a"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["tool_choice"] == "auto"
    assert payload["seed"] == 42
    assert payload["parallel_tool_calls"] is False
    assert payload["metadata"] == {"trace_id": "unit-test", "empty": None}


def test_build_upstream_payload_preserves_response_format_extra_fields(tmp_path):
    """response_format=json_schema 的嵌套结构不应被 Pydantic 丢弃"""
    config_path = _write_gateway_config(tmp_path, [_model_config("model-a")])
    service = FluxApiService(str(config_path))
    model = service.models[0]
    request = _chat_request(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }
    )

    payload = service._build_upstream_payload(model, request)

    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "answer_schema"
    assert (
        payload["response_format"]["json_schema"]["schema"]["properties"]["answer"][
            "type"
        ]
        == "string"
    )


def test_build_upstream_payload_keeps_json_object_for_non_schema_model(tmp_path):
    """json_object 是 JSON 模式，不应被 supports_json_schema=false 过滤掉"""
    config_path = _write_gateway_config(
        tmp_path,
        [_model_config("model-a", supports_json_schema=False)],
    )
    service = FluxApiService(str(config_path))
    model = service.models[0]
    request = _chat_request(response_format={"type": "json_object"})

    payload = service._build_upstream_payload(model, request)

    assert payload["response_format"] == {"type": "json_object"}


def test_get_available_model_filters_json_schema_capability(tmp_path):
    """需要 JSON 输出时，不应选择未声明支持 JSON Schema 的模型"""
    config_path = _write_gateway_config(
        tmp_path,
        [
            _model_config("plain", weight=100, supports_json_schema=False),
            _model_config("json", weight=1, supports_json_schema=True),
        ],
    )
    service = FluxApiService(str(config_path))

    selected = service.get_available_model(
        requested_model_name="auto",
        requires_json_schema=True,
    )

    assert selected is not None
    assert selected.id == "json"


def test_get_available_model_falls_back_when_requested_model_lacks_json_schema(
    tmp_path,
):
    """指定模型不支持 JSON 输出时，应尝试其他符合能力的模型"""
    config_path = _write_gateway_config(
        tmp_path,
        [
            _model_config("plain", weight=100, supports_json_schema=False),
            _model_config("json", weight=1, supports_json_schema=True),
        ],
    )
    service = FluxApiService(str(config_path))

    selected = service.get_available_model(
        requested_model_name="plain",
        requires_json_schema=True,
    )

    assert selected is not None
    assert selected.id == "json"


def test_get_available_model_returns_none_when_no_model_supports_json_schema(
    tmp_path,
):
    config_path = _write_gateway_config(
        tmp_path,
        [_model_config("plain", supports_json_schema=False)],
    )
    service = FluxApiService(str(config_path))

    selected = service.get_available_model(
        requested_model_name="auto",
        requires_json_schema=True,
    )

    assert selected is None


def test_request_requires_json_schema_only_for_non_text_response_format(tmp_path):
    config_path = _write_gateway_config(tmp_path, [_model_config("model-a")])
    service = FluxApiService(str(config_path))

    assert service._request_requires_json_schema(_chat_request()) is False
    assert (
        service._request_requires_json_schema(
            _chat_request(response_format={"type": "text"})
        )
        is False
    )
    assert (
        service._request_requires_json_schema(
            _chat_request(response_format={"type": "json_object"})
        )
        is False
    )
    assert (
        service._request_requires_json_schema(
            _chat_request(response_format={"type": "json_schema"})
        )
        is True
    )
