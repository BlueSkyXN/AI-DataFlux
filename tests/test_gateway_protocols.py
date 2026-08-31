"""Gateway Chat/Responses 透传、capability 和 failover 边界测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest
import yaml
from fastapi.testclient import TestClient

from src.gateway.app import create_app
from src.gateway.schemas import ChatCompletionRequest, ResponsesRequest
from src.gateway.service import FluxApiService, GatewayAPIError
import src.gateway.app as gateway_app_module


def _model(
    model_id: str,
    *,
    capabilities: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "model": f"upstream-{model_id}",
        "channel_id": "openai",
        "api_key": "test-key",
        "timeout": 60,
        "weight": 1,
        "temperature": 0.3,
        "safe_rps": 100,
        "capabilities": capabilities
        or [
            "chat_completions",
            "responses",
            "stream",
            "multimodal",
            "tools",
            "n",
            "logprobs",
            "json_schema",
            "previous_response_id",
        ],
    }
    return result


def _config(
    tmp_path: Path,
    models: list[dict[str, Any]],
) -> Path:
    channel: dict[str, Any] = {
        "name": "openai",
        "base_url": "https://api.example.test",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "responses": "/v1/responses",
        },
        "timeout": 60,
        "ssl_verify": True,
    }
    path = tmp_path / "gateway-protocols.yaml"
    path.write_text(
        yaml.safe_dump(
            {
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
                "channels": {"openai": channel},
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    return path


class _FakeContent:
    def __init__(self, chunks: list[bytes | Exception]) -> None:
        self.chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        payload: Any = None,
        chunks: list[bytes | Exception] | None = None,
    ) -> None:
        self.status = status
        self.payload = payload
        self.content = _FakeContent(chunks or [])
        self.closed = False
        self.connection = None

    async def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    async def text(self) -> str:
        return str(self.payload)

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, outcomes: list[_FakeResponse | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakePool:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    async def get_or_create(self, **_kwargs: Any) -> _FakeSession:
        return self.session


def _service_with_session(
    config_path: Path,
    outcomes: list[_FakeResponse | Exception],
) -> tuple[FluxApiService, _FakeSession]:
    service = FluxApiService(str(config_path))
    session = _FakeSession(outcomes)
    service.session_pool = _FakePool(session)  # type: ignore[assignment]
    return service, session


def test_chat_request_preserves_text_multimodal_tools_and_unknown_fields(tmp_path):
    config_path = _config(tmp_path, [_model("model-a")])
    service = FluxApiService(str(config_path))
    request = ChatCompletionRequest(
        model="model-a",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
                "future_message_field": {"kept": None},
            }
        ],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice="auto",
        n=2,
        logprobs=True,
        top_logprobs=3,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        },
        future_top_level={"nested": None},
    )

    payload = service._build_upstream_payload(service.models[0], request)

    assert payload["messages"] == request.model_dump(exclude_unset=True)["messages"]
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["n"] == 2
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 3
    assert payload["response_format"]["json_schema"]["name"] == "answer"
    assert payload["future_top_level"] == {"nested": None}


@pytest.mark.asyncio
async def test_non_streaming_chat_response_is_not_rebuilt(tmp_path):
    upstream = {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "model": "physical-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call-1", "type": "function"}],
                },
                "finish_reason": "tool_calls",
                "logprobs": {"content": []},
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "alternate"},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "details": {"x": 1}},
        "future_response_field": {"kept": True},
    }
    config_path = _config(tmp_path, [_model("model-a")])
    service, _session = _service_with_session(
        config_path,
        [_FakeResponse(payload=upstream)],
    )

    result = await service.chat_completion(
        ChatCompletionRequest(
            model="model-a",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert result == upstream


@pytest.mark.asyncio
async def test_responses_payload_and_response_are_forwarded_without_chat_translation(
    tmp_path,
):
    upstream = {
        "id": "resp_123",
        "object": "response",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
        ],
        "future_response_field": [1, 2],
    }
    config_path = _config(
        tmp_path,
        [_model("model-a")],
    )
    service, session = _service_with_session(
        config_path,
        [_FakeResponse(payload=upstream)],
    )
    request = ResponsesRequest(
        model="model-a",
        input=[{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        instructions="be concise",
        metadata={"trace": None},
    )

    result = await service.responses(request)

    assert result == upstream
    assert session.calls[0]["url"].endswith("/v1/responses")
    assert session.calls[0]["json"]["instructions"] == "be concise"
    assert session.calls[0]["json"]["metadata"] == {"trace": None}
    assert "messages" not in session.calls[0]["json"]


@pytest.mark.asyncio
async def test_previous_response_id_routes_only_to_native_capable_model(tmp_path):
    config_path = _config(
        tmp_path,
        [
            _model("plain", capabilities=["responses"]),
            _model(
                "stateful",
                capabilities=["responses", "previous_response_id"],
            ),
        ],
    )
    service, session = _service_with_session(
        config_path,
        [_FakeResponse(payload={"id": "resp_next", "object": "response"})],
    )

    result = await service.responses(
        ResponsesRequest(
            model="plain",
            input="continue",
            previous_response_id="resp_previous",
        )
    )

    assert result["id"] == "resp_next"
    assert session.calls[0]["json"]["model"] == "upstream-stateful"
    assert session.calls[0]["json"]["previous_response_id"] == "resp_previous"


@pytest.mark.asyncio
async def test_chat_and_responses_sse_are_forwarded_byte_for_byte(tmp_path):
    chunks = [
        b": upstream keepalive\n\n",
        b"event: response.output_text.delta\n",
        b'data: {"delta":"hello"}\n\n',
        b"data: [DONE]\n\n",
    ]
    config_path = _config(
        tmp_path,
        [_model("model-a")],
    )
    service, session = _service_with_session(
        config_path,
        [
            _FakeResponse(chunks=chunks),
            _FakeResponse(chunks=chunks),
        ],
    )

    chat_stream = await service.chat_completion(
        ChatCompletionRequest(
            model="model-a",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )
    responses_stream = await service.responses(
        ResponsesRequest(model="model-a", input="hello", stream=True)
    )

    assert b"".join([chunk async for chunk in chat_stream]) == b"".join(chunks)
    assert b"".join([chunk async for chunk in responses_stream]) == b"".join(chunks)
    assert len(session.calls) == 2


def test_capability_routing_uses_each_models_canonical_list(tmp_path):
    config_path = _config(
        tmp_path,
        [
            _model("plain", capabilities=["chat_completions"]),
            _model(
                "tool-model",
                capabilities=["chat_completions", "tools"],
            ),
        ],
    )
    service = FluxApiService(str(config_path))

    selected = service.get_available_model(
        requested_model_name="plain",
        required_capabilities={"chat_completions", "tools"},
    )

    assert selected is not None
    assert selected.id == "tool-model"
    capabilities = service.get_capabilities()
    plain = next(item for item in capabilities["data"] if item["id"] == "plain")
    assert "chat_completions" in plain["capabilities"]
    assert "tools" not in plain["capabilities"]


def test_text_and_advanced_chat_features_produce_expected_capability_set(tmp_path):
    config_path = _config(tmp_path, [_model("model-a")])
    service = FluxApiService(str(config_path))
    text_request = ChatCompletionRequest(
        model="model-a",
        messages=[{"role": "user", "content": "hello"}],
    )
    advanced_request = ChatCompletionRequest(
        model="model-a",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                ],
            }
        ],
        stream=True,
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        n=2,
        logprobs=True,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        },
    )

    text_payload = service._build_upstream_payload_template(
        text_request,
        endpoint="chat_completions",
    )
    advanced_payload = service._build_upstream_payload_template(
        advanced_request,
        endpoint="chat_completions",
    )

    assert service._required_capabilities(
        text_payload,
        endpoint="chat_completions",
    ) == {"chat_completions"}
    assert service._required_capabilities(
        advanced_payload,
        endpoint="chat_completions",
    ) == {
        "chat_completions",
        "json_schema",
        "logprobs",
        "multimodal",
        "n",
        "stream",
        "tools",
    }


def test_admin_capabilities_and_incoming_token_checker(tmp_path):
    config_path = _config(tmp_path, [_model("model-a")])
    app = create_app(
        str(config_path),
        incoming_token_checker=lambda token: token == "incoming-test-token",
    )

    with TestClient(app) as client:
        unauthorized = client.get("/admin/capabilities")
        authorized = client.get(
            "/admin/capabilities",
            headers={"Authorization": "Bearer incoming-test-token"},
        )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "authentication_error"
    assert authorized.status_code == 200
    assert authorized.json()["object"] == "list"
    assert authorized.json()["data"][0]["id"] == "model-a"
    assert authorized.json()["data"][0]["capabilities"]
    assert "supports_json_schema" not in authorized.json()["data"][0]
    assert "supports_advanced_params" not in authorized.json()["data"][0]


def test_gateway_http_routes_return_passthrough_models_health_and_errors(tmp_path):
    config_path = _config(tmp_path, [_model("model-a")])

    async def token_checker(token):
        return token == "incoming-test-token"

    app = create_app(str(config_path), incoming_token_checker=token_checker)
    service = gateway_app_module.get_service()
    service.chat_completion = AsyncMock(
        return_value={"id": "chatcmpl", "choices": [{"index": 0}]}
    )
    service.responses = AsyncMock(
        return_value={"id": "resp_1", "object": "response", "output": []}
    )
    headers = {"Authorization": "Bearer incoming-test-token"}

    with TestClient(app) as client:
        root = client.get("/")
        models = client.get("/v1/models", headers=headers)
        admin_models = client.get("/admin/models", headers=headers)
        health = client.get("/admin/health", headers=headers)
        chat = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "model-a", "messages": []},
        )
        responses = client.post(
            "/v1/responses",
            headers=headers,
            json={"model": "model-a", "input": "hello"},
        )

    assert root.status_code == 200
    assert models.json()["data"][0]["id"] == "model-a"
    assert admin_models.json()["total"] == 1
    assert health.json()["total_models"] == 1
    assert chat.json()["id"] == "chatcmpl"
    assert responses.json()["id"] == "resp_1"

    service.chat_completion = AsyncMock(
        side_effect=GatewayAPIError("bad request", status_code=400, code="bad_request")
    )
    service.responses = AsyncMock(side_effect=RuntimeError("internal"))
    with TestClient(app) as client:
        chat_error = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "model-a", "messages": []},
        )
        response_error = client.post(
            "/v1/responses",
            headers=headers,
            json={"model": "model-a", "input": "hello"},
        )
    assert chat_error.status_code == 400
    assert chat_error.json()["error"]["code"] == "bad_request"
    assert response_error.status_code == 500
    assert response_error.json()["error"]["code"] == "internal_error"


def test_gateway_http_routes_forward_raw_sse_and_service_unavailable(tmp_path):
    config_path = _config(tmp_path, [_model("model-a")])
    app = create_app(str(config_path), incoming_token_checker=lambda _token: True)
    service = gateway_app_module.get_service()

    async def stream():
        yield b"event: response.output_text.delta\n"
        yield b'data: {"delta":"ok"}\n\n'

    service.chat_completion = AsyncMock(return_value=stream())
    with TestClient(app) as client:
        streamed = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": True},
        )
        original_service = gateway_app_module._service
        gateway_app_module._service = None
        unavailable = client.get("/v1/models")
        gateway_app_module._service = original_service

    assert streamed.content == (
        b"event: response.output_text.delta\n" b'data: {"delta":"ok"}\n\n'
    )
    assert unavailable.status_code == 503


def test_no_capable_model_returns_standard_openai_error(tmp_path):
    config_path = _config(
        tmp_path,
        [_model("plain", capabilities=["chat_completions"])],
    )
    app = create_app(str(config_path), incoming_token_checker=lambda _token: True)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "plain",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "No available model supports the required capabilities: chat_completions, tools",
            "type": "invalid_request_error",
            "param": "model",
            "code": "unsupported_capability",
        }
    }


def test_unknown_model_returns_standard_not_found_error(tmp_path):
    config_path = _config(tmp_path, [_model("known")])
    app = create_app(str(config_path), incoming_token_checker=lambda _token: True)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "typo-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "The model 'typo-model' does not exist",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


def test_gateway_rejects_invalid_model_instead_of_silently_dropping_it(tmp_path):
    invalid = _model("invalid")
    invalid["safe_rps"] = "fast"
    config_path = _config(tmp_path, [_model("valid"), invalid])

    with pytest.raises(ValueError, match=r"models\[1\]\.safe_rps"):
        FluxApiService(str(config_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_failover_before_response_start_for_retryable_statuses(tmp_path, status):
    success = {"id": "ok", "choices": [{"index": 0}]}
    config_path = _config(tmp_path, [_model("first"), _model("second")])
    service, session = _service_with_session(
        config_path,
        [
            _FakeResponse(
                status=status,
                payload={"error": {"message": "retry", "type": "server_error"}},
            ),
            _FakeResponse(payload=success),
        ],
    )

    result = await service.chat_completion(
        ChatCompletionRequest(
            model="first",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert result == success
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_failover_before_response_start_for_connection_error(tmp_path):
    config_path = _config(tmp_path, [_model("first"), _model("second")])
    service, session = _service_with_session(
        config_path,
        [
            aiohttp.ClientConnectionError("connect failed"),
            _FakeResponse(payload={"id": "ok", "choices": []}),
        ],
    )

    result = await service.chat_completion(
        ChatCompletionRequest(
            model="first",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert result["id"] == "ok"
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_non_retryable_4xx_does_not_fail_over(tmp_path):
    config_path = _config(tmp_path, [_model("first"), _model("second")])
    service, session = _service_with_session(
        config_path,
        [
            _FakeResponse(
                status=400,
                payload={
                    "error": {
                        "message": "invalid request",
                        "type": "invalid_request_error",
                        "code": "bad_request",
                    }
                },
            ),
            _FakeResponse(payload={"id": "must-not-be-used"}),
        ],
    )

    with pytest.raises(GatewayAPIError) as exc_info:
        await service.chat_completion(
            ChatCompletionRequest(
                model="first",
                messages=[{"role": "user", "content": "hello"}],
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.error["code"] == "bad_request"
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_stream_failure_after_response_start_does_not_fail_over(tmp_path):
    config_path = _config(tmp_path, [_model("first"), _model("second")])
    service, session = _service_with_session(
        config_path,
        [
            _FakeResponse(
                chunks=[
                    b'data: {"delta":"started"}\n\n',
                    aiohttp.ClientPayloadError("truncated"),
                ]
            ),
            _FakeResponse(chunks=[b"data: must-not-be-used\n\n"]),
        ],
    )
    stream = await service.chat_completion(
        ChatCompletionRequest(
            model="first",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    with pytest.raises(aiohttp.ClientPayloadError):
        _ = [chunk async for chunk in stream]

    assert len(session.calls) == 1
