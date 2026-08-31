from __future__ import annotations

from io import BytesIO
import json
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import pytest

from src.control.client import ControlClient, ControlClientError


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body

    def __iter__(self):
        return iter(self.body.splitlines(keepends=True))


def test_control_client_json_request_builds_auth_query_and_body():
    client = ControlClient("http://control.test/", "test-token")
    response = _Response(b'{"job_id":"job-a"}')
    with patch("src.control.client.urlopen", return_value=response) as urlopen:
        payload = client.request_json(
            "POST",
            "/api/v1/jobs",
            query={"page": 2},
            body={"root_id": "project"},
            headers={"If-Match": "revision"},
        )

    assert payload == {"job_id": "job-a"}
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://control.test/api/v1/jobs?page=2"
    assert request.get_header("Authorization") == "Bearer test-token"
    assert request.get_header("If-match") == "revision"
    assert json.loads(request.data) == {"root_id": "project"}


@pytest.mark.parametrize("body", [b"not-json", b"[]"])
def test_control_client_rejects_invalid_json_shapes(body):
    client = ControlClient("http://control.test", "token")
    with patch("src.control.client.urlopen", return_value=_Response(body)):
        with pytest.raises(ControlClientError, match="invalid JSON|non-object"):
            client.request_json("GET", "/api/v1/jobs")


def test_control_client_maps_structured_http_and_connection_errors():
    client = ControlClient("http://control.test", "token")
    error_body = BytesIO(
        json.dumps(
            {
                "error": {
                    "code": "job_conflict",
                    "message": "cannot resume",
                    "details": {"status": "completed"},
                }
            }
        ).encode()
    )
    http_error = HTTPError(
        "http://control.test/api/v1/jobs/a/resume",
        409,
        "Conflict",
        {},
        error_body,
    )
    with patch("src.control.client.urlopen", side_effect=http_error):
        with pytest.raises(ControlClientError) as exc_info:
            client.request_json("POST", "/api/v1/jobs/a/resume", body={})
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "job_conflict"
    assert exc_info.value.details == {"status": "completed"}

    with patch(
        "src.control.client.urlopen", side_effect=URLError("connection refused")
    ):
        with pytest.raises(ControlClientError, match="connection failed"):
            client.request_json("GET", "/health")


def test_control_client_streams_multiline_sse_and_rejects_invalid_event():
    client = ControlClient("http://control.test", "token")
    stream = _Response(
        b"id: 1\n"
        b'data: {"seq":1,\n'
        b'data: "type":"started"}\n\n'
        b": keepalive\n\n"
    )
    with patch("src.control.client.urlopen", return_value=stream):
        events = list(client.stream_events("job-a", after_seq=2))
    assert events == [{"seq": 1, "type": "started"}]

    with patch(
        "src.control.client.urlopen", return_value=_Response(b"data: invalid\n\n")
    ):
        with pytest.raises(ControlClientError, match="invalid SSE JSON"):
            list(client.stream_events("job-a"))

    with patch("src.control.client.urlopen", return_value=_Response(b"data: []\n\n")):
        assert list(client.stream_events("job-a")) == []


def test_control_client_stream_maps_http_error():
    client = ControlClient("http://control.test", "token")
    error = HTTPError(
        "http://control.test/events",
        404,
        "Not Found",
        {},
        BytesIO(b"{}"),
    )
    with patch("src.control.client.urlopen", side_effect=error):
        with pytest.raises(ControlClientError) as exc_info:
            list(client.stream_events("missing"))
    assert exc_info.value.status_code == 404
