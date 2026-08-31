"""Small standard-library client for the versioned Control API."""

from __future__ import annotations

import json
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ControlClientError(RuntimeError):
    """Connection or HTTP failure returned by the Control API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details = details


class ControlClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = self._url(path, query)
        encoded = None
        request_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if body is not None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        request = Request(url, data=encoded, headers=request_headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            self._raise_http_error(exc)
        except (URLError, OSError, TimeoutError) as exc:
            raise ControlClientError(f"Control API connection failed: {exc}") from exc
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ControlClientError("Control API returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ControlClientError("Control API returned a non-object JSON response")
        return parsed

    def stream_events(
        self,
        job_id: str,
        *,
        after_seq: int = 0,
    ) -> Iterator[dict[str, Any]]:
        url = self._url(
            f"/api/v1/jobs/{job_id}/events/stream",
            {"after_seq": after_seq},
        )
        request = Request(
            url,
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self.token}",
            },
            method="GET",
        )
        try:
            response = urlopen(request, timeout=None)
        except HTTPError as exc:
            self._raise_http_error(exc)
        except (URLError, OSError, TimeoutError) as exc:
            raise ControlClientError(f"Control API connection failed: {exc}") from exc
        with response:
            data_lines: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if data_lines:
                        text = "\n".join(data_lines)
                        data_lines.clear()
                        try:
                            event = json.loads(text)
                        except json.JSONDecodeError as exc:
                            raise ControlClientError(
                                "Control API returned invalid SSE JSON"
                            ) from exc
                        if isinstance(event, dict):
                            yield event
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

    def _url(self, path: str, query: dict[str, Any] | None) -> str:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urlencode(query)}"
        return url

    @staticmethod
    def _raise_http_error(exc: HTTPError) -> None:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {}
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        message = error.get("message") or f"Control API returned HTTP {exc.code}"
        raise ControlClientError(
            str(message),
            status_code=exc.code,
            code=error.get("code"),
            details=error.get("details"),
        ) from exc
