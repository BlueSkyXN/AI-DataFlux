"""Single-user access-token resolution shared by Control and Gateway."""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
from dataclasses import dataclass
from typing import Any, Callable

from ..models.errors import ConfigError

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|app[_-]?secret|password|passwd|access[_-]?token|refresh[_-]?token|token)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_VALUE = re.compile(r"(?i)(\bBearer\s+)([^\s]+)")


@dataclass(frozen=True)
class AccessToken:
    value: str
    source: str
    generated: bool = False


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def resolve_access_token(
    config: dict[str, Any],
    *,
    host: str,
    allow_generate_loopback: bool = True,
) -> AccessToken:
    env_value = os.getenv("DATAFLUX_TOKEN", "").strip()
    if env_value:
        return AccessToken(value=env_value, source="DATAFLUX_TOKEN")

    server = config.get("server", {})
    yaml_value = (
        str(server.get("token", "")).strip() if isinstance(server, dict) else ""
    )
    if yaml_value:
        return AccessToken(value=yaml_value, source="server.token")

    if allow_generate_loopback and is_loopback_host(host):
        return AccessToken(
            value=secrets.token_urlsafe(32),
            source="generated",
            generated=True,
        )
    raise ConfigError("绑定非 loopback 地址时必须设置 DATAFLUX_TOKEN 或 server.token")


def make_token_checker(access_token: AccessToken) -> Callable[[str], bool]:
    expected = access_token.value

    def check(candidate: str) -> bool:
        return bool(candidate) and secrets.compare_digest(candidate, expected)

    return check


def redact_sensitive_text(value: Any, *, known_secrets: tuple[str, ...] = ()) -> str:
    """Redact common credential shapes before writing logs or Job errors."""

    text = str(value)
    for secret_value in known_secrets:
        if secret_value:
            text = text.replace(secret_value, "[REDACTED_SECRET]")
    text = _BEARER_VALUE.sub(r"\1[REDACTED_SECRET]", text)
    text = _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED_SECRET]",
        text,
    )
    return text[:2000]
