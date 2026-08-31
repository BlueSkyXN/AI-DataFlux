#!/usr/bin/env python3
"""Wait for a packaged Control Server and validate its public health contract."""

from __future__ import annotations

import argparse
import json
import time
from urllib.error import URLError
from urllib.request import urlopen


def wait_for_health(url: str, timeout: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status != 200:
                    raise RuntimeError(f"health returned HTTP {response.status}")
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("health response is not a JSON object")
            if payload.get("status") != "ok":
                raise RuntimeError(f"unexpected health status: {payload!r}")
            if not isinstance(payload.get("version"), str) or not payload["version"]:
                raise RuntimeError(f"health response has no version: {payload!r}")
            return payload
        except (OSError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"Control health did not become ready: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    payload = wait_for_health(args.url, args.timeout)
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
