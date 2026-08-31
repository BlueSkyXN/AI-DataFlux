"""Crash-conscious JSON helpers used by the file Job Repository."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


class CorruptJSONLError(ValueError):
    """Raised when a JSONL file is corrupt before its final line."""


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after an atomic replace."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path | str, data: Mapping[str, Any]) -> None:
    """Write JSON with temp + flush + fsync + ``os.replace`` semantics."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                data, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"JSON object expected: {path}")
    return data


def append_jsonl(path: Path | str, data: Mapping[str, Any]) -> None:
    """Append and durably flush exactly one JSONL record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with target.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def repair_jsonl_tail(path: Path | str) -> bool:
    """Remove a damaged final JSONL record and normalize the final newline.

    Returns ``True`` only when bytes were changed. Corruption before the final
    non-empty line remains a hard error.
    """

    target = Path(path)
    if not target.exists():
        return False
    raw = target.read_bytes()
    if not raw:
        return False

    lines = raw.splitlines(keepends=True)
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty[-1] if nonempty else -1
    offset = 0
    truncate_at: int | None = None
    for index, encoded_line in enumerate(lines):
        content = encoded_line.rstrip(b"\r\n")
        if not content.strip():
            offset += len(encoded_line)
            continue
        try:
            value = json.loads(content.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON object expected")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            if index != last_nonempty:
                raise CorruptJSONLError(
                    f"corrupt JSONL record at line {index + 1}: {target}"
                ) from error
            truncate_at = offset
            break
        offset += len(encoded_line)

    if truncate_at is not None:
        with target.open("r+b") as stream:
            stream.truncate(truncate_at)
            stream.flush()
            os.fsync(stream.fileno())
        return True
    if raw and not raw.endswith(b"\n"):
        with target.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return True
    return False


def read_jsonl_tolerant(path: Path | str) -> list[dict[str, Any]]:
    """Read JSONL while ignoring only a damaged or incomplete final record."""

    target = Path(path)
    if not target.exists():
        return []
    lines = target.read_bytes().splitlines()
    last_nonempty = max(
        (index for index, line in enumerate(lines) if line.strip()), default=-1
    )
    records: list[dict[str, Any]] = []
    for index, encoded_line in enumerate(lines):
        if not encoded_line.strip():
            continue
        try:
            line = encoded_line.decode("utf-8")
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index == last_nonempty:
                break
            raise CorruptJSONLError(
                f"corrupt JSONL record at line {index + 1}: {target}"
            ) from error
        if not isinstance(value, dict):
            if index == last_nonempty:
                break
            raise CorruptJSONLError(
                f"JSON object expected at line {index + 1}: {target}"
            )
        records.append(value)
    return records


def iter_file_size(paths: Iterable[Path]) -> int:
    """Return the total size of regular files without following symlinks."""

    total = 0
    for path in paths:
        if path.is_symlink():
            continue
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total
