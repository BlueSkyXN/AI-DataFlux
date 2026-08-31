"""Explicit task execution results shared by the processor and recovery code."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ..models.errors import ErrorType


class FailureStage(str, Enum):
    SOURCE = "source"
    MODEL = "model"
    CONTENT = "content"
    SYSTEM = "system"


_STAGE_ERROR_TYPES = {
    FailureStage.SOURCE: ErrorType.SOURCE,
    FailureStage.MODEL: ErrorType.API,
    FailureStage.CONTENT: ErrorType.CONTENT,
    FailureStage.SYSTEM: ErrorType.SYSTEM,
}


@dataclass(frozen=True)
class PreparedResult:
    """Validated model output with a stable writeback identity."""

    record_id: Any
    commit_id: str
    values: dict[str, Any]
    payload_hash: str
    produced_at: str

    @classmethod
    def create(cls, record_id: Any, values: dict[str, Any]) -> "PreparedResult":
        payload = json.dumps(
            {"record_id": record_id, "values": values},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            record_id=record_id,
            commit_id=uuid.uuid4().hex,
            values=dict(values),
            payload_hash=hashlib.sha256(payload).hexdigest(),
            produced_at=datetime.now(timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class TaskSuccess:
    prepared_result: PreparedResult


@dataclass(frozen=True)
class TaskFailure:
    stage: FailureStage
    code: str
    message: str
    retryable: bool
    retry_after_seconds: float = 0.0

    @property
    def error_type(self) -> ErrorType:
        return _STAGE_ERROR_TYPES[self.stage]


class SourceOperationError(RuntimeError):
    """Job-level source failure after its operation budget is exhausted."""

    def __init__(self, operation: str, failure: TaskFailure):
        self.operation = operation
        self.failure = failure
        super().__init__(
            f"source {operation} failed [{failure.code}]: {failure.message}"
        )
