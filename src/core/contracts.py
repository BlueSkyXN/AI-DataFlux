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

PREPARED_RESULT_SCHEMA_VERSION = 1


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
    schema_version: int = PREPARED_RESULT_SCHEMA_VERSION

    @staticmethod
    def calculate_payload_hash(record_id: Any, values: dict[str, Any]) -> str:
        payload = json.dumps(
            {"record_id": record_id, "values": values},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def create(cls, record_id: Any, values: dict[str, Any]) -> "PreparedResult":
        return cls(
            record_id=record_id,
            commit_id=uuid.uuid4().hex,
            values=dict(values),
            payload_hash=cls.calculate_payload_hash(record_id, values),
            produced_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "commit_id": self.commit_id,
            "values": dict(self.values),
            "payload_hash": self.payload_hash,
            "produced_at": self.produced_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedResult":
        version = data.get("schema_version")
        if type(version) is not int or version != PREPARED_RESULT_SCHEMA_VERSION:
            raise ValueError(
                "prepared result schema_version must be "
                f"{PREPARED_RESULT_SCHEMA_VERSION}"
            )
        values = data.get("values")
        if not isinstance(values, dict):
            raise ValueError("prepared result values must be an object")
        prepared = cls(
            schema_version=version,
            record_id=data.get("record_id"),
            commit_id=str(data["commit_id"]),
            values=dict(values),
            payload_hash=str(data["payload_hash"]),
            produced_at=str(data["produced_at"]),
        )
        prepared.validate()
        return prepared

    def validate(self) -> None:
        try:
            parsed_commit_id = uuid.UUID(hex=self.commit_id)
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                "prepared result commit_id must be a UUID4 hex value"
            ) from exc
        if parsed_commit_id.version != 4 or parsed_commit_id.hex != self.commit_id:
            raise ValueError("prepared result commit_id must be a UUID4 hex value")
        expected_hash = self.calculate_payload_hash(self.record_id, self.values)
        if self.payload_hash != expected_hash:
            raise ValueError("prepared result payload hash mismatch")
        try:
            produced_at = datetime.fromisoformat(self.produced_at)
        except ValueError as exc:
            raise ValueError("prepared result produced_at must be ISO-8601") from exc
        if produced_at.tzinfo is None:
            raise ValueError("prepared result produced_at must include a timezone")


@dataclass(frozen=True)
class TaskSuccess:
    prepared_result: PreparedResult


@dataclass(frozen=True)
class WritebackOutcome:
    committed: frozenset[Any]
    failed: frozenset[Any]
    unresolved: frozenset[Any]


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
