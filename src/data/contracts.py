"""Stable datasource contracts shared by runners and job orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AdapterCapabilities:
    """Behavior that a datasource can guarantee during recovery."""

    atomic_batch: bool
    idempotent_write: bool
    resumable: bool
    full_scan: bool


_DECLARED_CAPABILITIES: dict[str, AdapterCapabilities] = {
    "excel": AdapterCapabilities(False, True, True, True),
    "csv": AdapterCapabilities(False, True, True, True),
    "sqlite": AdapterCapabilities(True, True, True, True),
    "mysql": AdapterCapabilities(True, True, True, True),
    "postgresql": AdapterCapabilities(True, True, True, True),
    "feishu_bitable": AdapterCapabilities(False, True, True, True),
    "feishu_sheet": AdapterCapabilities(False, True, True, True),
}


def declared_adapter_capabilities(datasource_type: str) -> AdapterCapabilities:
    """Return static recovery guarantees without opening the datasource."""

    try:
        return _DECLARED_CAPABILITIES[datasource_type.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported datasource type: {datasource_type}") from exc


@dataclass(frozen=True)
class TaskRecord:
    """One datasource record with a JSON-serializable opaque identifier."""

    record_id: Any
    data: dict[str, Any]


@dataclass(frozen=True)
class TaskBatch:
    """A page of tasks and the datasource-owned cursor for the next page."""

    records: tuple[TaskRecord, ...]
    next_cursor: Any | None


@dataclass(frozen=True)
class WriteFailure:
    """A write failure attributable to one record."""

    record_id: Any
    code: str
    message: str
    retryable: bool = True


@dataclass(frozen=True)
class WritebackReceipt:
    """Durable acknowledgement returned by every datasource write."""

    batch_id: str
    persisted_ids: tuple[Any, ...] = ()
    failures: tuple[WriteFailure, ...] = ()
    atomic: bool = False
    committed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def succeeded(self) -> bool:
        return not self.failures

    @classmethod
    def persisted(
        cls,
        batch_id: str,
        record_ids: list[Any] | tuple[Any, ...],
        *,
        atomic: bool,
    ) -> "WritebackReceipt":
        return cls(
            batch_id=batch_id,
            persisted_ids=tuple(record_ids),
            atomic=atomic,
        )
