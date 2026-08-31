"""Stable datasource contracts shared by runners and job orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable


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


class CommitDisposition(str, Enum):
    """Datasource knowledge about one submitted write."""

    COMMITTED = "committed"
    REJECTED = "rejected"
    INDETERMINATE = "indeterminate"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class WritebackItem:
    """One record-level outcome in a datasource receipt."""

    record_id: Any
    disposition: CommitDisposition
    code: str = ""
    message: str = ""
    retryable: bool = False


@dataclass(frozen=True)
class WritebackReceipt:
    """Complete acknowledgement returned by every datasource write/readback."""

    batch_id: str
    submitted_ids: tuple[Any, ...]
    items: tuple[WritebackItem, ...]
    atomic: bool
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def succeeded(self) -> bool:
        return all(
            item.disposition == CommitDisposition.COMMITTED for item in self.items
        )

    @property
    def committed_ids(self) -> tuple[Any, ...]:
        return tuple(
            item.record_id
            for item in self.items
            if item.disposition == CommitDisposition.COMMITTED
        )

    @classmethod
    def committed(
        cls,
        batch_id: str,
        record_ids: Iterable[Any],
        *,
        atomic: bool,
    ) -> "WritebackReceipt":
        submitted = tuple(record_ids)
        return cls(
            batch_id=batch_id,
            submitted_ids=submitted,
            items=tuple(
                WritebackItem(record_id, CommitDisposition.COMMITTED)
                for record_id in submitted
            ),
            atomic=atomic,
        )

    @classmethod
    def rejected(
        cls,
        batch_id: str,
        record_ids: Iterable[Any],
        *,
        code: str,
        message: str,
        retryable: bool,
        atomic: bool,
    ) -> "WritebackReceipt":
        submitted = tuple(record_ids)
        return cls(
            batch_id=batch_id,
            submitted_ids=submitted,
            items=tuple(
                WritebackItem(
                    record_id,
                    CommitDisposition.REJECTED,
                    code,
                    message,
                    retryable,
                )
                for record_id in submitted
            ),
            atomic=atomic,
        )

    @classmethod
    def indeterminate(
        cls,
        batch_id: str,
        record_ids: Iterable[Any],
        *,
        code: str,
        message: str,
        atomic: bool,
    ) -> "WritebackReceipt":
        submitted = tuple(record_ids)
        return cls(
            batch_id=batch_id,
            submitted_ids=submitted,
            items=tuple(
                WritebackItem(
                    record_id,
                    CommitDisposition.INDETERMINATE,
                    code,
                    message,
                    False,
                )
                for record_id in submitted
            ),
            atomic=atomic,
        )


class WritebackContractError(RuntimeError):
    """Raised when a datasource receipt cannot be trusted."""


def _record_key(record_id: Any) -> str:
    try:
        return json.dumps(
            record_id,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise WritebackContractError(
            f"record_id is not JSON serializable: {record_id!r}"
        ) from exc


def _unique_record_map(
    record_ids: Iterable[Any],
    *,
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    duplicates: list[str] = []
    for record_id in record_ids:
        key = _record_key(record_id)
        if key in result:
            duplicates.append(key)
        else:
            result[key] = record_id
    if duplicates:
        raise WritebackContractError(
            f"{label} contains duplicate record IDs: {sorted(set(duplicates))}"
        )
    return result


def validate_writeback_receipt(
    receipt: WritebackReceipt,
    *,
    batch_id: str,
    submitted_ids: Iterable[Any],
) -> WritebackReceipt:
    """Validate exact ID coverage before any persisted count can advance."""

    if not isinstance(receipt, WritebackReceipt):
        raise WritebackContractError(
            f"datasource must return WritebackReceipt, got {type(receipt).__name__}"
        )
    if receipt.batch_id != batch_id:
        raise WritebackContractError(
            f"receipt batch_id mismatch: expected {batch_id!r}, got {receipt.batch_id!r}"
        )

    expected = _unique_record_map(submitted_ids, label="request submitted_ids")
    declared = _unique_record_map(
        receipt.submitted_ids,
        label="receipt submitted_ids",
    )
    if set(declared) != set(expected):
        missing = sorted(set(expected) - set(declared))
        unknown = sorted(set(declared) - set(expected))
        raise WritebackContractError(
            f"receipt submitted_ids mismatch: missing={missing}, unknown={unknown}"
        )

    for item in receipt.items:
        if not isinstance(item, WritebackItem):
            raise WritebackContractError(
                f"receipt item must be WritebackItem, got {type(item).__name__}"
            )

    item_ids = _unique_record_map(
        (item.record_id for item in receipt.items),
        label="receipt items",
    )
    if set(item_ids) != set(expected):
        missing = sorted(set(expected) - set(item_ids))
        unknown = sorted(set(item_ids) - set(expected))
        raise WritebackContractError(
            f"receipt item coverage mismatch: missing={missing}, unknown={unknown}"
        )

    for item in receipt.items:
        if not isinstance(item.disposition, CommitDisposition):
            raise WritebackContractError(
                f"invalid disposition for record {_record_key(item.record_id)}"
            )
        if item.disposition == CommitDisposition.COMMITTED and item.retryable:
            raise WritebackContractError(
                f"COMMITTED record {_record_key(item.record_id)} cannot be retryable"
            )

    try:
        completed_at = datetime.fromisoformat(receipt.completed_at)
    except (TypeError, ValueError) as exc:
        raise WritebackContractError("receipt completed_at must be ISO-8601") from exc
    if completed_at.tzinfo is None:
        raise WritebackContractError("receipt completed_at must include a timezone")
    return receipt
