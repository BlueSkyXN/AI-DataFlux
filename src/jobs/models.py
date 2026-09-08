"""Public models for durable batch jobs.

The job package deliberately keeps these models independent from the processing,
control, and CLI layers.  Callers may serialize them through ``to_dict()`` and
reconstruct them with the matching ``from_dict()`` helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
import time
from typing import Any, Mapping

SCHEMA_VERSION = 1
JOB_STATE_SCHEMA_VERSION = 2
SHARD_SCHEMA_VERSION = 2
RECORD_SCHEMA_VERSION = 2


def _require_schema_version(
    data: Mapping[str, Any], kind: str, expected: int = SCHEMA_VERSION
) -> int:
    version = data.get("schema_version")
    if type(version) is not int or version != expected:
        raise ValueError(f"{kind} schema_version must be {expected}")
    return version


class JobStatus(str, Enum):
    """Durable lifecycle states exposed by the Job subsystem."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    INTERRUPTED = "interrupted"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    COMPLETED_WITH_UNRESOLVED_WRITES = "completed_with_unresolved_writes"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecordStatus(str, Enum):
    """Durable per-record checkpoint states."""

    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    PENDING_COMMIT = "pending_commit"
    PERSISTED = "persisted"
    UNRESOLVED_WRITE = "unresolved_write"
    FAILED = "failed"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.COMPLETED_WITH_ERRORS,
        JobStatus.COMPLETED_WITH_UNRESOLVED_WRITES,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)
ACTIVE_JOB_STATUSES = frozenset({JobStatus.RUNNING, JobStatus.CANCELLING})
RECOVERABLE_JOB_STATUSES = frozenset(
    {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLING, JobStatus.INTERRUPTED}
)


def utc_timestamp() -> float:
    """Return a UTC Unix timestamp suitable for persisted ordering."""

    return time.time()


@dataclass(frozen=True)
class JobRequest:
    """Immutable, non-secret request metadata saved in ``request.json``."""

    job_id: str
    mode: str
    config_path: str
    config_sha256: str
    created_at: float = field(default_factory=utc_timestamp)
    options: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobRequest":
        return cls(
            schema_version=_require_schema_version(data, "job request"),
            job_id=str(data["job_id"]),
            mode=str(data["mode"]),
            config_path=str(data["config_path"]),
            config_sha256=str(data["config_sha256"]),
            created_at=float(data["created_at"]),
            options=dict(data.get("options") or {}),
        )


@dataclass
class JobCounts:
    """Monotonic and current counters persisted with a job state."""

    discovered: int = 0
    pending: int = 0
    in_flight: int = 0
    ai_complete: int = 0
    persisted: int = 0
    unresolved_writes: int = 0
    failed: int = 0
    cancelled: int = 0
    retries: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "JobCounts":
        raw = data or {}
        return cls(
            discovered=int(raw.get("discovered", 0)),
            pending=int(raw.get("pending", 0)),
            in_flight=int(raw.get("in_flight", 0)),
            ai_complete=int(raw.get("ai_complete", 0)),
            persisted=int(raw.get("persisted", 0)),
            unresolved_writes=int(raw.get("unresolved_writes", 0)),
            failed=int(raw.get("failed", 0)),
            cancelled=int(raw.get("cancelled", 0)),
            retries=int(raw.get("retries", 0)),
        )


@dataclass
class JobResourceState:
    """Resource-control state reported by the cooperative scheduler."""

    effective_max_in_flight: int = 1
    pressure: bool = False
    control_status: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "JobResourceState":
        raw = data or {}
        return cls(
            effective_max_in_flight=max(1, int(raw.get("effective_max_in_flight", 1))),
            pressure=bool(raw.get("pressure", False)),
            control_status=str(raw.get("control_status", "normal")),
        )


@dataclass
class JobState:
    """Revisioned durable job state saved atomically in ``state.json``."""

    revision: int
    job_id: str
    mode: str
    config_path: str
    config_sha256: str
    status: JobStatus
    created_at: float
    updated_at: float
    counts: JobCounts = field(default_factory=JobCounts)
    resource: JobResourceState = field(default_factory=JobResourceState)
    started_at: float | None = None
    finished_at: float | None = None
    last_error: str | None = None
    checkpoints: Mapping[str, Any] = field(default_factory=dict, repr=False)
    command_receipts: Mapping[str, Any] = field(default_factory=dict, repr=False)
    source_identity: Mapping[str, Any] | None = field(default=None, repr=False)
    schema_version: int = JOB_STATE_SCHEMA_VERSION

    @classmethod
    def initial(cls, request: JobRequest) -> "JobState":
        return cls(
            revision=0,
            job_id=request.job_id,
            mode=request.mode,
            config_path=request.config_path,
            config_sha256=request.config_sha256,
            status=JobStatus.QUEUED,
            created_at=request.created_at,
            updated_at=request.created_at,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES

    def to_dict(self) -> dict[str, Any]:
        data = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"checkpoints", "command_receipts", "source_identity"}
        }
        data["counts"] = self.counts.to_dict()
        data["resource"] = self.resource.to_dict()
        data["status"] = self.status.value
        return data

    def to_storage_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data["checkpoints"] = dict(self.checkpoints)
        data["command_receipts"] = dict(self.command_receipts)
        data["source_identity"] = (
            dict(self.source_identity) if self.source_identity is not None else None
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobState":
        version = _require_schema_version(data, "job state", JOB_STATE_SCHEMA_VERSION)
        revision = data.get("revision")
        if type(revision) is not int or revision < 0:
            raise ValueError("job state revision must be a non-negative integer")
        checkpoints = data.get("checkpoints")
        identity = data.get("source_identity")
        if identity is not None and not isinstance(identity, dict):
            raise ValueError("job source_identity must be an object")
        if not isinstance(checkpoints, dict):
            raise ValueError("job state checkpoints must be an object")
        for shard_id, payload in checkpoints.items():
            shard = ShardState.from_dict(payload)
            if shard.shard_id != shard_id or shard.job_id != data["job_id"]:
                raise ValueError("checkpoint identity does not match job state")
        return cls(
            schema_version=version,
            revision=revision,
            checkpoints=checkpoints,
            source_identity=identity,
            command_receipts=dict(data.get("command_receipts") or {}),
            job_id=str(data["job_id"]),
            mode=str(data["mode"]),
            config_path=str(data["config_path"]),
            config_sha256=str(data["config_sha256"]),
            status=JobStatus(str(data["status"])),
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
            started_at=(
                float(data["started_at"])
                if data.get("started_at") is not None
                else None
            ),
            finished_at=(
                float(data["finished_at"])
                if data.get("finished_at") is not None
                else None
            ),
            counts=JobCounts.from_dict(data.get("counts")),
            resource=JobResourceState.from_dict(data.get("resource")),
            last_error=(
                str(data["last_error"]) if data.get("last_error") is not None else None
            ),
        )


@dataclass(frozen=True)
class JobEvent:
    """One append-only event stored in ``events.jsonl``."""

    seq: int
    ts: float
    type: str
    job_id: str
    record_id: Any | None = None
    attempt: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobEvent":
        return cls(
            seq=int(data["seq"]),
            ts=float(data["ts"]),
            type=str(data["type"]),
            job_id=str(data["job_id"]),
            record_id=data.get("record_id"),
            attempt=(int(data["attempt"]) if data.get("attempt") is not None else None),
            payload=dict(data.get("payload") or {}),
        )


@dataclass(frozen=True)
class JobLease:
    """Worker ownership heartbeat persisted in ``lease.json``."""

    job_id: str
    owner_id: str
    acquired_at: float
    heartbeat_at: float
    heartbeat_interval_seconds: float = 5.0
    stale_after_seconds: float = 30.0
    schema_version: int = SCHEMA_VERSION

    def is_stale(self, now: float | None = None) -> bool:
        current = utc_timestamp() if now is None else now
        return current - self.heartbeat_at >= self.stale_after_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobLease":
        return cls(
            schema_version=_require_schema_version(data, "job lease"),
            job_id=str(data["job_id"]),
            owner_id=str(data["owner_id"]),
            acquired_at=float(data["acquired_at"]),
            heartbeat_at=float(data["heartbeat_at"]),
            heartbeat_interval_seconds=float(
                data.get("heartbeat_interval_seconds", 5.0)
            ),
            stale_after_seconds=float(data.get("stale_after_seconds", 30.0)),
        )


@dataclass(frozen=True)
class JobCommand:
    """Immutable command envelope stored under ``commands/``."""

    command_id: str
    job_id: str
    type: str
    created_at: float = field(default_factory=utc_timestamp)
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobCommand":
        return cls(
            schema_version=_require_schema_version(data, "job command"),
            command_id=str(data["command_id"]),
            job_id=str(data["job_id"]),
            type=str(data["type"]),
            created_at=float(data["created_at"]),
            payload=dict(data.get("payload") or {}),
        )


@dataclass(frozen=True)
class CommandReceipt:
    """Durable acknowledgement for a processed Job command."""

    command_id: str
    job_id: str
    accepted: bool
    processed_at: float = field(default_factory=utc_timestamp)
    message: str | None = None
    resulting_status: JobStatus | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resulting_status"] = (
            self.resulting_status.value if self.resulting_status is not None else None
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CommandReceipt":
        raw_status = data.get("resulting_status")
        return cls(
            schema_version=_require_schema_version(data, "command receipt"),
            command_id=str(data["command_id"]),
            job_id=str(data["job_id"]),
            accepted=bool(data["accepted"]),
            processed_at=float(data["processed_at"]),
            message=(str(data["message"]) if data.get("message") is not None else None),
            resulting_status=(JobStatus(str(raw_status)) if raw_status else None),
        )


@dataclass(frozen=True)
class RecordCheckpoint:
    """One resumable record snapshot stored inside a shard checkpoint."""

    record_id: Any
    status: RecordStatus
    attempt: int = 0
    retry_counts: Mapping[str, int] = field(default_factory=dict)
    input_data: Mapping[str, Any] | None = None
    prepared_ref: str | None = None
    prepared_hash: str | None = None
    commit_id: str | None = None
    commit_attempts: int = 0
    reconciliation_attempts: int = 0
    error: Mapping[str, Any] | None = None
    updated_at: float = field(default_factory=utc_timestamp)
    schema_version: int = RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecordCheckpoint":
        version = data.get("schema_version")
        if type(version) is not int or version != RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"record checkpoint schema_version must be {RECORD_SCHEMA_VERSION}"
            )
        return cls(
            schema_version=version,
            record_id=data.get("record_id"),
            status=RecordStatus(str(data["status"])),
            attempt=int(data.get("attempt", 0)),
            retry_counts={
                str(key): int(value)
                for key, value in (data.get("retry_counts") or {}).items()
            },
            input_data=(
                dict(data["input_data"])
                if isinstance(data.get("input_data"), Mapping)
                else None
            ),
            prepared_ref=(
                str(data["prepared_ref"])
                if data.get("prepared_ref") is not None
                else None
            ),
            prepared_hash=(
                str(data["prepared_hash"])
                if data.get("prepared_hash") is not None
                else None
            ),
            commit_id=(
                str(data["commit_id"]) if data.get("commit_id") is not None else None
            ),
            commit_attempts=int(data.get("commit_attempts", 0)),
            reconciliation_attempts=int(data.get("reconciliation_attempts", 0)),
            error=(
                dict(data["error"]) if isinstance(data.get("error"), Mapping) else None
            ),
            updated_at=float(data.get("updated_at", utc_timestamp())),
        )


@dataclass(frozen=True)
class ShardState:
    """Generic shard metadata; datasource-specific cursor payloads stay opaque."""

    shard_id: str
    job_id: str
    status: str
    updated_at: float = field(default_factory=utc_timestamp)
    cursor: Any | None = None
    counts: Mapping[str, int] = field(default_factory=dict)
    records: tuple[RecordCheckpoint, ...] = ()
    schema_version: int = SHARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["records"] = [record.to_dict() for record in self.records]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShardState":
        version = data.get("schema_version")
        if type(version) is not int or version != SHARD_SCHEMA_VERSION:
            raise ValueError(f"shard schema_version must be {SHARD_SCHEMA_VERSION}")
        return cls(
            schema_version=version,
            shard_id=str(data["shard_id"]),
            job_id=str(data["job_id"]),
            status=str(data["status"]),
            updated_at=float(data["updated_at"]),
            cursor=data.get("cursor"),
            counts={
                str(key): int(value)
                for key, value in (data.get("counts") or {}).items()
            },
            records=tuple(
                RecordCheckpoint.from_dict(item) for item in (data.get("records") or [])
            ),
        )


@dataclass(frozen=True)
class RecoveryDecision:
    """Pure recovery decision; callers choose when to persist it."""

    job_id: str
    recover: bool
    target_status: JobStatus
    reason: str


@dataclass(frozen=True)
class PruneCandidate:
    """One repository entry selected by a prune preview."""

    job_id: str
    status: JobStatus
    revision: int
    updated_at: float
    size_bytes: int


@dataclass(frozen=True)
class PrunePreview:
    """Immutable snapshot that must be passed back for confirmed pruning."""

    repository_root: str
    created_at: float
    older_than: float
    statuses: tuple[JobStatus, ...]
    candidates: tuple[PruneCandidate, ...]

    @property
    def total_bytes(self) -> int:
        return sum(candidate.size_bytes for candidate in self.candidates)


@dataclass(frozen=True)
class PruneResult:
    """Outcome of a dry-run or explicitly confirmed prune operation."""

    confirmed: bool
    selected_job_ids: tuple[str, ...]
    deleted_job_ids: tuple[str, ...]
    reclaimed_bytes: int
