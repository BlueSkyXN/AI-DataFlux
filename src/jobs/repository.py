"""Modular, file-backed Job Repository.

Each UUID4 job is stored under ``<root>/<job_id>/`` with immutable request
metadata, atomically replaced state/lease/shard JSON, an append-only event log,
and immutable command envelopes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterator, Mapping
from uuid import UUID, uuid4

from .io import append_jsonl, atomic_write_json, iter_file_size, read_json
from .io import read_jsonl_tolerant, repair_jsonl_tail
from .models import (
    ACTIVE_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    CommandReceipt,
    JobCommand,
    JobEvent,
    JobLease,
    JobRequest,
    JobState,
    JobStatus,
    PruneCandidate,
    PrunePreview,
    PruneResult,
    RecoveryDecision,
    ShardState,
    utc_timestamp,
)

DEFAULT_HEARTBEAT_SECONDS = 5.0
DEFAULT_LEASE_STALE_SECONDS = 30.0
_SECRET_OPTION_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "app_secret",
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "refresh_token",
        "secret",
        "token",
        "access_token",
    }
)


class JobRepositoryError(RuntimeError):
    """Base class for repository consistency errors."""


class JobNotFoundError(JobRepositoryError):
    """Raised when a requested Job directory does not exist."""


class RevisionConflictError(JobRepositoryError):
    """Raised when optimistic state revision validation fails."""


class LeaseConflictError(JobRepositoryError):
    """Raised when a fresh lease is already owned by another worker."""


class ImmutableRecordError(JobRepositoryError):
    """Raised when immutable request or command metadata already exists."""


def compute_config_hash(config: Any) -> str:
    """Hash a config-like object using deterministic canonical JSON."""

    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_config_file(path: Path | str) -> str:
    """Hash exact config-file bytes without parsing or exposing its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_uuid4(value: str) -> bool:
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        return False
    return parsed.version == 4 and str(parsed) == value.lower()


def _assert_non_secret_options(value: Any, path: str = "options") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SECRET_OPTION_KEYS or normalized.endswith(
                ("_password", "_secret", "_access_token", "_refresh_token")
            ):
                raise ValueError(
                    f"secret-like Job request option is not allowed: {path}.{key}"
                )
            _assert_non_secret_options(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_non_secret_options(item, f"{path}[{index}]")


@contextmanager
def _exclusive_file_lock(
    path: Path,
    *,
    timeout_seconds: float = 5.0,
    stale_after_seconds: float = 60.0,
) -> Iterator[None]:
    """Portable inter-process lock using exclusive file creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                age = utc_timestamp() - path.stat().st_mtime
                if age >= stale_after_seconds:
                    path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring repository lock: {path}")
            time.sleep(0.01)
            continue
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        finally:
            os.close(descriptor)
        break

    try:
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class FileJobRepository:
    """Durable repository rooted at ``.dataflux/jobs`` by default."""

    def __init__(self, root: Path | str = Path(".dataflux/jobs")):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        self._validate_job_id(job_id)
        return self.root / job_id

    def request_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "request.json"

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "state.json"

    def events_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "events.jsonl"

    def lease_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "lease.json"

    def shards_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "shards"

    def prepared_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "prepared"

    def commands_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "commands"

    def prepared_result_path(self, job_id: str, commit_id: str) -> Path:
        if len(commit_id) != 32 or any(
            character not in "0123456789abcdef" for character in commit_id
        ):
            raise ValueError("commit_id must be a lowercase UUID hex value")
        return self.prepared_dir(job_id) / f"{commit_id}.json"

    @staticmethod
    def _validate_job_id(job_id: str) -> None:
        if not _is_uuid4(job_id):
            raise ValueError(f"job_id must be a canonical UUID4: {job_id!r}")

    def _require_job(self, job_id: str) -> Path:
        directory = self.job_dir(job_id)
        if not directory.is_dir() or directory.is_symlink():
            raise JobNotFoundError(job_id)
        return directory

    def _job_lock(self, job_id: str) -> Path:
        return self.job_dir(job_id) / ".repository.lock"

    def new_request(
        self,
        *,
        mode: str,
        config_path: str,
        config_sha256: str,
        options: Mapping[str, Any] | None = None,
        job_id: str | None = None,
        created_at: float | None = None,
    ) -> JobRequest:
        """Create request metadata without writing it."""

        actual_job_id = job_id or str(uuid4())
        self._validate_job_id(actual_job_id)
        if len(config_sha256) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in config_sha256
        ):
            raise ValueError("config_sha256 must be a 64-character hexadecimal hash")
        normalized_options = dict(options or {})
        _assert_non_secret_options(normalized_options)
        return JobRequest(
            job_id=actual_job_id,
            mode=mode,
            config_path=config_path,
            config_sha256=config_sha256.lower(),
            created_at=utc_timestamp() if created_at is None else created_at,
            options=normalized_options,
        )

    def create_job(self, request: JobRequest) -> JobState:
        """Create the complete Job directory and initial queued state."""

        self._validate_job_id(request.job_id)
        _assert_non_secret_options(request.options)
        directory = self.job_dir(request.job_id)
        try:
            directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise ImmutableRecordError(
                f"job already exists: {request.job_id}"
            ) from error

        try:
            self.shards_dir(request.job_id).mkdir()
            self.prepared_dir(request.job_id).mkdir()
            self.commands_dir(request.job_id).mkdir()
            atomic_write_json(self.request_path(request.job_id), request.to_dict())
            state = JobState.initial(request)
            atomic_write_json(self.state_path(request.job_id), state.to_dict())
            self.append_event(
                request.job_id, "job_created", payload={"mode": request.mode}
            )
            return state
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def get_request(self, job_id: str) -> JobRequest:
        self._require_job(job_id)
        return JobRequest.from_dict(read_json(self.request_path(job_id)))

    def get_state(self, job_id: str) -> JobState:
        self._require_job(job_id)
        return JobState.from_dict(read_json(self.state_path(job_id)))

    def save_state(
        self,
        state: JobState,
        *,
        expected_revision: int | None = None,
    ) -> JobState:
        """Persist state with optimistic revision checking and atomic replace."""

        self._require_job(state.job_id)
        with _exclusive_file_lock(self._job_lock(state.job_id)):
            current = self.get_state(state.job_id)
            if expected_revision is not None and current.revision != expected_revision:
                raise RevisionConflictError(
                    f"expected revision {expected_revision}, found {current.revision}"
                )
            if state.job_id != current.job_id:
                raise ValueError("job state identity cannot change")
            persisted = replace(
                state,
                revision=current.revision + 1,
                updated_at=utc_timestamp(),
            )
            atomic_write_json(self.state_path(state.job_id), persisted.to_dict())
            return persisted

    def update_state(
        self,
        job_id: str,
        updater: Callable[[JobState], JobState],
        *,
        expected_revision: int | None = None,
    ) -> JobState:
        """Apply a pure state transform under the repository lock."""

        self._require_job(job_id)
        with _exclusive_file_lock(self._job_lock(job_id)):
            current = self.get_state(job_id)
            if expected_revision is not None and current.revision != expected_revision:
                raise RevisionConflictError(
                    f"expected revision {expected_revision}, found {current.revision}"
                )
            candidate = updater(current)
            if candidate.job_id != job_id:
                raise ValueError("job state identity cannot change")
            persisted = replace(
                candidate,
                revision=current.revision + 1,
                updated_at=utc_timestamp(),
            )
            atomic_write_json(self.state_path(job_id), persisted.to_dict())
            return persisted

    def transition(
        self,
        job_id: str,
        status: JobStatus,
        *,
        last_error: str | None = None,
        expected_revision: int | None = None,
    ) -> JobState:
        """Transition status and maintain start/finish timestamps."""

        now = utc_timestamp()

        def apply(current: JobState) -> JobState:
            started_at = current.started_at
            finished_at = current.finished_at
            if status == JobStatus.RUNNING and started_at is None:
                started_at = now
            if status in TERMINAL_JOB_STATUSES:
                finished_at = now
            elif status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.INTERRUPTED}:
                finished_at = None
            return replace(
                current,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                last_error=last_error,
            )

        state = self.update_state(job_id, apply, expected_revision=expected_revision)
        self.append_event(
            job_id,
            "status_changed",
            payload={"status": status.value, "revision": state.revision},
        )
        return state

    def append_event(
        self,
        job_id: str,
        event_type: str,
        *,
        record_id: Any | None = None,
        attempt: int | None = None,
        payload: Mapping[str, Any] | None = None,
        ts: float | None = None,
    ) -> JobEvent:
        self._require_job(job_id)
        lock_path = self.job_dir(job_id) / ".events.lock"
        with _exclusive_file_lock(lock_path):
            repair_jsonl_tail(self.events_path(job_id))
            existing = self.list_events(job_id)
            seq = existing[-1].seq + 1 if existing else 1
            event = JobEvent(
                seq=seq,
                ts=utc_timestamp() if ts is None else ts,
                type=event_type,
                job_id=job_id,
                record_id=record_id,
                attempt=attempt,
                payload=dict(payload or {}),
            )
            append_jsonl(self.events_path(job_id), event.to_dict())
            return event

    def list_events(self, job_id: str) -> list[JobEvent]:
        self._require_job(job_id)
        return [
            JobEvent.from_dict(item)
            for item in read_jsonl_tolerant(self.events_path(job_id))
        ]

    def get_lease(self, job_id: str) -> JobLease | None:
        self._require_job(job_id)
        path = self.lease_path(job_id)
        if not path.exists():
            return None
        return JobLease.from_dict(read_json(path))

    def acquire_lease(
        self,
        job_id: str,
        owner_id: str,
        *,
        now: float | None = None,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        stale_after_seconds: float = DEFAULT_LEASE_STALE_SECONDS,
    ) -> JobLease:
        self._require_job(job_id)
        current_time = utc_timestamp() if now is None else now
        with _exclusive_file_lock(self._job_lock(job_id)):
            existing = self.get_lease(job_id)
            if (
                existing is not None
                and existing.owner_id != owner_id
                and not existing.is_stale(current_time)
            ):
                raise LeaseConflictError(
                    f"job {job_id} is leased by {existing.owner_id}"
                )
            acquired_at = (
                existing.acquired_at
                if existing is not None and existing.owner_id == owner_id
                else current_time
            )
            lease = JobLease(
                job_id=job_id,
                owner_id=owner_id,
                acquired_at=acquired_at,
                heartbeat_at=current_time,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                stale_after_seconds=stale_after_seconds,
            )
            atomic_write_json(self.lease_path(job_id), lease.to_dict())
            return lease

    def heartbeat_lease(
        self, job_id: str, owner_id: str, *, now: float | None = None
    ) -> JobLease:
        self._require_job(job_id)
        current_time = utc_timestamp() if now is None else now
        with _exclusive_file_lock(self._job_lock(job_id)):
            existing = self.get_lease(job_id)
            if existing is None or existing.owner_id != owner_id:
                raise LeaseConflictError(f"worker {owner_id} does not own job {job_id}")
            lease = replace(existing, heartbeat_at=current_time)
            atomic_write_json(self.lease_path(job_id), lease.to_dict())
            return lease

    def release_lease(self, job_id: str, owner_id: str) -> None:
        self._require_job(job_id)
        with _exclusive_file_lock(self._job_lock(job_id)):
            existing = self.get_lease(job_id)
            if existing is None:
                return
            if existing.owner_id != owner_id:
                raise LeaseConflictError(f"worker {owner_id} does not own job {job_id}")
            self.lease_path(job_id).unlink(missing_ok=True)

    def save_shard(self, shard: ShardState) -> None:
        self._require_job(shard.job_id)
        if not shard.shard_id or any(char in shard.shard_id for char in "/\\"):
            raise ValueError("shard_id must be a simple file name")
        atomic_write_json(
            self.shards_dir(shard.job_id) / f"{shard.shard_id}.json",
            shard.to_dict(),
        )

    def save_prepared_result(
        self,
        job_id: str,
        commit_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        """Atomically persist one immutable PreparedResult blob."""

        self._require_job(job_id)
        if payload.get("commit_id") != commit_id:
            raise ValueError("prepared result commit_id does not match blob path")
        path = self.prepared_result_path(job_id, commit_id)
        reference = f"prepared/{commit_id}.json"
        with _exclusive_file_lock(self._job_lock(job_id)):
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_file():
                    raise ImmutableRecordError(
                        f"invalid prepared result blob: {commit_id}"
                    )
                if read_json(path) != dict(payload):
                    raise ImmutableRecordError(
                        f"prepared result commit_id collision: {commit_id}"
                    )
                return reference
            atomic_write_json(path, dict(payload))
        return reference

    def load_prepared_result(
        self,
        job_id: str,
        reference: str,
    ) -> dict[str, Any]:
        """Load one checkpoint-referenced blob without following arbitrary paths."""

        self._require_job(job_id)
        candidate = Path(reference)
        if (
            candidate.is_absolute()
            or candidate.parts[:1] != ("prepared",)
            or len(candidate.parts) != 2
            or candidate.suffix != ".json"
        ):
            raise JobRepositoryError(
                f"invalid prepared result reference: {reference!r}"
            )
        commit_id = candidate.stem
        path = self.prepared_result_path(job_id, commit_id)
        if path.is_symlink() or not path.is_file():
            raise JobRepositoryError(f"prepared result blob is missing: {reference!r}")
        return read_json(path)

    def get_shard(self, job_id: str, shard_id: str) -> ShardState:
        self._require_job(job_id)
        return ShardState.from_dict(
            read_json(self.shards_dir(job_id) / f"{shard_id}.json")
        )

    def list_shards(self, job_id: str) -> list[ShardState]:
        self._require_job(job_id)
        return [
            ShardState.from_dict(read_json(path))
            for path in sorted(self.shards_dir(job_id).glob("*.json"))
            if not path.is_symlink()
        ]

    def add_command(self, command: JobCommand) -> None:
        self._require_job(command.job_id)
        if not command.command_id or any(char in command.command_id for char in "/\\"):
            raise ValueError("command_id must be a simple file name")
        path = self.commands_dir(command.job_id) / f"{command.command_id}.json"
        with _exclusive_file_lock(self._job_lock(command.job_id)):
            if path.exists():
                raise ImmutableRecordError(
                    f"command already exists: {command.command_id}"
                )
            atomic_write_json(path, command.to_dict())

    def list_commands(self, job_id: str) -> list[JobCommand]:
        self._require_job(job_id)
        return sorted(
            (
                JobCommand.from_dict(read_json(path))
                for path in self.commands_dir(job_id).glob("*.json")
                if not path.name.endswith(".receipt.json") and not path.is_symlink()
            ),
            key=lambda command: (command.created_at, command.command_id),
        )

    def save_command_receipt(self, receipt: CommandReceipt) -> None:
        self._require_job(receipt.job_id)
        path = self.commands_dir(receipt.job_id) / f"{receipt.command_id}.receipt.json"
        with _exclusive_file_lock(self._job_lock(receipt.job_id)):
            if path.exists():
                raise ImmutableRecordError(
                    f"command receipt already exists: {receipt.command_id}"
                )
            atomic_write_json(path, receipt.to_dict())

    def get_command_receipt(
        self, job_id: str, command_id: str
    ) -> CommandReceipt | None:
        self._require_job(job_id)
        path = self.commands_dir(job_id) / f"{command_id}.receipt.json"
        if not path.exists():
            return None
        return CommandReceipt.from_dict(read_json(path))

    def list_job_ids(self) -> list[str]:
        job_ids = [
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and not path.is_symlink() and _is_uuid4(path.name)
        ]
        return sorted(job_ids, key=lambda job_id: self.get_state(job_id).created_at)

    def evaluate_recovery(
        self,
        job_id: str,
        is_resumable: bool | Callable[[JobState, JobRequest], bool],
        *,
        now: float | None = None,
    ) -> RecoveryDecision:
        """Decide recovery without importing or probing a datasource."""

        state = self.get_state(job_id)
        request = self.get_request(job_id)
        if state.is_terminal:
            return RecoveryDecision(job_id, False, state.status, "terminal")

        lease = self.get_lease(job_id)
        current_time = utc_timestamp() if now is None else now
        if (
            state.status in ACTIVE_JOB_STATUSES
            and lease is not None
            and not lease.is_stale(current_time)
        ):
            return RecoveryDecision(job_id, False, state.status, "lease_active")

        resumable = (
            bool(is_resumable(state, request))
            if callable(is_resumable)
            else bool(is_resumable)
        )
        if resumable:
            return RecoveryDecision(job_id, True, JobStatus.QUEUED, "resumable")
        return RecoveryDecision(job_id, False, JobStatus.BLOCKED, "not_resumable")

    def apply_recovery_decision(self, decision: RecoveryDecision) -> JobState:
        """Persist a previously evaluated recovery decision."""

        state = self.transition(
            decision.job_id,
            decision.target_status,
            last_error=None if decision.recover else decision.reason,
        )
        self.append_event(
            decision.job_id,
            "recovery_decision",
            payload={
                "recover": decision.recover,
                "target_status": decision.target_status.value,
                "reason": decision.reason,
            },
        )
        return state

    def preview_prune(
        self,
        *,
        older_than: float,
        statuses: set[JobStatus] | frozenset[JobStatus] | None = None,
    ) -> PrunePreview:
        """Return a dry-run snapshot; this method never deletes files."""

        selected_statuses = frozenset(statuses or TERMINAL_JOB_STATUSES)
        candidates: list[PruneCandidate] = []
        for job_id in self.list_job_ids():
            state = self.get_state(job_id)
            if state.status not in selected_statuses or state.updated_at >= older_than:
                continue
            directory = self.job_dir(job_id)
            size = iter_file_size(directory.rglob("*"))
            candidates.append(
                PruneCandidate(
                    job_id=job_id,
                    status=state.status,
                    revision=state.revision,
                    updated_at=state.updated_at,
                    size_bytes=size,
                )
            )
        return PrunePreview(
            repository_root=str(self.root),
            created_at=utc_timestamp(),
            older_than=older_than,
            statuses=tuple(sorted(selected_statuses, key=lambda item: item.value)),
            candidates=tuple(candidates),
        )

    def prune(self, preview: PrunePreview, *, confirm: bool = False) -> PruneResult:
        """Delete only unchanged preview candidates and only with explicit confirm."""

        if Path(preview.repository_root).resolve() != self.root:
            raise ValueError("prune preview belongs to another repository")
        selected = tuple(candidate.job_id for candidate in preview.candidates)
        if not confirm:
            return PruneResult(False, selected, (), 0)

        deleted: list[str] = []
        reclaimed = 0
        for candidate in preview.candidates:
            directory = self.job_dir(candidate.job_id)
            if not directory.is_dir() or directory.is_symlink():
                continue
            with _exclusive_file_lock(self._job_lock(candidate.job_id)):
                state = self.get_state(candidate.job_id)
                if (
                    state.revision != candidate.revision
                    or state.status != candidate.status
                    or state.updated_at != candidate.updated_at
                    or state.updated_at >= preview.older_than
                ):
                    continue
                shutil.rmtree(directory)
                deleted.append(candidate.job_id)
                reclaimed += candidate.size_bytes
        return PruneResult(True, selected, tuple(deleted), reclaimed)
