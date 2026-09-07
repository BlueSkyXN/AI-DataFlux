"""Durable per-record checkpoints used by background Job execution."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from src.jobs import (
    FileJobRepository,
    JobCounts,
    JobRepositoryError,
    RecordCheckpoint,
    RecordStatus,
    ShardState,
)

from .contracts import PreparedResult


class JobRecordTracker:
    """Persist record transitions before exposing them as Job progress."""

    def __init__(
        self,
        repository: FileJobRepository,
        job_id: str,
        *,
        shard_id: str = "records",
    ) -> None:
        self.repository = repository
        self.job_id = job_id
        self.shard_id = shard_id
        self._load_committed_state()

    def _load_committed_state(self) -> None:
        self._records: dict[str, RecordCheckpoint] = {}
        self._record_shards: dict[str, str] = {}
        self._shards: dict[str, ShardState] = {}
        self._next_scan_shard = 1

        for shard in self.repository.list_shards(self.job_id):
            self._shards[shard.shard_id] = shard
            if shard.shard_id.startswith("scan-"):
                try:
                    self._next_scan_shard = max(
                        self._next_scan_shard,
                        int(shard.shard_id.removeprefix("scan-")) + 1,
                    )
                except ValueError:
                    pass
            for record in shard.records:
                key = self._key(record.record_id)
                current = self._records.get(key)
                if current is None or record.updated_at >= current.updated_at:
                    self._records[key] = record
                    self._record_shards[key] = shard.shard_id

    @staticmethod
    def _key(record_id: Any) -> str:
        return json.dumps(
            record_id,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def get(self, record_id: Any) -> RecordCheckpoint | None:
        return self._records.get(self._key(record_id))

    def retry_counts_for(self, record_id: Any) -> dict[str, int]:
        checkpoint = self.get(record_id)
        return dict(checkpoint.retry_counts) if checkpoint is not None else {}

    def commit_attempts_for(self, record_id: Any) -> int:
        checkpoint = self.get(record_id)
        return checkpoint.commit_attempts if checkpoint is not None else 0

    def reconciliation_attempts_for(self, record_id: Any) -> int:
        checkpoint = self.get(record_id)
        return checkpoint.reconciliation_attempts if checkpoint is not None else 0

    def should_process_scanned(self, record_id: Any) -> bool:
        """Only unfinished model work may re-enter the model execution path."""

        checkpoint = self.get(record_id)
        if checkpoint is None:
            return True
        return checkpoint.status in {RecordStatus.PENDING, RecordStatus.IN_FLIGHT}

    def new_scan_shard_id(self) -> str:
        shard_id = f"scan-{self._next_scan_shard:06d}"
        self._next_scan_shard += 1
        return shard_id

    def register_scan_batch(
        self,
        shard_id: str,
        records: Iterable[tuple[Any, Mapping[str, Any]]],
        *,
        cursor: Any | None,
    ) -> None:
        """Checkpoint a datasource page before any AI request is issued."""

        if not shard_id or any(char in shard_id for char in "/\\"):
            raise ValueError("shard_id must be a simple file name")

        registered = 0
        for record_id, input_data in records:
            key = self._key(record_id)
            if key in self._records:
                continue
            self._records[key] = RecordCheckpoint(
                record_id=record_id,
                status=RecordStatus.PENDING,
                input_data=dict(input_data),
            )
            self._record_shards[key] = shard_id
            registered += 1

        previous = self._shards.get(shard_id)
        self._shards[shard_id] = ShardState(
            shard_id=shard_id,
            job_id=self.job_id,
            status=previous.status if previous is not None else "active",
            cursor=cursor,
        )
        self._persist_shard(shard_id)
        counts = self._publish_counts()
        self.repository.append_event(
            self.job_id,
            "shard_checkpointed",
            payload={
                "shard_id": shard_id,
                "record_count": registered,
                "discovered": counts.discovered,
            },
        )

    def mark_in_flight(
        self,
        record_id: Any,
        input_data: Mapping[str, Any],
    ) -> RecordCheckpoint:
        current = self.get(record_id)
        attempt = (current.attempt if current else 0) + 1
        return self._transition(
            record_id,
            RecordStatus.IN_FLIGHT,
            attempt=attempt,
            input_data=dict(input_data),
            retry_counts=dict(current.retry_counts) if current else {},
            event_type="record_in_flight",
        )

    def mark_pending(
        self,
        record_id: Any,
        input_data: Mapping[str, Any],
        *,
        error_code: str | None = None,
        retry_error_type: str | None = None,
    ) -> RecordCheckpoint:
        current = self.get(record_id)
        retry_counts = dict(current.retry_counts) if current else {}
        if retry_error_type:
            retry_counts[retry_error_type] = retry_counts.get(retry_error_type, 0) + 1
        return self._transition(
            record_id,
            RecordStatus.PENDING,
            attempt=current.attempt if current else 0,
            input_data=dict(input_data),
            error={"code": error_code} if error_code else None,
            retry_counts=retry_counts,
            event_type="record_retry_scheduled",
        )

    def mark_prepared(self, prepared: PreparedResult) -> RecordCheckpoint:
        """Persist blob first, then publish its immutable checkpoint reference."""

        prepared.validate()
        current = self.get(prepared.record_id)
        reference = self.repository.save_prepared_result(
            self.job_id,
            prepared.commit_id,
            prepared.to_dict(),
        )
        return self._transition(
            prepared.record_id,
            RecordStatus.PENDING_COMMIT,
            attempt=current.attempt if current else 1,
            input_data=current.input_data if current else None,
            prepared_ref=reference,
            prepared_hash=prepared.payload_hash,
            commit_id=prepared.commit_id,
            retry_counts=dict(current.retry_counts) if current else {},
            event_type="record_pending_commit",
        )

    def mark_commit_attempt(self, record_id: Any) -> RecordCheckpoint:
        current = self._require_prepared_checkpoint(record_id)
        return self._transition_from_prepared(
            current,
            RecordStatus.PENDING_COMMIT,
            commit_attempts=current.commit_attempts + 1,
            reconciliation_attempts=current.reconciliation_attempts,
            event_type="record_commit_attempted",
        )

    def mark_reconciliation_attempt(self, record_id: Any) -> RecordCheckpoint:
        current = self._require_prepared_checkpoint(record_id)
        return self._transition_from_prepared(
            current,
            RecordStatus.PENDING_COMMIT,
            commit_attempts=current.commit_attempts,
            reconciliation_attempts=current.reconciliation_attempts + 1,
            event_type="record_reconciliation_attempted",
        )

    def mark_persisted(self, record_id: Any) -> RecordCheckpoint:
        current = self._require_prepared_checkpoint(record_id)
        return self._transition_from_prepared(
            current,
            RecordStatus.PERSISTED,
            commit_attempts=current.commit_attempts,
            reconciliation_attempts=current.reconciliation_attempts,
            event_type="record_persisted",
        )

    def mark_unresolved(
        self,
        record_id: Any,
        *,
        error_code: str = "unresolved_write",
    ) -> RecordCheckpoint:
        current = self._require_prepared_checkpoint(record_id)
        return self._transition_from_prepared(
            current,
            RecordStatus.UNRESOLVED_WRITE,
            commit_attempts=current.commit_attempts,
            reconciliation_attempts=current.reconciliation_attempts,
            error={"code": error_code},
            event_type="record_write_unresolved",
        )

    def mark_failed(
        self,
        record_id: Any,
        *,
        error_code: str,
    ) -> RecordCheckpoint:
        current = self.get(record_id)
        if current is not None and current.prepared_ref is not None:
            return self._transition_from_prepared(
                current,
                RecordStatus.FAILED,
                commit_attempts=current.commit_attempts,
                reconciliation_attempts=current.reconciliation_attempts,
                error={"code": error_code},
                event_type="record_failed",
            )
        return self._transition(
            record_id,
            RecordStatus.FAILED,
            attempt=current.attempt if current else 0,
            input_data=current.input_data if current else None,
            error={"code": error_code},
            retry_counts=dict(current.retry_counts) if current else {},
            event_type="record_failed",
        )

    def pending_prepared_results(self) -> dict[Any, PreparedResult]:
        """Load only checkpoint-referenced blobs; orphan blobs stay inert."""

        prepared_results: dict[Any, PreparedResult] = {}
        for checkpoint in self._records.values():
            if checkpoint.status != RecordStatus.PENDING_COMMIT:
                continue
            if not (
                checkpoint.prepared_ref
                and checkpoint.prepared_hash
                and checkpoint.commit_id
            ):
                raise JobRepositoryError(
                    f"pending commit record lacks prepared reference: {checkpoint.record_id!r}"
                )
            payload = self.repository.load_prepared_result(
                self.job_id,
                checkpoint.prepared_ref,
            )
            try:
                prepared = PreparedResult.from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise JobRepositoryError(
                    f"invalid prepared result blob: {checkpoint.prepared_ref}"
                ) from exc
            if self._key(prepared.record_id) != self._key(checkpoint.record_id):
                raise JobRepositoryError("prepared result record_id mismatch")
            if prepared.commit_id != checkpoint.commit_id:
                raise JobRepositoryError("prepared result commit_id mismatch")
            if prepared.payload_hash != checkpoint.prepared_hash:
                raise JobRepositoryError("prepared result checkpoint hash mismatch")
            prepared_results[checkpoint.record_id] = prepared
        return prepared_results

    def counts(self, *, cancelled: int | None = None) -> JobCounts:
        records = tuple(self._records.values())
        retries = sum(sum(record.retry_counts.values()) for record in records)
        current_state = self.repository.get_state(self.job_id)
        return JobCounts(
            discovered=len(records),
            pending=sum(record.status == RecordStatus.PENDING for record in records),
            in_flight=sum(
                record.status == RecordStatus.IN_FLIGHT for record in records
            ),
            ai_complete=sum(record.prepared_ref is not None for record in records),
            persisted=sum(
                record.status == RecordStatus.PERSISTED for record in records
            ),
            unresolved_writes=sum(
                record.status == RecordStatus.UNRESOLVED_WRITE for record in records
            ),
            failed=sum(record.status == RecordStatus.FAILED for record in records),
            cancelled=(
                current_state.counts.cancelled if cancelled is None else cancelled
            ),
            retries=retries,
        )

    def _require_prepared_checkpoint(self, record_id: Any) -> RecordCheckpoint:
        current = self.get(record_id)
        if current is None or not (
            current.prepared_ref and current.prepared_hash and current.commit_id
        ):
            raise JobRepositoryError(
                f"record has no durable PreparedResult: {record_id!r}"
            )
        return current

    def _transition_from_prepared(
        self,
        current: RecordCheckpoint,
        status: RecordStatus,
        *,
        commit_attempts: int,
        reconciliation_attempts: int,
        event_type: str,
        error: Mapping[str, Any] | None = None,
    ) -> RecordCheckpoint:
        return self._transition(
            current.record_id,
            status,
            attempt=current.attempt,
            input_data=current.input_data,
            prepared_ref=current.prepared_ref,
            prepared_hash=current.prepared_hash,
            commit_id=current.commit_id,
            commit_attempts=commit_attempts,
            reconciliation_attempts=reconciliation_attempts,
            error=error,
            retry_counts=dict(current.retry_counts),
            event_type=event_type,
        )

    def _transition(
        self,
        record_id: Any,
        status: RecordStatus,
        *,
        attempt: int,
        input_data: Mapping[str, Any] | None = None,
        prepared_ref: str | None = None,
        prepared_hash: str | None = None,
        commit_id: str | None = None,
        commit_attempts: int = 0,
        reconciliation_attempts: int = 0,
        error: Mapping[str, Any] | None = None,
        retry_counts: Mapping[str, int] | None = None,
        event_type: str,
    ) -> RecordCheckpoint:
        checkpoint = RecordCheckpoint(
            record_id=record_id,
            status=status,
            attempt=attempt,
            retry_counts=dict(retry_counts or {}),
            input_data=input_data,
            prepared_ref=prepared_ref,
            prepared_hash=prepared_hash,
            commit_id=commit_id,
            commit_attempts=commit_attempts,
            reconciliation_attempts=reconciliation_attempts,
            error=error,
        )
        key = self._key(record_id)
        self._records[key] = checkpoint
        target_shard = self._record_shards.setdefault(key, self.shard_id)
        self._persist_shard(target_shard)
        self._publish_counts()
        self.repository.append_event(
            self.job_id,
            event_type,
            record_id=record_id,
            attempt=attempt,
            payload={
                "status": status.value,
                **({"error_code": error.get("code")} if error else {}),
            },
        )
        return checkpoint

    def _persist_shard(self, shard_id: str) -> None:
        previous = self._shards.get(shard_id)
        records = tuple(
            self._records[key]
            for key in sorted(self._records)
            if self._record_shards.get(key) == shard_id
        )
        terminal = {
            RecordStatus.PERSISTED,
            RecordStatus.UNRESOLVED_WRITE,
            RecordStatus.FAILED,
        }
        status = (
            "completed"
            if records and all(record.status in terminal for record in records)
            else "active"
        )
        shard = ShardState(
            shard_id=shard_id,
            job_id=self.job_id,
            status=status,
            cursor=previous.cursor if previous is not None else None,
            counts=self._counts_for(records),
            records=records,
        )
        try:
            self.repository.save_shard(shard)
        except BaseException:
            self._load_committed_state()
            raise
        self._shards[shard_id] = shard

    def _publish_counts(self) -> JobCounts:
        return self.repository.get_state(self.job_id).counts

    @staticmethod
    def _counts_for(records: tuple[RecordCheckpoint, ...]) -> dict[str, int]:
        return {
            "discovered": len(records),
            "pending": sum(record.status == RecordStatus.PENDING for record in records),
            "in_flight": sum(
                record.status == RecordStatus.IN_FLIGHT for record in records
            ),
            "ai_complete": sum(record.prepared_ref is not None for record in records),
            "persisted": sum(
                record.status == RecordStatus.PERSISTED for record in records
            ),
            "unresolved_writes": sum(
                record.status == RecordStatus.UNRESOLVED_WRITE for record in records
            ),
            "failed": sum(record.status == RecordStatus.FAILED for record in records),
            "retries": sum(sum(record.retry_counts.values()) for record in records),
        }
