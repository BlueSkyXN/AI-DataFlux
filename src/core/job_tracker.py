"""Durable per-record checkpoints used by background Job execution."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, Iterable, Mapping

from src.jobs import (
    FileJobRepository,
    JobCounts,
    RecordCheckpoint,
    RecordStatus,
    ShardState,
)


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
        self._records: dict[str, RecordCheckpoint] = {}
        self._record_shards: dict[str, str] = {}
        self._shards: dict[str, ShardState] = {}
        self._next_scan_shard = 1

        for shard in repository.list_shards(job_id):
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

    def should_process_scanned(self, record_id: Any) -> bool:
        """Avoid repeating AI work for already durable or replayable outputs."""

        checkpoint = self.get(record_id)
        if checkpoint is None:
            return True
        if checkpoint.status == RecordStatus.PERSISTED:
            return False
        return not (
            checkpoint.status in {RecordStatus.AI_COMPLETE, RecordStatus.FAILED}
            and checkpoint.result is not None
        )

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
            result=current.result if current else None,
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
            result=current.result if current else None,
            error={"code": error_code} if error_code else None,
            retry_counts=retry_counts,
            event_type="record_retry_scheduled",
        )

    def mark_ai_complete(
        self,
        record_id: Any,
        result: Mapping[str, Any],
    ) -> RecordCheckpoint:
        current = self.get(record_id)
        return self._transition(
            record_id,
            RecordStatus.AI_COMPLETE,
            attempt=current.attempt if current else 1,
            input_data=current.input_data if current else None,
            result=dict(result),
            event_type="record_ai_complete",
        )

    def mark_persisted(self, record_id: Any) -> RecordCheckpoint:
        current = self.get(record_id)
        return self._transition(
            record_id,
            RecordStatus.PERSISTED,
            attempt=current.attempt if current else 0,
            input_data=current.input_data if current else None,
            result=current.result if current else None,
            event_type="record_persisted",
        )

    def mark_failed(
        self,
        record_id: Any,
        *,
        error_code: str,
        result: Mapping[str, Any] | None = None,
    ) -> RecordCheckpoint:
        current = self.get(record_id)
        return self._transition(
            record_id,
            RecordStatus.FAILED,
            attempt=current.attempt if current else 0,
            input_data=current.input_data if current else None,
            result=(
                dict(result)
                if result is not None
                else (current.result if current else None)
            ),
            error={"code": error_code},
            event_type="record_failed",
        )

    def replayable_results(self) -> dict[Any, dict[str, Any]]:
        return {
            checkpoint.record_id: dict(checkpoint.result)
            for checkpoint in self._records.values()
            if checkpoint.status in {RecordStatus.AI_COMPLETE, RecordStatus.FAILED}
            and checkpoint.result is not None
        }

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
            ai_complete=sum(
                record.status in {RecordStatus.AI_COMPLETE, RecordStatus.PERSISTED}
                for record in records
            ),
            persisted=sum(
                record.status == RecordStatus.PERSISTED for record in records
            ),
            failed=sum(record.status == RecordStatus.FAILED for record in records),
            cancelled=(
                current_state.counts.cancelled if cancelled is None else cancelled
            ),
            retries=retries,
        )

    def _transition(
        self,
        record_id: Any,
        status: RecordStatus,
        *,
        attempt: int,
        input_data: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        retry_counts: Mapping[str, int] | None = None,
        event_type: str,
    ) -> RecordCheckpoint:
        current = self.get(record_id)
        checkpoint = RecordCheckpoint(
            record_id=record_id,
            status=status,
            attempt=attempt,
            retry_counts=(
                dict(retry_counts)
                if retry_counts is not None
                else (dict(current.retry_counts) if current else {})
            ),
            input_data=input_data,
            result=result,
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
        terminal = {RecordStatus.PERSISTED, RecordStatus.FAILED}
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
        self.repository.save_shard(shard)
        self._shards[shard_id] = shard

    def _publish_counts(self) -> JobCounts:
        counts = self.counts()
        self.repository.update_state(
            self.job_id,
            lambda state: replace(state, counts=counts),
        )
        return counts

    @staticmethod
    def _counts_for(records: tuple[RecordCheckpoint, ...]) -> dict[str, int]:
        return {
            "discovered": len(records),
            "pending": sum(record.status == RecordStatus.PENDING for record in records),
            "in_flight": sum(
                record.status == RecordStatus.IN_FLIGHT for record in records
            ),
            "ai_complete": sum(
                record.status in {RecordStatus.AI_COMPLETE, RecordStatus.PERSISTED}
                for record in records
            ),
            "persisted": sum(
                record.status == RecordStatus.PERSISTED for record in records
            ),
            "failed": sum(record.status == RecordStatus.FAILED for record in records),
            "retries": sum(sum(record.retry_counts.values()) for record in records),
        }
