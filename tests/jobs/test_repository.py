"""Behavior tests for the durable file Job Repository."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.jobs import (
    CommandReceipt,
    CorruptJSONLError,
    FileJobRepository,
    ImmutableRecordError,
    JobCommand,
    JobStatus,
    LeaseConflictError,
    RevisionConflictError,
    SHARD_SCHEMA_VERSION,
    ShardState,
    compute_config_hash,
    hash_config_file,
)


@pytest.fixture
def repository(tmp_path: Path) -> FileJobRepository:
    return FileJobRepository(tmp_path / ".dataflux" / "jobs")


@pytest.fixture
def created_job(repository: FileJobRepository):
    request = repository.new_request(
        mode="process",
        config_path="config.yaml",
        config_sha256=compute_config_hash({"datasource": {"type": "sqlite"}}),
        options={"max_in_flight": 4},
        created_at=10.0,
    )
    repository.create_job(request)
    return request


def test_create_job_writes_modular_layout_and_immutable_request(
    repository: FileJobRepository, created_job
):
    directory = repository.job_dir(created_job.job_id)
    assert {path.name for path in directory.iterdir()} == {
        "request.json",
        "state.json",
        "events.jsonl",
        "shards",
        "prepared",
        "commands",
    }
    assert repository.get_request(created_job.job_id) == created_job
    assert repository.get_state(created_job.job_id).status == JobStatus.QUEUED
    assert repository.list_events(created_job.job_id)[0].type == "job_created"

    with pytest.raises(ImmutableRecordError):
        repository.create_job(created_job)


def test_request_metadata_rejects_secret_options(repository: FileJobRepository):
    with pytest.raises(ValueError, match="secret-like"):
        repository.new_request(
            mode="process",
            config_path="config.yaml",
            config_sha256=compute_config_hash({"a": 1}),
            options={"provider": {"api_key": "must-not-be-persisted"}},
        )


def test_state_updates_are_atomic_and_revision_checked(
    repository: FileJobRepository, created_job, monkeypatch
):
    import src.jobs.io as job_io

    replacements = []
    real_replace = job_io.os.replace

    def record_replace(source, target):
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(job_io.os, "replace", record_replace)
    initial = repository.get_state(created_job.job_id)
    updated = repository.save_state(
        replace(initial, status=JobStatus.RUNNING),
        expected_revision=initial.revision,
    )

    assert updated.revision == initial.revision + 1
    assert repository.get_state(created_job.job_id).status == JobStatus.RUNNING
    assert replacements
    assert replacements[-1][1] == repository.state_path(created_job.job_id)
    assert not list(repository.job_dir(created_job.job_id).glob("*.tmp"))

    with pytest.raises(RevisionConflictError):
        repository.save_state(updated, expected_revision=initial.revision)


def test_events_ignore_only_a_truncated_final_line(
    repository: FileJobRepository, created_job
):
    repository.append_event(created_job.job_id, "first")
    with repository.events_path(created_job.job_id).open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write('{"seq":999,"type":"truncated"')

    assert [event.type for event in repository.list_events(created_job.job_id)] == [
        "job_created",
        "first",
    ]
    recovered = repository.append_event(created_job.job_id, "after_recovery")
    assert recovered.seq == 3
    assert [event.type for event in repository.list_events(created_job.job_id)] == [
        "job_created",
        "first",
        "after_recovery",
    ]


def test_events_reject_corruption_before_final_line(
    repository: FileJobRepository, created_job
):
    path = repository.events_path(created_job.job_id)
    valid = repository.list_events(created_job.job_id)[0].to_dict()
    path.write_text(
        json.dumps(valid) + "\nnot-json\n" + json.dumps(valid) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CorruptJSONLError):
        repository.list_events(created_job.job_id)


def test_events_ignore_truncated_final_utf8_sequence(
    repository: FileJobRepository, created_job
):
    with repository.events_path(created_job.job_id).open("ab") as stream:
        stream.write(b'{"payload":"\xe4')
    assert [event.type for event in repository.list_events(created_job.job_id)] == [
        "job_created"
    ]


def test_lease_conflict_heartbeat_staleness_and_takeover(
    repository: FileJobRepository, created_job
):
    lease = repository.acquire_lease(created_job.job_id, "worker-a", now=100.0)
    assert lease.heartbeat_interval_seconds == 5.0
    assert lease.stale_after_seconds == 30.0
    assert lease.is_stale(129.9) is False
    assert lease.is_stale(130.0) is True

    with pytest.raises(LeaseConflictError):
        repository.acquire_lease(created_job.job_id, "worker-b", now=120.0)

    heartbeat = repository.heartbeat_lease(created_job.job_id, "worker-a", now=125.0)
    assert heartbeat.heartbeat_at == 125.0
    takeover = repository.acquire_lease(created_job.job_id, "worker-b", now=155.0)
    assert takeover.owner_id == "worker-b"


def test_recovery_uses_injected_resumability_decision(
    repository: FileJobRepository, created_job
):
    repository.transition(created_job.job_id, JobStatus.RUNNING)
    repository.acquire_lease(created_job.job_id, "dead-worker", now=100.0)

    recover = repository.evaluate_recovery(
        created_job.job_id,
        lambda state, request: request.config_sha256 == state.config_sha256,
        now=131.0,
    )
    assert recover.recover is True
    assert recover.target_status == JobStatus.QUEUED
    assert repository.apply_recovery_decision(recover).status == JobStatus.QUEUED

    repository.transition(created_job.job_id, JobStatus.INTERRUPTED)
    blocked = repository.evaluate_recovery(created_job.job_id, False, now=200.0)
    assert blocked.recover is False
    assert blocked.target_status == JobStatus.BLOCKED


def test_shards_commands_and_command_receipts_round_trip(
    repository: FileJobRepository, created_job
):
    shard = ShardState(
        shard_id="0001",
        job_id=created_job.job_id,
        status="pending",
        cursor={"offset": 20},
        counts={"discovered": 10},
    )
    repository.save_shard(shard)
    assert repository.get_shard(created_job.job_id, "0001") == shard

    command = JobCommand("cancel-1", created_job.job_id, "cancel")
    repository.add_command(command)
    assert repository.list_commands(created_job.job_id) == [command]
    receipt = CommandReceipt(
        "cancel-1",
        created_job.job_id,
        True,
        resulting_status=JobStatus.CANCELLING,
    )
    repository.save_command_receipt(receipt)
    assert repository.get_command_receipt(created_job.job_id, "cancel-1") == receipt


@pytest.mark.parametrize("schema_value", [None, 1, 3])
def test_old_or_missing_shard_schema_is_rejected(
    repository: FileJobRepository,
    created_job,
    schema_value,
):
    payload = {
        "shard_id": "legacy",
        "job_id": created_job.job_id,
        "status": "active",
        "updated_at": 1.0,
        "cursor": None,
        "counts": {},
        "records": [],
    }
    if schema_value is not None:
        payload["schema_version"] = schema_value
    path = repository.shards_dir(created_job.job_id) / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="shard schema_version"):
        repository.get_shard(created_job.job_id, "legacy")


def test_old_record_checkpoint_schema_is_rejected(
    repository: FileJobRepository,
    created_job,
):
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "shard_id": "legacy-record",
        "job_id": created_job.job_id,
        "status": "active",
        "updated_at": 1.0,
        "cursor": None,
        "counts": {},
        "records": [
            {
                "record_id": "a",
                "status": "pending",
                "updated_at": 1.0,
            }
        ],
    }
    path = repository.shards_dir(created_job.job_id) / "legacy-record.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="record checkpoint schema_version"):
        repository.get_shard(created_job.job_id, "legacy-record")


def test_prune_is_dry_run_until_explicit_confirmation(
    repository: FileJobRepository, created_job
):
    repository.transition(created_job.job_id, JobStatus.COMPLETED)
    preview = repository.preview_prune(older_than=float("inf"))
    assert [candidate.job_id for candidate in preview.candidates] == [
        created_job.job_id
    ]

    dry_run = repository.prune(preview)
    assert dry_run.confirmed is False
    assert repository.job_dir(created_job.job_id).exists()

    result = repository.prune(preview, confirm=True)
    assert result.deleted_job_ids == (created_job.job_id,)
    assert not repository.job_dir(created_job.job_id).exists()


def test_config_hash_is_deterministic_and_file_hash_uses_exact_bytes(tmp_path: Path):
    assert compute_config_hash({"b": 2, "a": 1}) == compute_config_hash(
        {"a": 1, "b": 2}
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(b"a: 1\n")
    assert hash_config_file(config_path) != compute_config_hash({"a": 1})
