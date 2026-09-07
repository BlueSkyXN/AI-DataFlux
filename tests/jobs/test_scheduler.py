"""Resource admission and cooperative concurrency tests."""

from types import SimpleNamespace

from src.jobs import (
    ResourceProbe,
    ResourceScheduler,
    ResourceSnapshot,
    default_max_active_jobs,
)


def snapshot(*, pressure: bool, status: str = "normal") -> ResourceSnapshot:
    return ResourceSnapshot(
        sampled_at=1.0,
        status=status,
        pressure=pressure,
        reasons=("cpu",) if pressure else (),
    )


def test_scheduler_admits_in_created_at_fifo_order():
    scheduler = ResourceScheduler(max_active_jobs=1)
    scheduler.enqueue("later", created_at=20.0, max_in_flight=4)
    scheduler.enqueue("first", created_at=10.0, max_in_flight=2)
    scheduler.enqueue("last", created_at=30.0, max_in_flight=1)

    decision = scheduler.tick(snapshot(pressure=False))
    assert decision.admitted_job_ids == ("first",)
    assert decision.targets == {"first": 2}

    scheduler.release("first")
    assert scheduler.tick(snapshot(pressure=False)).admitted_job_ids == ("later",)


def test_pressure_blocks_admission_and_cooperatively_reduces_targets():
    scheduler = ResourceScheduler(max_active_jobs=2)
    scheduler.enqueue("active", created_at=1.0, max_in_flight=8)
    scheduler.tick(snapshot(pressure=False))
    scheduler.enqueue("queued", created_at=2.0, max_in_flight=3)

    pressured = scheduler.tick(snapshot(pressure=True))
    assert pressured.admitted_job_ids == ()
    assert pressured.targets["active"] == 4
    scheduler.tick(snapshot(pressure=True))
    scheduler.tick(snapshot(pressure=True))
    assert scheduler.target_concurrency("active") == 1
    assert scheduler.active_job_ids == ("active",)  # no preemption

    recovered = scheduler.tick(snapshot(pressure=False))
    assert recovered.targets["active"] == 2
    assert recovered.admitted_job_ids == ("queued",)


def test_psutil_missing_degrades_to_hard_limit_only():
    probe = ResourceProbe(psutil_module=None)
    resource = probe.sample(force=True, now=10.0)
    assert resource.status == "degraded"
    assert resource.pressure is False
    assert resource.reasons == ("psutil_unavailable",)

    scheduler = ResourceScheduler(max_active_jobs=1, probe=probe)
    scheduler.enqueue("job-a", created_at=1.0, max_in_flight=2)
    scheduler.enqueue("job-b", created_at=2.0, max_in_flight=2)
    assert scheduler.tick(now=10.0).admitted_job_ids == ("job-a",)
    assert scheduler.tick(now=11.0).admitted_job_ids == ()


def test_resource_thresholds_detect_cpu_memory_and_minimum_free_space():
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 90.0,
        virtual_memory=lambda: SimpleNamespace(
            percent=81.0,
            available=256 * 1024 * 1024,
        ),
    )
    resource = ResourceProbe(psutil_module=fake_psutil).sample(force=True)
    assert resource.pressure is True
    assert resource.reasons == ("cpu", "memory_percent", "memory_available")


def test_default_active_limit_matches_bounded_cpu_formula(monkeypatch):
    monkeypatch.setattr("src.jobs.scheduler.os.cpu_count", lambda: 32)
    assert default_max_active_jobs() == 4
    monkeypatch.setattr("src.jobs.scheduler.os.cpu_count", lambda: 2)
    assert default_max_active_jobs() == 1


def test_fifo_queue_drains_after_pressure_without_exceeding_active_limit():
    scheduler = ResourceScheduler(max_active_jobs=3)
    for index in range(25):
        scheduler.enqueue(str(index), created_at=float(index), max_in_flight=4)
    assert scheduler.tick(snapshot(pressure=True)).admitted_job_ids == ()
    admitted = []
    while scheduler.queued_job_ids:
        decision = scheduler.tick(snapshot(pressure=False))
        admitted.extend(decision.admitted_job_ids)
        assert len(scheduler.active_job_ids) <= 3
        assert all(1 <= target <= 4 for target in decision.targets.values())
        for job_id in tuple(scheduler.active_job_ids):
            scheduler.release(job_id)
    assert admitted == [str(index) for index in range(25)]
