"""FIFO admission and cooperative resource control for Jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import os
import time
from typing import Any

try:
    import psutil as _psutil
except ImportError:  # pragma: no cover - exercised through injected ``None``
    _psutil = None


MEBIBYTE = 1024 * 1024


@dataclass(frozen=True)
class ResourcePolicy:
    """Resource thresholds and sampling policy."""

    cpu_percent_limit: float = 85.0
    memory_percent_limit: float = 80.0
    min_available_memory_bytes: int = 512 * MEBIBYTE
    sample_interval_seconds: float = 2.0


@dataclass(frozen=True)
class ResourceSnapshot:
    """One host resource sample used for scheduling decisions."""

    sampled_at: float
    status: str
    pressure: bool
    cpu_percent: float | None = None
    memory_percent: float | None = None
    available_memory_bytes: int | None = None
    reasons: tuple[str, ...] = ()


class ResourceProbe:
    """Optional-psutil sampler with a two-second default cache."""

    def __init__(
        self,
        policy: ResourcePolicy | None = None,
        *,
        psutil_module: Any = _psutil,
    ):
        self.policy = policy or ResourcePolicy()
        self.psutil = psutil_module
        self._cached: ResourceSnapshot | None = None

    def sample(
        self, *, force: bool = False, now: float | None = None
    ) -> ResourceSnapshot:
        current = time.time() if now is None else now
        if (
            not force
            and self._cached is not None
            and current - self._cached.sampled_at < self.policy.sample_interval_seconds
        ):
            return self._cached

        if self.psutil is None:
            snapshot = ResourceSnapshot(
                sampled_at=current,
                status="degraded",
                pressure=False,
                reasons=("psutil_unavailable",),
            )
            self._cached = snapshot
            return snapshot

        try:
            cpu_percent = float(self.psutil.cpu_percent(interval=None))
            memory = self.psutil.virtual_memory()
            memory_percent = float(memory.percent)
            available = int(memory.available)
        except Exception:
            snapshot = ResourceSnapshot(
                sampled_at=current,
                status="degraded",
                pressure=False,
                reasons=("psutil_probe_failed",),
            )
            self._cached = snapshot
            return snapshot

        reasons: list[str] = []
        if cpu_percent >= self.policy.cpu_percent_limit:
            reasons.append("cpu")
        if memory_percent >= self.policy.memory_percent_limit:
            reasons.append("memory_percent")
        if available <= self.policy.min_available_memory_bytes:
            reasons.append("memory_available")
        snapshot = ResourceSnapshot(
            sampled_at=current,
            status="normal",
            pressure=bool(reasons),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            available_memory_bytes=available,
            reasons=tuple(reasons),
        )
        self._cached = snapshot
        return snapshot


@dataclass(order=True, frozen=True)
class _QueuedJob:
    created_at: float
    sequence: int
    job_id: str = field(compare=False)
    max_in_flight: int = field(compare=False)


@dataclass
class ActiveJob:
    """Cooperative target for one admitted job."""

    job_id: str
    desired_max_in_flight: int
    target_max_in_flight: int
    admitted_at: float


@dataclass(frozen=True)
class SchedulerDecision:
    """Result of one non-blocking scheduler tick."""

    snapshot: ResourceSnapshot
    admitted_job_ids: tuple[str, ...]
    targets: dict[str, int]


def default_max_active_jobs() -> int:
    """Return the bounded automatic admission limit."""

    return min(4, max(1, (os.cpu_count() or 2) // 2))


class ResourceScheduler:
    """FIFO scheduler with admission control and no preemption."""

    def __init__(
        self,
        *,
        max_active_jobs: int | None = None,
        probe: ResourceProbe | None = None,
    ):
        limit = (
            default_max_active_jobs() if max_active_jobs is None else max_active_jobs
        )
        if limit < 1:
            raise ValueError("max_active_jobs must be positive")
        self.max_active_jobs = limit
        self.probe = probe or ResourceProbe()
        self._queue: list[_QueuedJob] = []
        self._queued_ids: set[str] = set()
        self._active: dict[str, ActiveJob] = {}
        self._sequence = 0

    def enqueue(self, job_id: str, *, created_at: float, max_in_flight: int) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        if job_id in self._queued_ids or job_id in self._active:
            return
        self._sequence += 1
        heapq.heappush(
            self._queue,
            _QueuedJob(created_at, self._sequence, job_id, max_in_flight),
        )
        self._queued_ids.add(job_id)

    def remove_queued(self, job_id: str) -> bool:
        if job_id not in self._queued_ids:
            return False
        self._queued_ids.remove(job_id)
        self._queue = [item for item in self._queue if item.job_id != job_id]
        heapq.heapify(self._queue)
        return True

    def release(self, job_id: str) -> bool:
        return self._active.pop(job_id, None) is not None

    def tick(
        self, snapshot: ResourceSnapshot | None = None, *, now: float | None = None
    ) -> SchedulerDecision:
        """Sample, adjust cooperative targets, and admit FIFO jobs."""

        current = time.time() if now is None else now
        actual_snapshot = snapshot or self.probe.sample(now=current)

        if actual_snapshot.pressure:
            for active in self._active.values():
                active.target_max_in_flight = max(1, active.target_max_in_flight // 2)
        else:
            for active in self._active.values():
                if active.target_max_in_flight < active.desired_max_in_flight:
                    active.target_max_in_flight += 1

        admitted: list[str] = []
        if not actual_snapshot.pressure:
            while self._queue and len(self._active) < self.max_active_jobs:
                queued = heapq.heappop(self._queue)
                self._queued_ids.remove(queued.job_id)
                self._active[queued.job_id] = ActiveJob(
                    job_id=queued.job_id,
                    desired_max_in_flight=queued.max_in_flight,
                    target_max_in_flight=queued.max_in_flight,
                    admitted_at=current,
                )
                admitted.append(queued.job_id)

        return SchedulerDecision(
            snapshot=actual_snapshot,
            admitted_job_ids=tuple(admitted),
            targets={
                job_id: active.target_max_in_flight
                for job_id, active in self._active.items()
            },
        )

    def target_concurrency(self, job_id: str) -> int:
        active = self._active.get(job_id)
        if active is None:
            return 1
        return active.target_max_in_flight

    @property
    def active_job_ids(self) -> tuple[str, ...]:
        return tuple(self._active)

    @property
    def queued_job_ids(self) -> tuple[str, ...]:
        return tuple(item.job_id for item in sorted(self._queue))
