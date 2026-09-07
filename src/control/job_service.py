"""Control-layer coordinator for durable background Jobs."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from src.config import (
    compile_job_config,
    execution_config_hash,
    load_config,
    resolve_workspace_path,
    resolve_workspace_roots,
)
from src.core.job_runner import run_processing_job
from src.config.security import redact_sensitive_text
from src.data import declared_adapter_capabilities
from src.jobs import (
    CommandReceipt,
    FileJobRepository,
    JobCommand,
    JobResourceState,
    JobState,
    JobStatus,
    JobWorker,
    ResourcePolicy,
    ResourceProbe,
    ResourceScheduler,
    RevisionConflictError,
)

MAX_JOB_MAX_IN_FLIGHT = 10_000
_ALLOWED_JOB_OPTIONS = frozenset({"max_in_flight"})


def _validated_max_in_flight(options: dict[str, Any], default: int) -> int:
    unknown = sorted(set(options) - _ALLOWED_JOB_OPTIONS)
    if unknown:
        raise ValueError(f"Job options 包含未知键: {unknown}")
    value = options.get("max_in_flight", default)
    if type(value) is not int or not 1 <= value <= MAX_JOB_MAX_IN_FLIGHT:
        raise ValueError(
            f"Job options.max_in_flight 必须是 1 到 {MAX_JOB_MAX_IN_FLIGHT} 的整数"
        )
    return value


class JobService:
    """Own queue admission, cooperative cancellation, and API-facing reads."""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.root_config = load_config(self.config_path)
        self.config = compile_job_config(self.root_config, self.config_path)
        roots = resolve_workspace_roots(self.root_config, self.config_path)
        self.roots = roots
        state_dir = Path(self.config["workspace"]["state_dir"]).expanduser()
        if not state_dir.is_absolute():
            state_dir = Path(self.config_path).parent / state_dir
        state_dir = state_dir.resolve(strict=False)
        if not any(
            _is_within(state_dir, allowed_root) for allowed_root in roots.values()
        ):
            raise ValueError("workspace.state_dir 必须位于 workspace.roots 内")
        self.repository = FileJobRepository(state_dir)

        scheduler_config = self.config.get("scheduler", {})
        max_active = scheduler_config.get("max_active_jobs", "auto")
        policy = ResourcePolicy(
            cpu_percent_limit=float(scheduler_config.get("cpu_high_watermark", 85)),
            memory_percent_limit=float(
                scheduler_config.get("memory_high_watermark", 80)
            ),
            min_available_memory_bytes=int(
                scheduler_config.get("min_free_memory_mb", 512)
            )
            * 1024
            * 1024,
            sample_interval_seconds=float(
                scheduler_config.get("sample_interval_seconds", 2)
            ),
        )
        self.scheduler = ResourceScheduler(
            max_active_jobs=None if max_active == "auto" else int(max_active),
            probe=ResourceProbe(policy),
        )
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._stop_event = asyncio.Event()
        self._started = asyncio.Event()
        self._shutdown_task: asyncio.Task | None = None
        self._last_snapshot = self.scheduler.probe.sample()

    def resolve_path(self, root_id: str, relative_path: str) -> Path:
        return resolve_workspace_path(
            self.root_config, root_id, relative_path, self.config_path
        )

    def submit(
        self,
        *,
        root_id: str,
        relative_path: str,
        options: dict[str, Any] | None = None,
    ) -> JobState:
        config_path = self.resolve_path(root_id, relative_path)
        if config_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("Job config 必须是 YAML 文件")
        root_config = load_config(config_path)
        merged = compile_job_config(root_config, config_path)
        configured_max_in_flight = int(
            merged.get("datasource", {})
            .get("concurrency", {})
            .get("max_in_flight", 100)
        )
        submitted_options = dict(options or {})
        max_in_flight = _validated_max_in_flight(
            submitted_options,
            configured_max_in_flight,
        )
        public_options = {"max_in_flight": max_in_flight}
        request = self.repository.new_request(
            mode="background",
            config_path=str(config_path),
            config_sha256=execution_config_hash(root_config, config_path),
            options=public_options,
        )
        state = self.repository.create_job(request)
        self.scheduler.enqueue(
            request.job_id,
            created_at=request.created_at,
            max_in_flight=max_in_flight,
        )
        return state

    def list_states(self) -> list[JobState]:
        states = [
            self.repository.get_state(job_id)
            for job_id in self.repository.list_job_ids()
        ]
        return sorted(states, key=lambda state: state.created_at, reverse=True)

    def resource_status(self) -> dict[str, Any]:
        snapshot = self._last_snapshot
        active = list(self.scheduler.active_job_ids)
        queued = list(self.scheduler.queued_job_ids)
        if not self._started.is_set():
            snapshot = self.scheduler.probe.sample()
            states = self.list_states()
            active = [
                state.job_id
                for state in states
                if state.status in {JobStatus.RUNNING, JobStatus.CANCELLING}
            ]
            queued = [
                state.job_id for state in states if state.status == JobStatus.QUEUED
            ]
        return {
            **asdict(snapshot),
            "resource_control": snapshot.status,
            "max_active_jobs": self.scheduler.max_active_jobs,
            "active_jobs": active,
            "queued_jobs": queued,
        }

    def list_events(
        self, job_id: str, *, after_seq: int = 0, limit: int = 100
    ) -> tuple[list[dict[str, Any]], int]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit 必须在 1 到 1000 之间")
        events = [
            event
            for event in self.repository.list_events(job_id)
            if event.seq > after_seq
        ][:limit]
        next_seq = events[-1].seq if events else after_seq
        return [event.to_dict() for event in events], next_seq

    def cancel(self, job_id: str) -> JobState:
        command = JobCommand(command_id=str(uuid4()), job_id=job_id, type="cancel")
        updated, receipt = self._apply_command(command)
        if not receipt.accepted:
            raise ValueError(f"Job 当前状态不能取消: {updated.status.value}")
        return updated

    def resume(self, job_id: str) -> JobState:
        state = self.repository.get_state(job_id)
        if state.status not in {
            JobStatus.BLOCKED,
            JobStatus.INTERRUPTED,
            JobStatus.FAILED,
        }:
            raise ValueError(f"Job 当前状态不能恢复: {state.status.value}")
        request = self.repository.get_request(job_id)
        _validated_max_in_flight(dict(request.options), 1)
        if not Path(request.config_path).is_file():
            raise ValueError("Job config 不存在")
        current_config = load_config(request.config_path)
        current_hash = execution_config_hash(current_config, request.config_path)
        command = JobCommand(
            command_id=str(uuid4()),
            job_id=job_id,
            type="resume",
            payload={"accepted_config_sha256": current_hash},
        )
        updated, receipt = self._apply_command(command)
        if not receipt.accepted:
            raise ValueError(receipt.message)
        return updated

    def _apply_command(self, command: JobCommand) -> tuple[JobState, CommandReceipt]:
        state = self.repository.get_state(command.job_id)
        target = None
        message = "unsupported_command"
        if command.type == "cancel":
            if state.status == JobStatus.QUEUED:
                target = JobStatus.CANCELLED
            elif state.status in {JobStatus.RUNNING, JobStatus.CANCELLING}:
                target = JobStatus.CANCELLING
            message = target.value if target else "job_not_cancellable"
        elif command.type == "resume":
            message = "job_not_resumable"
            if state.status in {
                JobStatus.FAILED,
                JobStatus.BLOCKED,
                JobStatus.INTERRUPTED,
            }:
                request = self.repository.get_request(command.job_id)
                config = load_config(request.config_path)
                if execution_config_hash(
                    config, request.config_path
                ) == command.payload.get("accepted_config_sha256"):
                    target, message = JobStatus.QUEUED, "queued"
                else:
                    message = "resume_config_changed"
        updated, receipt = self.repository.apply_command(
            command, target=target, expected_revision=state.revision, message=message
        )
        if receipt.accepted:
            if updated.status == JobStatus.QUEUED:
                request = self.repository.get_request(command.job_id)
                self.scheduler.enqueue(
                    command.job_id,
                    created_at=updated.created_at,
                    max_in_flight=_validated_max_in_flight(dict(request.options), 1),
                )
            else:
                self.scheduler.remove_queued(command.job_id)
                if updated.status == JobStatus.CANCELLING:
                    self._cancel_events.setdefault(
                        command.job_id, asyncio.Event()
                    ).set()
        return updated, receipt

    async def recover(self) -> None:
        for state in self.list_states():
            request = self.repository.get_request(state.job_id)
            if state.status != JobStatus.QUEUED:
                self.scheduler.remove_queued(state.job_id)
            if state.job_id in self._tasks:
                continue
            if state.is_terminal or state.status == JobStatus.BLOCKED:
                continue
            try:
                max_in_flight = _validated_max_in_flight(dict(request.options), 1)
            except ValueError:
                self.repository.transition(
                    state.job_id,
                    JobStatus.BLOCKED,
                    last_error="invalid_job_options",
                )
                continue
            if state.status == JobStatus.QUEUED:
                self.scheduler.enqueue(
                    state.job_id,
                    created_at=state.created_at,
                    max_in_flight=max_in_flight,
                )
                continue
            if state.status not in {
                JobStatus.RUNNING,
                JobStatus.CANCELLING,
                JobStatus.INTERRUPTED,
            }:
                continue
            lease = self.repository.get_lease(state.job_id)
            if lease is not None and not lease.is_stale():
                continue
            if state.status == JobStatus.CANCELLING:
                self.repository.transition(
                    state.job_id, JobStatus.CANCELLED, expected_revision=state.revision
                )
                continue
            resumable = False
            if Path(request.config_path).is_file():
                try:
                    root_config = load_config(request.config_path)
                    job_config = root_config.job
                    if job_config is None:
                        raise ValueError("job section is required")
                    datasource_type = job_config.datasource.type
                    capabilities = declared_adapter_capabilities(datasource_type)
                    resumable = (
                        capabilities.resumable
                        and capabilities.idempotent_write
                        and execution_config_hash(root_config, request.config_path)
                        == self.repository.effective_config_hash(state.job_id)
                    )
                except Exception:
                    logging.warning(
                        "Job %s 自动恢复能力检查失败", state.job_id, exc_info=True
                    )
            status = JobStatus.QUEUED if resumable else JobStatus.BLOCKED
            self.repository.transition(
                state.job_id,
                status,
                last_error=None if resumable else "automatic_recovery_blocked",
            )
            if resumable:
                self.scheduler.enqueue(
                    state.job_id,
                    created_at=state.created_at,
                    max_in_flight=max_in_flight,
                )

    async def start(self) -> asyncio.Task:
        task = asyncio.create_task(self.run_loop())
        ready = asyncio.create_task(self._started.wait())
        try:
            done, _ = await asyncio.wait(
                {task, ready}, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                await task
            return task
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)

    async def run_loop(self) -> None:
        with self.repository.supervisor_lock():
            try:
                await self.recover()
                self._started.set()
                await self._run_owned_loop()
            finally:
                await self.stop()

    async def _run_owned_loop(self) -> None:
        interval = self.scheduler.probe.policy.sample_interval_seconds
        while not self._stop_event.is_set():
            await self.recover()
            self._consume_commands()
            decision = self.scheduler.tick()
            self._last_snapshot = decision.snapshot
            for job_id, target in decision.targets.items():
                try:

                    def update_resource_state(
                        state: JobState,
                        target_value: int = target,
                    ) -> JobState:
                        return replace(
                            state,
                            resource=JobResourceState(
                                effective_max_in_flight=target_value,
                                pressure=decision.snapshot.pressure,
                                control_status=decision.snapshot.status,
                            ),
                        )

                    self.repository.update_state(
                        job_id,
                        update_resource_state,
                    )
                except Exception:
                    logging.debug("Job resource state update skipped", exc_info=True)
            for job_id in decision.admitted_job_ids:
                cancel_event = self._cancel_events.setdefault(job_id, asyncio.Event())
                worker = JobWorker(self.repository, run_processing_job)

                def target_provider(job_id_value: str = job_id) -> int:
                    return self.scheduler.target_concurrency(job_id_value)

                task = asyncio.create_task(
                    worker.run(
                        job_id,
                        target_concurrency_provider=target_provider,
                        cancel_event=cancel_event,
                    )
                )
                self._tasks[job_id] = task

                def release_job(
                    completed: asyncio.Task,
                    job_id_value: str = job_id,
                ) -> None:
                    self._release_job(job_id_value, completed)

                task.add_done_callback(release_job)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    def _discover_queued_jobs(self) -> None:
        """Discover Jobs created by a separate Control process."""

        for state in self.list_states():
            if state.status != JobStatus.QUEUED:
                self.scheduler.remove_queued(state.job_id)
                continue
            request = self.repository.get_request(state.job_id)
            try:
                max_in_flight = _validated_max_in_flight(dict(request.options), 1)
            except ValueError:
                self.scheduler.remove_queued(state.job_id)
                self.repository.transition(
                    state.job_id,
                    JobStatus.BLOCKED,
                    last_error="invalid_job_options",
                )
                continue
            self.scheduler.enqueue(
                state.job_id,
                created_at=state.created_at,
                max_in_flight=max_in_flight,
            )

    def _consume_commands(self) -> None:
        """Apply durable commands so standalone Control and Worker cooperate."""

        for state in self.list_states():
            if state.status == JobStatus.CANCELLING:
                self._cancel_events.setdefault(state.job_id, asyncio.Event()).set()
            for command in self.repository.list_commands(state.job_id):
                if (
                    self.repository.get_command_receipt(
                        state.job_id, command.command_id
                    )
                    is not None
                ):
                    continue
                try:
                    self._apply_command(command)
                except RevisionConflictError:
                    continue  # 外部 Control 更新了 revision，下轮重新检查。

    def _release_job(self, job_id: str, task: asyncio.Task) -> None:
        if not task.cancelled():
            try:
                error = task.exception()
            except Exception:
                logging.exception("Background Job %s failed", job_id)
            else:
                if error is not None:
                    logging.error(
                        "Background Job %s failed: %s",
                        job_id,
                        redact_sensitive_text(error),
                    )
        self.scheduler.release(job_id)
        self._tasks.pop(job_id, None)
        self._cancel_events.pop(job_id, None)

    async def stop(self) -> None:
        self._stop_event.set()
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._stop_jobs())
        await asyncio.shield(self._shutdown_task)

    async def _stop_jobs(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stream_events(
        self, job_id: str, *, after_seq: int = 0
    ) -> AsyncIterator[str]:
        cursor = after_seq
        while True:
            events = [
                event
                for event in self.repository.list_events(job_id)
                if event.seq > cursor
            ]
            for event in events:
                cursor = event.seq
                payload = json.dumps(event.to_dict(), ensure_ascii=False)
                yield f"id: {event.seq}\ndata: {payload}\n\n"
            state = self.repository.get_state(job_id)
            if (state.is_terminal or state.status == JobStatus.BLOCKED) and not events:
                break
            await asyncio.sleep(0.5)


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path == root or root in path.parents
    except RuntimeError:
        return False
