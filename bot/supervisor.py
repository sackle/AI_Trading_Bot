"""Bounded worker supervision for the US and NSE trading loops."""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable

import config
from bot.runtime_coordinator import RuntimeCoordinator


WorkerTarget = Callable[[threading.Event], None]


@dataclass
class WorkerSpec:
    name: str
    target: WorkerTarget


@dataclass
class _WorkerRecord:
    spec: WorkerSpec
    thread: threading.Thread | None = None
    restart_count: int = 0
    last_error: str = ""
    next_start_at: float = 0.0
    exited: threading.Event = field(default_factory=threading.Event)


class RuntimeSupervisor:
    """Restart unexpected worker exits with bounded backoff and a safe pause."""

    def __init__(
        self,
        coordinator: RuntimeCoordinator,
        specs: list[WorkerSpec],
        *,
        stop_event: threading.Event | None = None,
        max_restarts: int | None = None,
        backoff_seconds: float | None = None,
        poll_seconds: float | None = None,
    ):
        if not specs:
            raise ValueError("RuntimeSupervisor requires at least one worker.")
        self.coordinator = coordinator
        self.records = {spec.name: _WorkerRecord(spec) for spec in specs}
        self.stop_event = stop_event or threading.Event()
        self.max_restarts = int(max_restarts or config.WORKER_MAX_RESTARTS)
        self.backoff_seconds = float(backoff_seconds or config.WORKER_RESTART_BACKOFF_SECONDS)
        self.poll_seconds = float(poll_seconds or config.WORKER_SUPERVISOR_POLL_SECONDS)

    @property
    def worker_names(self) -> list[str]:
        return list(self.records)

    def _spawn(self, record: _WorkerRecord) -> None:
        record.exited.clear()
        self.coordinator.worker_started(record.spec.name, restart_count=record.restart_count)

        def run() -> None:
            try:
                record.spec.target(self.stop_event)
            except BaseException as exc:  # preserve crash detail for the supervisor/UI
                record.last_error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            finally:
                record.exited.set()

        record.thread = threading.Thread(
            target=run,
            name=f"trading-worker-{record.spec.name}",
            daemon=False,
        )
        record.thread.start()

    def run(self) -> None:
        for record in self.records.values():
            self._spawn(record)

        try:
            while not self.stop_event.wait(self.poll_seconds):
                all_healthy = True
                coordinated = self.coordinator.snapshot()
                for record in self.records.values():
                    thread = record.thread
                    if thread is not None and thread.is_alive():
                        worker = coordinated.get("workers", {}).get(record.spec.name, {})
                        age = worker.get("heartbeat_age_seconds")
                        if age is None or age > self.coordinator.heartbeat_ttl_seconds:
                            all_healthy = False
                            self.coordinator.worker_stale(
                                record.spec.name,
                                f"No heartbeat within {self.coordinator.heartbeat_ttl_seconds}s.",
                            )
                        elif not worker.get("healthy"):
                            all_healthy = False
                        continue

                    all_healthy = False
                    if not record.exited.is_set():
                        continue
                    if not record.last_error:
                        record.last_error = "worker returned unexpectedly"

                    record.restart_count += 1
                    self.coordinator.worker_failed(
                        record.spec.name,
                        record.last_error,
                        restart_count=record.restart_count,
                    )
                    if record.restart_count > self.max_restarts:
                        self.stop_event.set()
                        break

                    delay = min(
                        self.backoff_seconds * (2 ** (record.restart_count - 1)), 60.0
                    )
                    if self.stop_event.wait(delay):
                        break
                    record.last_error = ""
                    self._spawn(record)

                if all_healthy and all(
                    self.coordinator.snapshot().get("workers", {}).get(name, {}).get("healthy")
                    for name in self.worker_names
                ):
                    self.coordinator.clear_system_pause_if_healthy(self.worker_names)
        finally:
            self.stop_event.set()
            for record in self.records.values():
                thread = record.thread
                if thread is not None:
                    thread.join(timeout=config.WORKER_SHUTDOWN_TIMEOUT_SECONDS)
                    # Keep the process lock held until every order-producing
                    # worker has actually exited. A worker blocked in a
                    # bounded network request may need longer than the normal
                    # shutdown grace period, but must never overlap a new run.
                    while thread.is_alive():
                        thread.join(timeout=1.0)
                self.coordinator.worker_stopped(record.spec.name)
            self.coordinator.stop("Supervisor stopped.")
