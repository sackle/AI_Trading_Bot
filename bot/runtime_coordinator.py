"""Durable coordination state for the multi-market trading runtime.

This module is intentionally small and broker-neutral.  The US Alpaca
account and NSE paper account are denominated in different currencies, so it
would be misleading and unsafe to combine their cash or P&L.  Instead this
store synchronizes the operational concerns that *can* be shared: lifecycle,
heartbeats, crash details, restart counts, and an operator/system entry pause.

Writes use replace-on-complete semantics, so the dashboard never consumes a
partially written JSON document while a worker updates its heartbeat.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _now().isoformat()


class RuntimeCoordinator:
    """Coordinate workers without conflating their broker accounts."""

    def __init__(self, path: str | Path | None = None, heartbeat_ttl_seconds: int | None = None):
        config.ensure_runtime_directories()
        self.path = Path(path or config.runtime_path(config.COORDINATOR_STATE_PATH))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_ttl_seconds = int(
            heartbeat_ttl_seconds or config.WORKER_HEARTBEAT_TTL_SECONDS
        )
        self._lock = threading.RLock()
        self.session_id: str | None = None

    @staticmethod
    def _blank_state() -> dict[str, Any]:
        return {
            "version": 1,
            "session_id": None,
            "mode": "PAPER",
            "runtime_status": "not_started",
            "started_at": None,
            "updated_at": None,
            "stopped_at": None,
            "operator_entry_pause": False,
            "operator_pause_reason": "",
            "system_entry_pause": False,
            "system_pause_reason": "",
            "supervisor": {"pid": None, "status": "not_started"},
            "workers": {},
        }

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._blank_state()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("coordinator state is not an object")
        except (OSError, ValueError, json.JSONDecodeError):
            return self._blank_state()
        state = self._blank_state()
        state.update(payload)
        state["workers"] = payload.get("workers", {}) if isinstance(payload.get("workers"), dict) else {}
        state["supervisor"] = payload.get("supervisor", {}) if isinstance(payload.get("supervisor"), dict) else {}
        return state

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _timestamp()
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(temp, self.path)

    def begin_session(self, *, mode: str, workers: list[str]) -> str:
        """Create a fresh supervisor session and make stale status impossible."""
        with self._lock:
            previous = self._load_unlocked()
            self.session_id = uuid.uuid4().hex
            state = self._blank_state()
            state.update(
                {
                    "session_id": self.session_id,
                    "mode": mode,
                    "runtime_status": "starting",
                    "started_at": _timestamp(),
                    # A deliberate operator pause survives process restarts.
                    # System pauses are recalculated from the new workers and
                    # clear only after every required worker is healthy.
                    "operator_entry_pause": bool(previous.get("operator_entry_pause")),
                    "operator_pause_reason": previous.get("operator_pause_reason", ""),
                    "supervisor": {"pid": os.getpid(), "status": "starting"},
                    "workers": {
                        name: {
                            "status": "starting",
                            "started_at": None,
                            "last_heartbeat_at": None,
                            "last_success_at": None,
                            "last_error": "",
                            "restart_count": 0,
                            "metrics": {},
                        }
                        for name in workers
                    },
                }
            )
            self._save_unlocked(state)
            return self.session_id

    def _update_worker_unlocked(
        self,
        state: dict[str, Any],
        name: str,
        *,
        status: str | None = None,
        detail: str | None = None,
        metrics: dict[str, Any] | None = None,
        restart_count: int | None = None,
        touch_heartbeat: bool = True,
    ) -> dict[str, Any]:
        worker = state.setdefault("workers", {}).setdefault(
            name,
            {
                "status": "starting",
                "started_at": None,
                "last_heartbeat_at": None,
                "last_success_at": None,
                "last_error": "",
                "restart_count": 0,
                "metrics": {},
            },
        )
        now = _timestamp()
        if worker.get("started_at") is None and status in {"starting", "running", "idle"}:
            worker["started_at"] = now
        if touch_heartbeat:
            worker["last_heartbeat_at"] = now
        if status is not None:
            worker["status"] = status
            if status in {"running", "idle"}:
                worker["last_success_at"] = now
                worker["last_error"] = ""
        if detail is not None:
            worker["last_error"] = detail
        if metrics is not None:
            worker["metrics"] = dict(metrics)
        if restart_count is not None:
            worker["restart_count"] = int(restart_count)
        return worker

    def worker_started(self, name: str, *, restart_count: int = 0) -> None:
        with self._lock:
            state = self._load_unlocked()
            self._update_worker_unlocked(state, name, status="starting", detail="", restart_count=restart_count)
            state["runtime_status"] = "running"
            state.setdefault("supervisor", {})["status"] = "running"
            self._save_unlocked(state)

    def heartbeat(
        self,
        name: str,
        *,
        status: str = "running",
        metrics: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            state = self._load_unlocked()
            self._update_worker_unlocked(state, name, status=status, metrics=metrics)
            if status in {"degraded", "failed"} and config.PAUSE_ALL_ON_WORKER_FAILURE:
                state["system_entry_pause"] = True
                state["system_pause_reason"] = f"{name} worker is {status}."
            state["runtime_status"] = "running"
            state.setdefault("supervisor", {})["status"] = "running"
            self._save_unlocked(state)

    def worker_failed(self, name: str, detail: str, *, restart_count: int) -> None:
        with self._lock:
            state = self._load_unlocked()
            self._update_worker_unlocked(
                state, name, status="failed", detail=detail, restart_count=restart_count
            )
            if config.PAUSE_ALL_ON_WORKER_FAILURE:
                state["system_entry_pause"] = True
                state["system_pause_reason"] = f"{name} worker failed: {detail}"
            self._save_unlocked(state)

    def worker_stale(self, name: str, detail: str) -> None:
        """Pause entries when a live worker stops publishing heartbeats."""
        with self._lock:
            state = self._load_unlocked()
            self._update_worker_unlocked(
                state,
                name,
                status="degraded",
                detail=detail,
                touch_heartbeat=False,
            )
            if config.PAUSE_ALL_ON_WORKER_FAILURE:
                state["system_entry_pause"] = True
                state["system_pause_reason"] = f"{name} worker heartbeat is stale: {detail}"
            self._save_unlocked(state)

    def worker_stopped(self, name: str, detail: str = "") -> None:
        with self._lock:
            state = self._load_unlocked()
            self._update_worker_unlocked(state, name, status="stopped", detail=detail)
            self._save_unlocked(state)

    def set_operator_pause(self, paused: bool, reason: str = "") -> None:
        with self._lock:
            state = self._load_unlocked()
            state["operator_entry_pause"] = bool(paused)
            state["operator_pause_reason"] = reason if paused else ""
            self._save_unlocked(state)

    def entries_allowed(self) -> tuple[bool, str]:
        state = self.snapshot()
        if state.get("operator_entry_pause"):
            return False, state.get("operator_pause_reason") or "Entries paused by operator."
        if state.get("system_entry_pause"):
            return False, state.get("system_pause_reason") or "Entries paused by runtime supervisor."
        return True, ""

    def clear_system_pause_if_healthy(self, required_workers: list[str]) -> bool:
        """Re-open system entry gate only when every configured worker is healthy."""
        with self._lock:
            state = self._load_unlocked()
            workers = state.get("workers", {})
            if not required_workers or not all(
                workers.get(name, {}).get("status") in {"running", "idle"}
                for name in required_workers
            ):
                return False
            if state.get("system_entry_pause"):
                state["system_entry_pause"] = False
                state["system_pause_reason"] = ""
                self._save_unlocked(state)
            return True

    def stop(self, detail: str = "") -> None:
        with self._lock:
            state = self._load_unlocked()
            state["runtime_status"] = "stopped"
            state["stopped_at"] = _timestamp()
            state.setdefault("supervisor", {})["status"] = "stopped"
            if detail:
                state["supervisor"]["detail"] = detail
            self._save_unlocked(state)

    def snapshot(self) -> dict[str, Any]:
        """Return a copy with derived liveness fields for UI/API consumers."""
        with self._lock:
            state = deepcopy(self._load_unlocked())

        now = _now()
        healthy = True
        for worker in state.get("workers", {}).values():
            heartbeat = worker.get("last_heartbeat_at")
            age_seconds = None
            if heartbeat:
                try:
                    timestamp = datetime.fromisoformat(str(heartbeat))
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    age_seconds = max(0.0, (now - timestamp.astimezone(timezone.utc)).total_seconds())
                except ValueError:
                    pass
            worker["heartbeat_age_seconds"] = age_seconds
            worker["healthy"] = (
                worker.get("status") in {"running", "idle"}
                and age_seconds is not None
                and age_seconds <= self.heartbeat_ttl_seconds
            )
            if not worker["healthy"]:
                healthy = False

        state["healthy"] = bool(state.get("runtime_status") == "running" and healthy)
        state["entries_allowed"] = not (
            state.get("operator_entry_pause") or state.get("system_entry_pause")
        )
        return state
