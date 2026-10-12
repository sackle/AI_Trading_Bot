"""Single-process guard for order-producing runtime launches."""

from __future__ import annotations

import os
from pathlib import Path

import config


class RuntimeProcessLock:
    """Advisory POSIX lock that prevents a second launcher from double-trading."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or config.runtime_path(config.RUNTIME_LOCK_PATH))
        self._handle = None

    def acquire(self) -> None:
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - project runs on macOS/Linux
            raise RuntimeError("This runtime requires POSIX file locking.") from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                "Another trading runtime already owns the lock. "
                f"Inspect {self.path} before starting a second process."
            ) from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(str(os.getpid()))
        self._handle.flush()

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
