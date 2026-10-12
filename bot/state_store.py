"""
Small JSON state store for the runtime snapshot.

Improvements over the original:
- Catches ``json.JSONDecodeError`` so a corrupt state file doesn't crash the
  bot -- it backs up the corrupt file and returns ``None`` (clean start).
- Atomic save was already in place (``.tmp`` + ``Path.replace``).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import config
from bot.runtime_state import RuntimeState


class RuntimeStateStore:
    def __init__(self, path: str | None = None):
        config.ensure_runtime_directories()
        self.path = Path(path or config.runtime_path(config.STATE_PATH))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> RuntimeState | None:
        if not self.path.exists():
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return RuntimeState.from_dict(json.load(handle))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # Back up the corrupt file so we can inspect it later, then
            # return None so the bot starts with a clean slate.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup = self.path.with_suffix(f".corrupt.{stamp}.json")
            try:
                shutil.copy2(self.path, backup)
            except OSError:
                pass
            print(
                f"[state_store] WARNING: could not parse {self.path} "
                f"({type(exc).__name__}: {exc}). "
                f"Backed up to {backup.name} and starting fresh."
            )
            return None

    def save(self, state: RuntimeState) -> Path:
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=True, default=str)
        tmp_path.replace(self.path)
        return self.path
