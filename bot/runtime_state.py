"""
Serializable runtime/session state for the Alpaca bot.

This lets the loop survive restarts with enough context to resume safely:
last processed bars, open-position metadata, rolling P&L, guardrail state,
and recent signal decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: _serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    return value


@dataclass
class RuntimeState:
    mode: str
    current_date: str = ""
    day_start_equity: float | None = None
    last_equity: float | None = None
    daily_realized_pnl: float = 0.0
    cumulative_realized_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    runtime_status: str = "initializing"
    broker_connected: bool = False
    last_broker_error: str = ""
    auth_error: bool = False
    degraded_since: datetime | None = None
    recovery_count: int = 0
    new_entries_allowed: bool = True
    entry_halt_reason: str = ""
    started_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    last_sync_at: datetime | None = None
    last_processed_bar: dict[str, str | None] = field(default_factory=dict)
    market_data_snapshot: dict[str, Any] = field(default_factory=dict)
    open_positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_signal_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_guardrail_snapshot: dict[str, Any] = field(default_factory=dict)
    recent_notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.recent_notes = [message] + self.recent_notes[:9]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimeState":
        state = cls(
            mode=payload.get("mode", "PAPER"),
            current_date=payload.get("current_date", ""),
            day_start_equity=payload.get("day_start_equity"),
            last_equity=payload.get("last_equity"),
            daily_realized_pnl=float(payload.get("daily_realized_pnl", 0.0)),
            cumulative_realized_pnl=float(payload.get("cumulative_realized_pnl", 0.0)),
            wins=int(payload.get("wins", 0)),
            losses=int(payload.get("losses", 0)),
            consecutive_losses=int(payload.get("consecutive_losses", 0)),
            runtime_status=payload.get("runtime_status", "initializing"),
            broker_connected=bool(payload.get("broker_connected", False)),
            last_broker_error=payload.get("last_broker_error", ""),
            auth_error=bool(payload.get("auth_error", False)),
            degraded_since=datetime.fromisoformat(payload["degraded_since"]) if payload.get("degraded_since") else None,
            recovery_count=int(payload.get("recovery_count", 0)),
            new_entries_allowed=bool(payload.get("new_entries_allowed", True)),
            entry_halt_reason=payload.get("entry_halt_reason", ""),
            started_at=datetime.fromisoformat(payload["started_at"]) if payload.get("started_at") else None,
            last_heartbeat_at=datetime.fromisoformat(payload["last_heartbeat_at"]) if payload.get("last_heartbeat_at") else None,
            last_sync_at=datetime.fromisoformat(payload["last_sync_at"]) if payload.get("last_sync_at") else None,
            last_processed_bar=payload.get("last_processed_bar", {}) or {},
            market_data_snapshot=payload.get("market_data_snapshot", {}) or {},
            open_positions=payload.get("open_positions", {}) or {},
            last_signal_by_symbol=payload.get("last_signal_by_symbol", {}) or {},
            latest_guardrail_snapshot=payload.get("latest_guardrail_snapshot", {}) or {},
            recent_notes=list(payload.get("recent_notes", []) or []),
        )
        if state.day_start_equity is not None:
            state.day_start_equity = float(state.day_start_equity)
        if state.last_equity is not None:
            state.last_equity = float(state.last_equity)
        return state
