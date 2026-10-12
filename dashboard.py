"""
Local monitoring dashboard for the Alpaca trading bot.

This stays intentionally read-only with one exception: it can run the
offline logic-check script and show its output. It does not start/stop the
trading loop or place/cancel orders from the browser.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, url_for

import config
from bot.brokers import create_broker
from bot.runtime_coordinator import RuntimeCoordinator
from bot.state_store import RuntimeStateStore

ROOT = Path(__file__).resolve().parent
LOGIC_TEST_PYTHON = ROOT / ".venv" / "bin" / "python"
LAST_LOGIC_CHECK = None
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 5001
_ACCOUNT_SNAPSHOT_LOCK = threading.Lock()
_ACCOUNT_SNAPSHOT = {
    "connected": False,
    "error": "Account data has not been fetched yet.",
    "info": None,
    "positions": [],
}
_ACCOUNT_SNAPSHOT_UPDATED_AT = 0.0
_ACCOUNT_SNAPSHOT_REFRESHING = False


def build_strategy_cards() -> list[dict]:
    return [
        {
            "name": "Mean Reversion",
            "symbols": ", ".join(config.MEAN_REVERSION_SYMBOLS),
            "timeframe": (
                f"{config.MEAN_REVERSION_TIMEFRAME_AMOUNT} "
                f"{config.MEAN_REVERSION_TIMEFRAME_UNIT} candles"
            ),
            "description": "Fades statistically stretched SPY and QQQ moves back toward the mean.",
            "details": [
                f"Lookback: {config.MEAN_REVERSION_LOOKBACK}",
                f"SPY threshold: {config.MEAN_REVERSION_PARAMS['SPY']['std_multiplier']} stdev",
                f"QQQ threshold: {config.MEAN_REVERSION_PARAMS['QQQ']['std_multiplier']} stdev",
                "Exit: back to the mean",
                "Trailing stop: 1.5x ATR (added to lock in profits and prevent trend-fade losses)"
            ],
        },
        {
            "name": "Momentum Breakout",
            "symbols": ", ".join(config.MOMENTUM_SYMBOLS),
            "timeframe": f"{config.MOMENTUM_TIMEFRAME_AMOUNT} {config.MOMENTUM_TIMEFRAME_UNIT} candles",
            "description": "Buys confirmed BTC/USD breakouts with a volume filter and manages risk with a ratcheting ATR stop.",
            "details": [
                f"Donchian lookback: {config.MOMENTUM_LOOKBACK}",
                f"Volume filter: {config.MOMENTUM_VOLUME_MULTIPLIER}x average",
                f"Trail stop: {config.MOMENTUM_TRAIL_ATR_MULTIPLIER}x ATR",
                "Shorting: disabled for crypto spot",
            ],
        },
        {
            "name": "Trend Following",
            "symbols": ", ".join(config.TREND_SYMBOLS),
            "timeframe": f"{config.TREND_TIMEFRAME_AMOUNT} {config.TREND_TIMEFRAME_UNIT} candles",
            "description": "Rides GLD and USO trend changes with EMA crossovers and a wider trailing stop.",
            "details": [
                f"Fast EMA: {config.TREND_FAST_EMA}",
                f"Slow EMA: {config.TREND_SLOW_EMA}",
                f"Trail stop: {config.TREND_TRAIL_ATR_MULTIPLIER}x ATR",
                f"Shorting enabled: {'yes' if config.TREND_ALLOW_SHORT else 'no'}",
            ],
        },
    ]


def detect_bot_status() -> dict:
    coordinated = RuntimeCoordinator().snapshot()
    if coordinated.get("session_id"):
        workers = coordinated.get("workers", {})
        unhealthy_workers = [
            name for name, worker in workers.items() if not worker.get("healthy")
        ]
        running = bool(coordinated.get("healthy"))
        return {
            "running": running,
            "count": len(workers),
            "processes": [],
            "detail": (
                "Supervised workers are healthy."
                if running
                else "Runtime state is stale or one or more supervised workers are unhealthy: "
                + (", ".join(unhealthy_workers) or "supervisor stopped")
            ),
            "source": "runtime_coordinator",
        }
    try:
        result = subprocess.run(
            ["pgrep", "-af", "python.*run_bot.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {
            "running": False,
            "count": 0,
            "processes": [],
            "detail": "Process inspection is unavailable on this machine.",
        }

    processes = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, command = line.partition(" ")
        processes.append({"pid": pid, "command": command})

    return {
        "running": bool(processes),
        "count": len(processes),
        "processes": processes,
        "detail": "Trading loop detected." if processes else "Trading loop is not running.",
    }


def _fetch_account_snapshot() -> dict:
    """Perform the potentially slow broker reads outside a web request."""
    missing_fields = config.missing_alpaca_credential_fields()
    if missing_fields:
        return {
            "connected": False,
            "error": f"Missing credentials: {', '.join(missing_fields)}",
            "info": None,
            "positions": [],
        }

    try:
        broker = create_broker()
        info = broker.get_account_info()
        positions = broker.get_positions()
        return {"connected": True, "error": None, "info": info, "positions": positions}
    except Exception as exc:
        return {"connected": False, "error": str(exc), "info": None, "positions": []}


def _refresh_account_snapshot():
    global _ACCOUNT_SNAPSHOT, _ACCOUNT_SNAPSHOT_UPDATED_AT, _ACCOUNT_SNAPSHOT_REFRESHING
    try:
        snapshot = _fetch_account_snapshot()
    except Exception as exc:  # defensive: a monitor must not crash its refresher
        snapshot = {"connected": False, "error": str(exc), "info": None, "positions": []}
    finally:
        with _ACCOUNT_SNAPSHOT_LOCK:
            _ACCOUNT_SNAPSHOT = snapshot
            _ACCOUNT_SNAPSHOT_UPDATED_AT = time.monotonic()
            _ACCOUNT_SNAPSHOT_REFRESHING = False


def _schedule_account_snapshot_refresh():
    """Start at most one background account refresh when the cache is stale."""
    global _ACCOUNT_SNAPSHOT_REFRESHING
    if config.missing_alpaca_credential_fields():
        return
    now = time.monotonic()
    with _ACCOUNT_SNAPSHOT_LOCK:
        is_fresh = now - _ACCOUNT_SNAPSHOT_UPDATED_AT < config.DASHBOARD_ACCOUNT_REFRESH_SECONDS
        if is_fresh or _ACCOUNT_SNAPSHOT_REFRESHING:
            return
        _ACCOUNT_SNAPSHOT_REFRESHING = True
    threading.Thread(
        target=_refresh_account_snapshot,
        daemon=True,
        name="dashboard-account-refresh",
    ).start()


def load_account_snapshot() -> dict:
    """Return a cached account state immediately; refresh broker I/O in background."""
    missing_fields = config.missing_alpaca_credential_fields()
    if missing_fields:
        return {
            "connected": False,
            "error": f"Missing credentials: {', '.join(missing_fields)}",
            "info": None,
            "positions": [],
            "refreshing": False,
        }

    _schedule_account_snapshot_refresh()
    with _ACCOUNT_SNAPSHOT_LOCK:
        snapshot = dict(_ACCOUNT_SNAPSHOT)
        snapshot["positions"] = list(_ACCOUNT_SNAPSHOT.get("positions", []))
        snapshot["refreshing"] = _ACCOUNT_SNAPSHOT_REFRESHING
    return snapshot


def _read_csv_rows(path: Path, limit: int = 10) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    return df.tail(limit).iloc[::-1].to_dict(orient="records")


def _read_jsonl_rows(path: Path, limit: int = 12) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[-limit:][::-1]


def load_runtime_snapshot() -> dict:
    state = RuntimeStateStore().load()
    coordination = load_coordination_snapshot()
    coordinated_session = bool(coordination.get("session_id"))
    workers = coordination.get("workers", {})
    us_worker = workers.get("us", {})
    us_metrics = us_worker.get("metrics", {})
    coordinated_status = (
        "running"
        if coordination.get("healthy")
        else "stopped"
        if coordination.get("runtime_status") == "stopped"
        else "degraded"
    )
    pause_reason = (
        coordination.get("operator_pause_reason")
        or coordination.get("system_pause_reason")
        or ""
    )
    if state is None:
        return {
            "available": coordinated_session,
            "status": coordinated_status if coordinated_session else "not_started",
            "mode": coordination.get("mode", "PAPER" if config.ALPACA_PAPER else "LIVE"),
            "last_heartbeat": us_worker.get("last_heartbeat_at") if coordinated_session else None,
            "broker_connected": bool(us_metrics.get("broker_connected", False)) if coordinated_session else False,
            "auth_error": False,
            "last_broker_error": "",
            "day_start_equity": 0.0,
            "last_equity": 0.0,
            "daily_realized_pnl": 0.0,
            "cumulative_realized_pnl": 0.0,
            "entry_gate": (
                "open" if coordination.get("entries_allowed") else "halted"
            ) if coordinated_session else "unknown",
            "entry_halt_reason": pause_reason,
            "open_positions": [],
            "signals": [],
            "market_data": us_metrics.get("market_data", {}),
            "guardrails": {},
            "recent_notes": [],
            "workers": workers,
        }

    positions = []
    for symbol, position in state.open_positions.items():
        positions.append(
            {
                "symbol": symbol,
                "side": position.get("side", ""),
                "qty": float(position.get("qty", 0.0)),
                "entry_price": float(position.get("entry_price", 0.0)),
                "stop_price": float(position.get("stop_price", 0.0)),
                "strategy": position.get("strategy", ""),
                "risk_amount": float(position.get("risk_amount", 0.0)),
                "notional": float(position.get("notional", 0.0)),
                "asset_class": position.get("asset_class", ""),
                "adopted": bool(position.get("adopted", False)),
            }
        )

    signals = []
    for symbol, signal in state.last_signal_by_symbol.items():
        payload = signal.copy()
        payload["symbol"] = symbol
        signals.append(payload)
    signals.sort(key=lambda item: item.get("timestamp", ""), reverse=True)

    last_heartbeat = (
        us_worker.get("last_heartbeat_at")
        if coordinated_session
        else state.last_heartbeat_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if state.last_heartbeat_at
        else "-"
    )
    res = {
        "available": True,
        "status": coordinated_status if coordinated_session else state.runtime_status,
        "mode": coordination.get("mode", state.mode) if coordinated_session else state.mode,
        "last_heartbeat": last_heartbeat,
        "broker_connected": bool(us_metrics.get("broker_connected", False)) if coordinated_session else state.broker_connected,
        "auth_error": state.auth_error,
        "last_broker_error": state.last_broker_error,
        "degraded_since": state.degraded_since.strftime("%Y-%m-%d %H:%M:%S UTC") if state.degraded_since else None,
        "recovery_count": state.recovery_count,
        "adopted_position_policy": config.ADOPTED_POSITION_POLICY,
        "day_start_equity": float(state.day_start_equity or 0.0),
        "last_equity": float(state.last_equity or 0.0),
        "daily_realized_pnl": float(state.daily_realized_pnl),
        "cumulative_realized_pnl": float(state.cumulative_realized_pnl),
        "wins": state.wins,
        "losses": state.losses,
        "consecutive_losses": state.consecutive_losses,
        "entry_gate": (
            "open" if coordination.get("entries_allowed") and state.new_entries_allowed else "halted"
        ) if coordinated_session else ("open" if state.new_entries_allowed else "halted"),
        "entry_halt_reason": pause_reason or state.entry_halt_reason,
        "open_positions": positions,
        "signals": signals[:8],
        "market_data": state.market_data_snapshot or us_metrics.get("market_data", {}),
        "guardrails": state.latest_guardrail_snapshot,
        "recent_notes": state.recent_notes[:6],
        "workers": workers,
    }

    try:
        from bot.api_command_center import get_command_center_data
        res["command_center"] = get_command_center_data()
    except Exception:
        res["command_center"] = {}
        
    return res


def load_coordination_snapshot() -> dict:
    """Return the cross-market operational status without combining currencies."""
    return RuntimeCoordinator().snapshot()


def load_trade_log(path: Path | None = None) -> dict:
    path = Path(path or ROOT / config.TRADES_LOG_PATH)
    if not path.exists() or path.stat().st_size == 0:
        return {
            "rows": [],
            "summary": {
                "count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl": 0.0,
                "adopted_count": 0,
                "adopted_realized_pnl": 0.0,
                "total_count": 0,
                "total_realized_pnl": 0.0,
            },
        }

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    if df.empty:
        return {
            "rows": [],
            "summary": {
                "count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "realized_pnl": 0.0,
                "adopted_count": 0,
                "adopted_realized_pnl": 0.0,
                "total_count": 0,
                "total_realized_pnl": 0.0,
            },
        }

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce").fillna(0.0)
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").fillna(0.0)
    df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce").fillna(0.0)
    df["position_size"] = pd.to_numeric(df["position_size"], errors="coerce").fillna(0.0)
    df["strategy"] = df.get("strategy", "").fillna("")
    df["is_adopted"] = df["strategy"].eq("adopted_on_restart")
    df = df.sort_values("timestamp", ascending=False, na_position="last")

    strategy_df = df.loc[~df["is_adopted"]].copy()
    adopted_df = df.loc[df["is_adopted"]].copy()

    wins = int((strategy_df["profit_loss"] > 0).sum())
    losses = int((strategy_df["profit_loss"] < 0).sum())
    count = int(len(strategy_df))
    win_rate = (wins / count * 100.0) if count else 0.0

    rows = []
    for _, row in df.head(10).iterrows():
        timestamp = row["timestamp"]
        rows.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(timestamp) else "-",
                "instrument": row.get("instrument", ""),
                "direction": row.get("direction", ""),
                "entry_price": float(row["entry_price"]),
                "exit_price": float(row["exit_price"]),
                "profit_loss": float(row["profit_loss"]),
                "position_size": float(row["position_size"]),
                "strategy": row.get("strategy", "") or "-",
                "reason": row.get("reason", "") or "-",
                "is_adopted": bool(row.get("is_adopted", False)),
                "source_label": "Restart sync" if bool(row.get("is_adopted", False)) else "Bot signal",
            }
        )

    return {
        "rows": rows,
        "summary": {
            "count": count,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "realized_pnl": round(float(strategy_df["profit_loss"].sum()), 2),
            "adopted_count": int(len(adopted_df)),
            "adopted_realized_pnl": round(float(adopted_df["profit_loss"].sum()), 2),
            "total_count": int(len(df)),
            "total_realized_pnl": round(float(df["profit_loss"].sum()), 2),
        },
    }


def build_sparkline_points(values: list[float], width: int = 420, height: int = 160, pad: int = 16) -> str:
    if not values:
        return ""

    if len(values) == 1:
        values = [values[0], values[0]]

    low = min(values)
    high = max(values)
    span = high - low or 1.0
    x_step = (width - 2 * pad) / (len(values) - 1)

    points = []
    for index, value in enumerate(values):
        x = pad + index * x_step
        normalized = (value - low) / span
        y = height - pad - normalized * (height - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def load_daily_pnl(path: Path | None = None) -> dict:
    path = Path(path or ROOT / config.DAILY_PNL_LOG_PATH)
    if not path.exists() or path.stat().st_size == 0:
        return {
            "rows": [],
            "summary": {"days": 0, "latest_equity": 0.0, "cumulative_pnl": 0.0, "best_day": 0.0},
            "chart_points": "",
        }

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    if df.empty:
        return {
            "rows": [],
            "summary": {"days": 0, "latest_equity": 0.0, "cumulative_pnl": 0.0, "best_day": 0.0},
            "chart_points": "",
        }

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["starting_equity"] = pd.to_numeric(df["starting_equity"], errors="coerce").fillna(0.0)
    df["ending_equity"] = pd.to_numeric(df["ending_equity"], errors="coerce").fillna(0.0)
    df["daily_pnl"] = pd.to_numeric(df["daily_pnl"], errors="coerce").fillna(0.0)
    df = df.sort_values("date")

    rows = []
    for _, row in df.tail(10).iloc[::-1].iterrows():
        date_value = row["date"]
        rows.append(
            {
                "date": date_value.strftime("%Y-%m-%d") if pd.notna(date_value) else "-",
                "starting_equity": float(row["starting_equity"]),
                "ending_equity": float(row["ending_equity"]),
                "daily_pnl": float(row["daily_pnl"]),
            }
        )

    return {
        "rows": rows,
        "summary": {
            "days": int(len(df)),
            "latest_equity": round(float(df.iloc[-1]["ending_equity"]), 2),
            "cumulative_pnl": round(float(df["daily_pnl"].sum()), 2),
            "best_day": round(float(df["daily_pnl"].max()), 2),
        },
        "chart_points": build_sparkline_points(df["ending_equity"].tolist()),
    }


def _build_portfolio_curve_payload(
    values: list[float],
    *,
    source: str,
    source_label: str,
    badge_label: str,
    session_anchor: float | None = None,
) -> dict:
    latest = float(values[-1])
    start = float(values[0])
    visible_low = float(min(values))
    visible_high = float(max(values))
    session_anchor = start if session_anchor in (None, 0) else float(session_anchor)
    return {
        "available": True,
        "source": source,
        "chart_points": build_sparkline_points(values),
        "summary": {
            "points": len(values),
            "badge_label": badge_label,
            "latest_equity": round(latest, 2),
            "window_change": round(latest - start, 2),
            "session_change": round(latest - session_anchor, 2),
            "visible_low": round(visible_low, 2),
            "visible_high": round(visible_high, 2),
            "visible_range": round(visible_high - visible_low, 2),
            "source_label": source_label,
            "note": "Zoomed to the visible range so small equity moves stay readable.",
        },
    }


def load_portfolio_curve(
    live_path: Path | None = None,
    daily_path: Path | None = None,
) -> dict:
    live_path = Path(live_path or ROOT / config.runtime_path(config.EQUITY_CURVE_LOG_PATH))
    if live_path.exists() and live_path.stat().st_size > 0:
        try:
            live_df = pd.read_csv(live_path)
        except pd.errors.EmptyDataError:
            live_df = pd.DataFrame()

        if not live_df.empty:
            live_df["timestamp"] = pd.to_datetime(live_df["timestamp"], utc=True, errors="coerce")
            live_df["equity"] = pd.to_numeric(live_df["equity"], errors="coerce")
            live_df["day_start_equity"] = pd.to_numeric(live_df.get("day_start_equity"), errors="coerce")
            live_df = live_df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")

            if not live_df.empty:
                recent = live_df.tail(240)
                session_anchor = recent["day_start_equity"].dropna()
                return _build_portfolio_curve_payload(
                    recent["equity"].astype(float).tolist(),
                    source="live_snapshots",
                    source_label="Live bot heartbeats (latest 240 snapshots max)",
                    badge_label=f"{len(recent)} snapshots",
                    session_anchor=None if session_anchor.empty else float(session_anchor.iloc[-1]),
                )

    daily_path = Path(daily_path or ROOT / config.DAILY_PNL_LOG_PATH)
    if daily_path.exists() and daily_path.stat().st_size > 0:
        try:
            daily_df = pd.read_csv(daily_path)
        except pd.errors.EmptyDataError:
            daily_df = pd.DataFrame()

        if not daily_df.empty:
            daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
            daily_df["starting_equity"] = pd.to_numeric(daily_df["starting_equity"], errors="coerce")
            daily_df["ending_equity"] = pd.to_numeric(daily_df["ending_equity"], errors="coerce")
            daily_df = daily_df.dropna(subset=["date", "ending_equity"]).sort_values("date")

            if not daily_df.empty:
                return _build_portfolio_curve_payload(
                    daily_df["ending_equity"].astype(float).tolist(),
                    source="daily_closes",
                    source_label="Daily close-to-close history",
                    badge_label=f"{len(daily_df)} tracked days",
                    session_anchor=float(daily_df.iloc[-1]["starting_equity"]),
                )

    return {
        "available": False,
        "source": "none",
        "chart_points": "",
        "summary": {
            "points": 0,
            "badge_label": "0 snapshots",
            "latest_equity": 0.0,
            "window_change": 0.0,
            "session_change": 0.0,
            "visible_low": 0.0,
            "visible_high": 0.0,
            "visible_range": 0.0,
            "source_label": "No equity history yet.",
            "note": "",
        },
    }


def load_signal_log(path: Path | None = None) -> dict:
    rows = _read_csv_rows(path or ROOT / config.runtime_path(config.SIGNALS_LOG_PATH), limit=10)
    if not rows:
        return {"rows": [], "summary": {"count": 0, "entered": 0, "blocked": 0, "filtered": 0}}

    counts = {"entered": 0, "blocked": 0, "filtered": 0}
    for row in rows:
        status = str(row.get("status", "")).lower()
        if status in counts:
            counts[status] += 1
    return {"rows": rows, "summary": {"count": len(rows), **counts}}


def load_guardrail_log(path: Path | None = None) -> dict:
    rows = _read_csv_rows(path or ROOT / config.runtime_path(config.GUARDRAILS_LOG_PATH), limit=10)
    blocking_statuses = {"blocked", "halted", "rejected", "denied"}
    return {
        "rows": rows,
        "summary": {
            "count": len(rows),
            "blocked": sum(
                1 for row in rows if str(row.get("status", "")).strip().lower() in blocking_statuses
            ),
        },
    }


def load_event_log(path: Path | None = None) -> dict:
    rows = _read_jsonl_rows(path or ROOT / config.runtime_path(config.EVENTS_LOG_PATH), limit=12)
    return {"rows": rows, "summary": {"count": len(rows)}}


def load_backtest_summary(path: Path | None = None) -> dict:
    path = Path(path or ROOT / config.runtime_path(config.LATEST_BACKTEST_SUMMARY_PATH))
    if not path.exists():
        return {"available": False, "summary": None}
    with open(path, "r", encoding="utf-8") as handle:
        return {"available": True, "summary": json.load(handle)}


def run_logic_checks() -> dict:
    python_executable = LOGIC_TEST_PYTHON if LOGIC_TEST_PYTHON.exists() else Path(sys.executable)
    command = [str(python_executable), "tests/test_logic.py"]

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output = f"{output}\n\nstderr:\n{result.stderr.strip()}".strip()
        return {
            "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "command": shlex.join(command),
            "output": output or "(no output)",
        }
    except subprocess.TimeoutExpired:
        return {
            "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "ok": False,
            "returncode": -1,
            "command": shlex.join(command),
            "output": "Logic checks timed out after 120 seconds.",
        }


def parse_dashboard_port(raw_value: str | None, default: int = DEFAULT_DASHBOARD_PORT) -> int:
    """Parses a user-supplied dashboard port and validates the range."""
    if raw_value is None or not raw_value.strip():
        return default

    try:
        port = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid DASHBOARD_PORT={raw_value!r}. Use an integer between 1 and 65535.") from exc

    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid DASHBOARD_PORT={raw_value!r}. Use an integer between 1 and 65535.")
    return port


def is_port_available(host: str, port: int) -> bool:
    """True when a host:port pair can be bound for a local dev server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_dashboard_port(
    host: str = DEFAULT_DASHBOARD_HOST,
    preferred_port: int = DEFAULT_DASHBOARD_PORT,
    search_limit: int = 20,
) -> int:
    """
    Returns the preferred port when available, otherwise the next free
    localhost port within a small forward search window.
    """
    for port in range(preferred_port, preferred_port + search_limit + 1):
        if is_port_available(host, port):
            return port
    raise RuntimeError(
        f"Could not find a free dashboard port between {preferred_port} and "
        f"{preferred_port + search_limit} on {host}."
    )


def format_dashboard_start_message(host: str, requested_port: int, selected_port: int) -> str:
    url = f"http://{host}:{selected_port}"
    if selected_port == requested_port:
        return f"Dashboard ready at {url}"
    return (
        f"Dashboard port {requested_port} is busy. Using {url} instead. "
        f"Set DASHBOARD_PORT={selected_port} if you want to keep this port fixed."
    )


def load_portfolio_heatmap() -> dict:
    state = RuntimeStateStore().load()
    if not state or not state.open_positions:
        return {"has_positions": False, "total_exposure_pct": 0, "classes": {}}
    
    total_equity = float(state.last_equity) if state.last_equity else 1.0
    heatmap = {}
    total_exposure = 0.0
    
    for sym, p in state.open_positions.items():
        notional = float(p.get("notional", 0.0))
        total_exposure += notional
        ac = p.get("asset_class") or "equity"
        if ac not in heatmap:
            heatmap[ac] = {"total_notional": 0.0, "positions": []}
        heatmap[ac]["total_notional"] += notional
        
        # Determine color based on side and PNL if available
        color_class = "long" if p.get("side") == "long" else "short"
        
        heatmap[ac]["positions"].append({
            "symbol": sym,
            "side": p.get("side", ""),
            "notional": notional,
            "weight": round(notional / total_equity * 100, 2),
            "color": color_class
        })
        
    for ac, data in heatmap.items():
        data["weight"] = round(data["total_notional"] / total_equity * 100, 2)
        # Sort positions inside class by size
        data["positions"].sort(key=lambda x: x["notional"], reverse=True)
        
    return {
        "has_positions": True,
        "total_exposure_pct": round(total_exposure / total_equity * 100, 2),
        "classes": heatmap
    }



def get_consensus_markdown():
    import os, config
    path = os.path.join(config.ROOT_DIR, "runtime", "consensus.md")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return "No consensus file found."


def make_json_safe(value):
    """Return JSON-valid dashboard data, replacing non-finite log values with null.

    CSV ingestion can produce ``NaN`` for an omitted column. Python's JSON encoder
    historically permits that token, but it is not valid JSON and browsers reject it
    when polling ``/api/snapshot``. Keep the monitor resilient to incomplete rows.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return str(value)

def collect_dashboard_snapshot() -> dict:
    return {
        "title": "Alpaca Trading Bot Dashboard",
        "mode": "PAPER" if config.ALPACA_PAPER else "LIVE",
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "consensus_md": get_consensus_markdown(),
        "bot_status": detect_bot_status(),
        "coordination": load_coordination_snapshot(),
        "runtime": load_runtime_snapshot(),
        "account": load_account_snapshot(),
        "trades": load_trade_log(),
        "daily_pnl": load_daily_pnl(),
        "portfolio_curve": load_portfolio_curve(),
        "heatmap": load_portfolio_heatmap(),
        "signals": load_signal_log(),
        "guardrails": load_guardrail_log(),
        "events": load_event_log(),
        "backtest": load_backtest_summary(),
        "symbols": getattr(config, "SYMBOLS", []),
        "strategies": build_strategy_cards(),
        "risk_per_trade_pct": config.RISK_PER_TRADE_PCT * 100.0,
        "hard_stop_atr_multiple": config.HARD_STOP_ATR_MULTIPLE,
        "max_position_pct": (
            config.MAX_POSITION_PCT_OF_EQUITY * 100.0
            if config.MAX_POSITION_PCT_OF_EQUITY is not None
            else None
        ),
        "max_open_positions": config.MAX_OPEN_POSITIONS,
        "max_gross_exposure_pct": config.MAX_GROSS_EXPOSURE_PCT_OF_EQUITY * 100.0,
        "max_portfolio_heat_pct": config.MAX_PORTFOLIO_HEAT_PCT_OF_EQUITY * 100.0,
        "max_daily_loss_pct": config.MAX_DAILY_LOSS_PCT_OF_EQUITY * 100.0,
        "bot_command": config.BOT_LAUNCH_COMMAND,
        "dashboard_command": config.DASHBOARD_LAUNCH_COMMAND,
        "logic_command": f"{LOGIC_TEST_PYTHON if LOGIC_TEST_PYTHON.exists() else sys.executable} tests/test_logic.py",
        "backtest_command": config.BACKTEST_LAUNCH_COMMAND,
    }


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))


    @app.get("/api/snapshot")
    def api_snapshot():
        return jsonify(make_json_safe(collect_dashboard_snapshot()))

    @app.get("/api/runtime")
    def api_runtime():
        return jsonify(make_json_safe(load_coordination_snapshot()))

    @app.get("/healthz")
    def api_healthz():
        runtime = load_coordination_snapshot()
        status = 200 if runtime.get("healthy") else 503
        return jsonify(make_json_safe({"status": "healthy" if status == 200 else "unhealthy", "runtime": runtime})), status

    @app.get("/")
    def dashboard_home():
        snapshot = make_json_safe(collect_dashboard_snapshot())
        return render_template(
            "dashboard.html",
            snapshot=snapshot,
            logic_check=LAST_LOGIC_CHECK,
        )

    @app.get("/api/bars/<path:symbol>")
    def api_bars(symbol):
        if not hasattr(app, "api_bars_cache"):
            app.api_bars_cache = {}
            app.api_bars_lock = threading.Lock()

        now = time.time()
        with app.api_bars_lock:
            cached = app.api_bars_cache.get(symbol)
            if cached and now - cached["time"] < config.DASHBOARD_BARS_REFRESH_SECONDS:
                return {"data": cached.get("data", []), "refreshing": cached.get("refreshing", False)}
            if cached and cached.get("refreshing"):
                return {"data": cached.get("data", []), "refreshing": True}
            app.api_bars_cache[symbol] = {
                "time": cached["time"] if cached else 0.0,
                "data": cached.get("data", []) if cached else [],
                "refreshing": True,
                "error": None,
            }

        def refresh_bars():
            try:
                broker = create_broker()
                bars = broker.get_bars(symbol, 1, "Hour", limit=200)
                data = [
                    {
                        "time": int(timestamp.timestamp()),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    }
                    for timestamp, row in bars.iterrows()
                ]
                result = {"time": time.time(), "data": data, "refreshing": False, "error": None}
            except Exception as exc:
                result = {
                    "time": time.time(),
                    "data": [],
                    "refreshing": False,
                    "error": str(exc),
                }
            with app.api_bars_lock:
                app.api_bars_cache[symbol] = result

        threading.Thread(target=refresh_bars, daemon=True, name=f"dashboard-bars-{symbol}").start()
        return {"data": cached.get("data", []) if cached else [], "refreshing": True}

    @app.post("/run-logic-checks")
    def dashboard_run_logic_checks():
        global LAST_LOGIC_CHECK
        LAST_LOGIC_CHECK = run_logic_checks()
        return redirect(url_for("dashboard_home"))

    return app


app = create_app()


if __name__ == "__main__":
    try:
        host = DEFAULT_DASHBOARD_HOST
        requested_port = parse_dashboard_port(os.getenv("DASHBOARD_PORT"))
        selected_port = choose_dashboard_port(host=host, preferred_port=requested_port)
        print(format_dashboard_start_message(host, requested_port, selected_port))
        app.run(host=host, port=selected_port, debug=False)
    except (RuntimeError, ValueError) as exc:
        print(f"\n{exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        sys.exit(0)
