"""
Structured runtime artifacts beyond the core trade/P&L CSVs.

Event log rotation: ``session_events.jsonl`` is automatically rotated when
it exceeds ``_EVENT_LOG_MAX_BYTES`` (1 MB by default). Old files are kept
alongside the current one as ``session_events.<ISO-date>.jsonl``.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

SIGNAL_FIELDS = [
    "timestamp",
    "symbol",
    "strategy",
    "side",
    "action",
    "status",
    "price",
    "bar_time",
    "timeframe",
    "reason",
]

GUARDRAIL_FIELDS = [
    "timestamp",
    "rule_name",
    "symbol",
    "strategy",
    "status",
    "detail",
    "metrics_json",
]

# Rotate when the JSONL event log exceeds this size.
_EVENT_LOG_MAX_BYTES = 1_000_000  # 1 MB


def _ensure_header(path: Path, fields: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(fields)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def event_path() -> Path:
    config.ensure_runtime_directories()
    path = Path(config.runtime_path(config.EVENTS_LOG_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def signals_path() -> Path:
    config.ensure_runtime_directories()
    path = Path(config.runtime_path(config.SIGNALS_LOG_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def guardrails_path() -> Path:
    config.ensure_runtime_directories()
    path = Path(config.runtime_path(config.GUARDRAILS_LOG_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def latest_backtest_summary_path() -> Path:
    config.ensure_runtime_directories()
    return Path(config.runtime_path(config.LATEST_BACKTEST_SUMMARY_PATH))


def _rotate_event_log_if_needed(path: Path) -> None:
    """Rotate the JSONL event log if it exceeds the size threshold."""
    try:
        if path.exists() and path.stat().st_size >= _EVENT_LOG_MAX_BYTES:
            stamp = _now().strftime("%Y%m%dT%H%M%S")
            archive_name = path.stem + f".{stamp}" + path.suffix
            archive_path = path.parent / archive_name
            path.rename(archive_path)
    except OSError:
        pass  # best effort — don't crash the bot for log rotation


def log_event(event_type: str, message: str, payload: dict[str, Any] | None = None, severity: str = "info") -> None:
    event = {
        "timestamp": _now().isoformat(),
        "type": event_type,
        "severity": severity,
        "message": message,
        "payload": payload or {},
    }
    path = event_path()
    _rotate_event_log_if_needed(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def log_signal(
    *,
    symbol: str,
    strategy: str,
    side: str,
    action: str,
    status: str,
    price: float | None,
    bar_time: datetime | None,
    timeframe: str,
    reason: str = "",
) -> None:
    path = signals_path()
    _ensure_header(path, SIGNAL_FIELDS)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                _now().isoformat(),
                symbol,
                strategy,
                side,
                action,
                status,
                "" if price is None else round(float(price), 4),
                "" if bar_time is None else bar_time.isoformat(),
                timeframe,
                reason,
            ]
        )


def log_guardrail(
    *,
    rule_name: str,
    status: str,
    detail: str,
    symbol: str = "",
    strategy: str = "",
    metrics: dict[str, Any] | None = None,
) -> None:
    path = guardrails_path()
    _ensure_header(path, GUARDRAIL_FIELDS)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                _now().isoformat(),
                rule_name,
                symbol,
                strategy,
                status,
                detail,
                json.dumps(metrics or {}, sort_keys=True),
            ]
        )


def save_latest_backtest_summary(summary: dict[str, Any]) -> Path:
    path = latest_backtest_summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# Session performance report
# ---------------------------------------------------------------------------

def generate_session_performance_report(
    trades_path: str | None = None,
    equity_path: str | None = None,
) -> dict[str, Any]:
    """
    Reads ``trades.csv`` and ``equity_curve.csv`` and computes a set of
    paper-trading performance metrics.  Returns a dict suitable for JSON
    serialization or dashboard rendering.

    Metrics:
    - total_trades, winners, losers, win_rate
    - avg_win, avg_loss, expectancy, profit_factor
    - max_drawdown_pct (from equity curve)
    - sharpe_ratio (annualized, from daily returns if enough data)
    - breakdown by strategy (native vs adopted)
    """
    trades_path = trades_path or config.TRADES_LOG_PATH
    equity_path = equity_path or config.EQUITY_CURVE_LOG_PATH

    # -- Parse trades -------------------------------------------------------
    trades: list[dict] = []
    if os.path.exists(trades_path):
        with open(trades_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    trades.append({
                        "instrument": row.get("instrument", ""),
                        "strategy": row.get("strategy", ""),
                        "pnl": float(row.get("profit_loss", 0)),
                        "direction": row.get("direction", ""),
                        "reason": row.get("reason", ""),
                    })
                except (ValueError, KeyError):
                    continue

    total = len(trades)
    if total == 0:
        return {"total_trades": 0, "message": "No trades to analyze."}

    pnls = [t["pnl"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    avg_win = sum(winners) / len(winners) if winners else 0.0
    avg_loss = sum(losers) / len(losers) if losers else 0.0
    win_rate = len(winners) / total if total else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if total else 0.0
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # -- Per-strategy breakdown ---------------------------------------------
    strategy_stats: dict[str, dict] = {}
    for t in trades:
        strat = t["strategy"] or "unknown"
        bucket = strategy_stats.setdefault(strat, {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0})
        bucket["trades"] += 1
        bucket["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            bucket["wins"] += 1
        elif t["pnl"] < 0:
            bucket["losses"] += 1
    for strat, stats in strategy_stats.items():
        stats["win_rate"] = round(stats["wins"] / stats["trades"], 4) if stats["trades"] else 0.0
        stats["pnl"] = round(stats["pnl"], 2)

    # -- Max drawdown from equity curve ------------------------------------
    max_dd_pct = 0.0
    sharpe = None
    if os.path.exists(equity_path):
        equities: list[float] = []
        with open(equity_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    equities.append(float(row["equity"]))
                except (ValueError, KeyError):
                    continue

        if len(equities) >= 2:
            peak = equities[0]
            max_dd = 0.0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
            max_dd_pct = round(max_dd * 100, 4)

            # Sharpe from equity changes (annualized, assuming 252 trading days).
            returns = [
                (equities[i] - equities[i - 1]) / equities[i - 1]
                for i in range(1, len(equities))
                if equities[i - 1] > 0
            ]
            if len(returns) >= 5:
                mean_r = sum(returns) / len(returns)
                var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                std_r = math.sqrt(var_r) if var_r > 0 else 0.0
                if std_r > 0:
                    sharpe = round((mean_r / std_r) * math.sqrt(252), 4)

    report = {
        "generated_at": _now().isoformat(),
        "total_trades": total,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "total_pnl": round(sum(pnls), 2),
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "strategy_breakdown": strategy_stats,
    }

    # Persist to runtime/reports/
    reports_dir = Path(config.RUNTIME_DIR) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"performance_{_now().strftime('%Y%m%d')}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)

    return report
