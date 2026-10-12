"""
CSV trade + daily P&L logging.

trades.csv     -- one row per closed trade (timestamp, instrument,
                   direction, entry price, exit price, profit/loss,
                   position size).
daily_pnl.csv  -- one row per calendar day (UTC), updated throughout the
                   day as equity moves and finalized by the last update
                   before the date rolls over.
"""

import csv
import os
from datetime import datetime

import config

TRADE_FIELDS = [
    "timestamp",
    "instrument",
    "direction",
    "entry_price",
    "exit_price",
    "profit_loss",
    "position_size",
    "strategy",
    "reason",
]

DAILY_FIELDS = ["date", "starting_equity", "ending_equity", "daily_pnl"]
EQUITY_CURVE_FIELDS = ["timestamp", "equity", "day_start_equity"]


def _ensure_header(path: str, fields: list):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fields)


def log_trade(
    timestamp: datetime,
    instrument: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    position_size: float,
    strategy: str = "",
    reason: str = "",
    path: str = None,
):
    path = path or config.TRADES_LOG_PATH
    _ensure_header(path, TRADE_FIELDS)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                timestamp.isoformat(),
                instrument,
                direction,
                round(entry_price, 4),
                round(exit_price, 4),
                round(pnl, 2),
                position_size,
                strategy,
                reason,
            ]
        )


def log_daily_pnl(date_str: str, starting_equity: float, ending_equity: float, path: str = None):
    """
    Upserts today's row. Called every loop tick with the latest running
    equity, so the row for `date_str` is continuously refreshed and ends
    up holding the true end-of-day figure once the date rolls over.
    """
    path = path or config.DAILY_PNL_LOG_PATH
    _ensure_header(path, DAILY_FIELDS)

    rows = []
    found = False
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["date"] == date_str:
                row["ending_equity"] = f"{ending_equity:.2f}"
                row["daily_pnl"] = f"{ending_equity - float(row['starting_equity']):.2f}"
                found = True
            rows.append(row)

    if found:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    date_str,
                    round(starting_equity, 2),
                    round(ending_equity, 2),
                    round(ending_equity - starting_equity, 2),
                ]
            )


def log_equity_snapshot(timestamp: datetime, equity: float, day_start_equity: float, path: str = None):
    """
    Appends a lightweight runtime equity point so the dashboard can render a
    zoomed intraday portfolio chart instead of only daily close-to-close data.
    """
    path = path or config.runtime_path(config.EQUITY_CURVE_LOG_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _ensure_header(path, EQUITY_CURVE_FIELDS)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                timestamp.isoformat(),
                round(equity, 2),
                round(day_start_equity, 2),
            ]
        )
