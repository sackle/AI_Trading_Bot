"""Persistent paper execution ledger for NSE strategies.

The broker is the source of truth for the NSE paper account. Earlier versions
kept one transient position dictionary in the broker and another in the
trading loop, while reporting a constant account balance. That made a restart
lose positions and made every sizing decision use the original capital. This
implementation persists cash, positions, order metadata, and last observed
prices atomically after every state-changing operation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import config
from bot.portfolio import OpenPosition

logger = logging.getLogger("aegis.nse.broker")
IST = ZoneInfo("Asia/Kolkata")

NSE_TRADES_LOG = os.path.join(config.ROOT_DIR, "runtime", "nse_trades.csv")

_EQUITY_SLIPPAGE = 0.0005  # 0.05%
_FNO_SLIPPAGE = 0.001  # 0.10%
_FNO_LOT_SIZES = {
    "NIFTY": 75,
    "BANKNIFTY": 30,
    "FINNIFTY": 40,
    "RELIANCE": 250,
    "TCS": 150,
    "INFY": 300,
    "HDFCBANK": 550,
    "ICICIBANK": 700,
}


def _is_fno(symbol: str) -> bool:
    return symbol in _FNO_LOT_SIZES or symbol.endswith("-FUT") or "-CE" in symbol or "-PE" in symbol


class NSEPaperBroker:
    """Paper account with restart-safe bookkeeping and conservative fills."""

    def __init__(self, state_path: str | Path | None = None, starting_cash: float | None = None):
        config.ensure_runtime_directories()
        self.state_path = Path(state_path or config.runtime_path(config.NSE_STATE_PATH))
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._starting_cash = float(
            config.NSE_PAPER_EQUITY if starting_cash is None else starting_cash
        )
        self._state = self._load_state()
        self._ensure_log_header()

    def _blank_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "cash": self._starting_cash,
            "positions": {},
            "orders": {},
            "latest_prices": {},
            "order_counter": 0,
            "current_date": datetime.now(IST).date().isoformat(),
            "day_start_equity": self._starting_cash,
            "daily_realized_pnl": 0.0,
            "cumulative_realized_pnl": 0.0,
            "consecutive_losses": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._blank_state()
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("state is not an object")
            state = self._blank_state()
            state.update(loaded)
            for field, default in (("positions", {}), ("orders", {}), ("latest_prices", {})):
                if not isinstance(state.get(field), dict):
                    state[field] = default
            state["cash"] = float(state.get("cash", self._starting_cash))
            state["order_counter"] = int(state.get("order_counter", 0))
            state["day_start_equity"] = float(state.get("day_start_equity", state["cash"]))
            state["daily_realized_pnl"] = float(state.get("daily_realized_pnl", 0.0))
            state["cumulative_realized_pnl"] = float(state.get("cumulative_realized_pnl", 0.0))
            state["consecutive_losses"] = int(state.get("consecutive_losses", 0))
            for order in state["orders"].values():
                if order.get("status") != "accepted":
                    continue
                symbol = order.get("symbol")
                # Accepted orders are only committed when register_position
                # atomically writes the fill and the position. Anything left
                # accepted after a restart was interrupted before commit.
                if symbol not in state["positions"]:
                    order["status"] = "canceled"
            return state
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup = self.state_path.with_suffix(f".corrupt.{stamp}.json")
            try:
                self.state_path.replace(backup)
            except OSError:
                pass
            logger.warning("NSE paper state was unreadable (%s); starting a fresh ledger.", exc)
            return self._blank_state()

    def _save_locked(self) -> None:
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _next_order_id_locked(self, prefix: str) -> str:
        self._state["order_counter"] += 1
        return f"NSE-{prefix}-{self._state['order_counter']:06d}"

    @property
    def positions(self) -> dict[str, OpenPosition]:
        """A copy of canonical positions for compatibility with callers."""
        return {
            symbol: OpenPosition.from_dict(payload)
            for symbol, payload in self._state["positions"].items()
        }

    def get_open_position_models(self) -> list[OpenPosition]:
        with self._lock:
            return list(self.positions.values())

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------
    def submit_market_order(self, symbol: str, qty: float, side: str) -> str:
        qty = float(qty)
        side = side.lower()
        if qty <= 0:
            raise ValueError("Order quantity must be positive.")
        if side not in {"long", "short"}:
            raise ValueError("Order side must be 'long' or 'short'.")
        with self._lock:
            if symbol in self._state["positions"]:
                raise RuntimeError(f"An NSE paper position is already open for {symbol}.")
            order_id = self._next_order_id_locked("PAPER")
            self._state["orders"][order_id] = {
                "id": order_id,
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "status": "accepted",
            }
            self._save_locked()
        logger.info("[NSE Paper] %s %s x%.0f order_id=%s", side.upper(), symbol, qty, order_id)
        return order_id

    def get_fill_price(self, order_id: str, fallback: float, side: str, symbol: str) -> float:
        if fallback <= 0:
            raise ValueError("A positive market price is required for an NSE paper fill.")
        with self._lock:
            order = self._state["orders"].get(order_id)
            if order is None:
                raise RuntimeError(f"Unknown NSE paper order: {order_id}")
            if order.get("filled_avg_price") is not None:
                return float(order["filled_avg_price"])
            slip = _FNO_SLIPPAGE if _is_fno(symbol) else _EQUITY_SLIPPAGE
            fill_price = round(float(fallback) * (1 + slip if side == "long" else 1 - slip), 2)
            order["proposed_fill_price"] = fill_price
            return fill_price

    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        if float(qty) <= 0 or float(stop_price) <= 0:
            raise ValueError("Stop quantity and price must be positive.")
        with self._lock:
            order_id = self._next_order_id_locked("STOP")
            self._state["orders"][order_id] = {
                "id": order_id,
                "symbol": symbol,
                "qty": float(qty),
                "position_side": side,
                "stop_price": float(stop_price),
                "status": "accepted",
            }
            self._save_locked()
        return order_id

    def cancel_order(self, order_id: str) -> None:
        if not order_id:
            return
        with self._lock:
            order = self._state["orders"].get(order_id)
            if order and order.get("status") == "accepted":
                order["status"] = "canceled"
                self._save_locked()

    def get_order_snapshot(self, order_id: str) -> dict | None:
        with self._lock:
            order = self._state["orders"].get(order_id)
            return dict(order) if order is not None else None

    # ------------------------------------------------------------------
    # Canonical position and account ledger
    # ------------------------------------------------------------------
    def register_position(self, position: OpenPosition) -> None:
        """Commit a filled position and cash movement as one durable action."""
        self.ensure_session()
        with self._lock:
            if position.symbol in self._state["positions"]:
                raise RuntimeError(f"An NSE paper position is already open for {position.symbol}.")
            notional = abs(float(position.entry_price) * float(position.qty))
            if position.side == "long" and self._state["cash"] + 1e-9 < notional:
                raise RuntimeError("Insufficient NSE paper cash for this position.")
            if position.entry_order_id:
                order = self._state["orders"].get(position.entry_order_id)
                if order is None or order.get("symbol") != position.symbol:
                    raise RuntimeError("Cannot commit NSE position without its matching entry order.")
                order["status"] = "filled"
                order["filled_avg_price"] = float(position.entry_price)
            self._state["cash"] += -notional if position.side == "long" else notional
            self._state["positions"][position.symbol] = position.to_dict()
            self._state["latest_prices"][position.symbol] = float(position.entry_price)
            self._save_locked()

    def update_position(self, position: OpenPosition) -> None:
        with self._lock:
            if position.symbol not in self._state["positions"]:
                raise RuntimeError(f"Cannot update unknown NSE position {position.symbol}.")
            self._state["positions"][position.symbol] = position.to_dict()
            self._save_locked()

    def close_position(self, symbol: str, exit_price: float) -> tuple[OpenPosition, float] | None:
        """Atomically remove a position, settle cash, and return realized P&L."""
        self.ensure_session()
        with self._lock:
            payload = self._state["positions"].get(symbol)
            if payload is None:
                return None
            position = OpenPosition.from_dict(payload)
            exit_price = float(exit_price)
            if exit_price <= 0:
                raise ValueError("Exit price must be positive.")
            value = exit_price * position.qty
            self._state["cash"] += value if position.side == "long" else -value
            self._state["positions"].pop(symbol, None)
            self._state["latest_prices"][symbol] = exit_price
            direction = 1.0 if position.side == "long" else -1.0
            pnl = (exit_price - position.entry_price) * direction * position.qty
            self._state["daily_realized_pnl"] += pnl
            self._state["cumulative_realized_pnl"] += pnl
            if pnl < 0:
                self._state["consecutive_losses"] += 1
            elif pnl > 0:
                self._state["consecutive_losses"] = 0
            if position.stop_order_id:
                order = self._state["orders"].get(position.stop_order_id)
                if order and order.get("status") == "accepted":
                    order["status"] = "canceled"
            self._save_locked()
        return position, float(pnl)

    def close_position_market(self, symbol: str) -> None:
        with self._lock:
            payload = self._state["positions"].get(symbol)
            if payload is None:
                return
            price = self._state["latest_prices"].get(symbol, payload["entry_price"])
        self.close_position(symbol, float(price))

    def mark_price(self, symbol: str, price: float) -> None:
        if price <= 0:
            return
        with self._lock:
            self._state["latest_prices"][symbol] = float(price)
            # Persist only prices that value an open position.  Persisting
            # every scanned symbol would turn one NSE tick into dozens of
            # unnecessary JSON replacements without improving recovery.
            if symbol in self._state["positions"]:
                self._save_locked()

    def get_latest_price(self, symbol: str) -> float | None:
        with self._lock:
            price = self._state["latest_prices"].get(symbol)
            return None if price is None else float(price)

    def get_equity(self) -> float:
        with self._lock:
            equity = float(self._state["cash"])
            for symbol, payload in self._state["positions"].items():
                position = OpenPosition.from_dict(payload)
                price = float(self._state["latest_prices"].get(symbol, position.entry_price))
                equity += price * position.qty if position.side == "long" else -price * position.qty
            return float(equity)

    def ensure_session(self) -> None:
        """Reset daily entry controls at the NSE local calendar boundary."""
        current_date = datetime.now(IST).date().isoformat()
        with self._lock:
            if self._state.get("current_date") == current_date:
                return
            self._state["current_date"] = current_date
            self._state["day_start_equity"] = self.get_equity()
            self._state["daily_realized_pnl"] = 0.0
            self._state["consecutive_losses"] = 0
            self._save_locked()

    def account_risk_state(self) -> dict[str, float | int | str]:
        self.ensure_session()
        with self._lock:
            return {
                "current_date": str(self._state["current_date"]),
                "day_start_equity": float(self._state["day_start_equity"]),
                "daily_realized_pnl": float(self._state["daily_realized_pnl"]),
                "cumulative_realized_pnl": float(self._state["cumulative_realized_pnl"]),
                "consecutive_losses": int(self._state["consecutive_losses"]),
            }

    def get_account_info(self) -> dict:
        with self._lock:
            cash = float(self._state["cash"])
        return {
            "equity": self.get_equity(),
            "cash": cash,
            "buying_power": cash,
            "currency": "INR",
            "pattern_day_trader": False,
            "daytrade_count": 0,
        }

    def get_positions(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "symbol": position.symbol,
                    "side": position.side,
                    "qty": position.qty,
                    "avg_entry_price": position.entry_price,
                }
                for position in self.positions.values()
            ]

    def is_market_open(self, symbol: str = "") -> bool:
        from bot.nse.data_client import NSEDataClient

        return NSEDataClient.is_market_open()

    # ------------------------------------------------------------------
    # Trade audit log
    # ------------------------------------------------------------------
    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        strategy: str,
        reason: str,
        instrument_type: str = "equity",
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_loss": round(pnl, 2),
            "strategy": strategy,
            "reason": reason,
            "instrument_type": instrument_type,
        }
        file_exists = Path(NSE_TRADES_LOG).exists()
        with open(NSE_TRADES_LOG, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _ensure_log_header(self) -> None:
        if not Path(NSE_TRADES_LOG).exists():
            with open(NSE_TRADES_LOG, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [
                        "timestamp",
                        "symbol",
                        "side",
                        "qty",
                        "entry_price",
                        "exit_price",
                        "profit_loss",
                        "strategy",
                        "reason",
                        "instrument_type",
                    ]
                )
