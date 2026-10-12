"""Position Manager service — single source of truth for all positions.

Responsibilities:
1. Track open positions (symbol, side, qty, entry_price, stop_price, strategy)
2. Process position lifecycle events from any service (strategy workers, risk engine)
3. Publish position.{symbol} events when positions change
4. Maintain the runtime state (wins, losses, P&L, consecutive_losses)
5. Reconcile with broker periodically

Replaces the monolithic Portfolio class in the v2 bot. In the distributed
architecture, this is the ONLY service that can modify position state.
All other services send commands via the message bus.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from bot.distributed.base import BaseService
from bot.portfolio import OpenPosition, Portfolio
from bot.state_store import RuntimeStateStore
import config

logger = logging.getLogger("aegis.position_manager")


class PositionManager(BaseService):
    name = "position_manager"
    heartbeat_interval = 5

    def __init__(self, bus=None):
        super().__init__(bus)
        self.portfolio = Portfolio(broker=None)  # broker managed separately
        self.state_store = RuntimeStateStore()
        # Load persisted state
        state = self.state_store.load()
        if state and state.open_positions:
            for symbol, pos_dict in state.open_positions.items():
                try:
                    self.portfolio.positions[symbol] = OpenPosition.from_dict(pos_dict)
                except Exception:
                    logger.warning("Could not restore position for %s", symbol)
        self.wins = getattr(state, "wins", 0) if state else 0
        self.losses = getattr(state, "losses", 0) if state else 0
        self.consecutive_losses = getattr(state, "consecutive_losses", 0) if state else 0
        self.cumulative_pnl = getattr(state, "cumulative_realized_pnl", 0.0) if state else 0.0

    def register_handlers(self) -> None:
        self.bus.subscribe("position.open", self._on_open_request)
        self.bus.subscribe("position.close", self._on_close_request)
        self.bus.subscribe("position.update_stop", self._on_update_stop)
        self.bus.subscribe("position.close_all", self._on_close_all)

    def on_start(self) -> None:
        # Publish all current positions so strategy workers know the state
        for symbol, pos in self.portfolio.positions.items():
            self._publish_position_event(symbol, pos, "restored")
        logger.info("Position Manager started with %d open positions", len(self.portfolio.positions))

    def on_tick(self) -> None:
        # Persist state every tick
        self._persist_state()

    def on_stop(self) -> None:
        self._persist_state()
        logger.info("Position Manager stopped. Final P&L: %.2f", self.cumulative_pnl)

    # ------------------------------------------------------------------
    # Event handlers (commands sent from other services)
    # ------------------------------------------------------------------
    def _on_open_request(self, channel: str, msg: dict) -> None:
        """Open a new position. msg: {symbol, side, qty, entry_price, strategy, stop_price, atr_at_entry, risk_amount, notional}"""
        symbol = msg["symbol"]
        if self.portfolio.has_position(symbol):
            logger.warning("Position already open for %s, ignoring", symbol)
            return
        pos = OpenPosition(
            symbol=symbol,
            side=msg["side"],
            qty=msg["qty"],
            entry_price=msg["entry_price"],
            entry_time=datetime.now(timezone.utc),
            atr_at_entry=msg.get("atr_at_entry", 0.0),
            stop_price=msg["stop_price"],
            extreme_price=msg["entry_price"],
            strategy=msg.get("strategy", ""),
            risk_amount=msg.get("risk_amount", 0.0),
            notional=msg.get("notional", 0.0),
            asset_class=msg.get("asset_class", ""),
        )
        self.portfolio.open_position(pos)
        self._publish_position_event(symbol, pos, "opened")

    def _on_close_request(self, channel: str, msg: dict) -> None:
        """Close a position. msg: {symbol, exit_price, reason, strategy}"""
        symbol = msg["symbol"]
        reason = msg.get("reason", "manual_close")
        exit_price = float(msg["exit_price"])
        pos = self.portfolio.get(symbol)
        if pos is None:
            logger.warning("No position to close for %s", symbol)
            return
        result = self.portfolio.close_position(symbol, exit_price, reason)
        if result:
            pnl = result["pnl"]
            self._record_trade_outcome(pos, pnl, reason)
            self._publish_trade_event(symbol, pos, pnl, reason, msg.get("approximate", False))

    def _on_update_stop(self, channel: str, msg: dict) -> None:
        """Update trailing stop for a position. msg: {symbol, new_stop_price}"""
        symbol = msg["symbol"]
        new_stop = float(msg["new_stop_price"])
        pos = self.portfolio.get(symbol)
        if pos:
            pos.stop_price = new_stop
            self._publish_position_event(symbol, pos, "stop_updated")

    def _on_close_all(self, channel: str, msg: dict) -> None:
        """Emergency close all positions. msg: {reason}"""
        reason = msg.get("reason", "emergency_flatten")
        for symbol in list(self.portfolio.positions.keys()):
            pos = self.portfolio.get(symbol)
            self.bus.publish("broker.close_position", {
                "symbol": symbol,
                "reason": reason,
            })
        logger.warning("Emergency close-all triggered: %s", reason)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _record_trade_outcome(self, pos: OpenPosition, pnl: float, reason: str) -> None:
        self.cumulative_pnl += pnl
        if pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.losses += 1
            self.consecutive_losses += 1
            self.bus.publish("position.loss_streak", {
                "consecutive_losses": self.consecutive_losses,
                "max_allowed": config.MAX_CONSECUTIVE_LOSSES,
                "halt_triggered": self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES,
                "symbol": pos.symbol,
                "pnl": pnl,
            })

    def _publish_position_event(self, symbol: str, pos: OpenPosition, event: str) -> None:
        self.bus.publish(f"position.{symbol}", {
            "event": event,
            "symbol": symbol,
            "side": pos.side,
            "qty": pos.qty,
            "entry_price": pos.entry_price,
            "stop_price": pos.stop_price,
            "strategy": pos.strategy,
            "risk_amount": pos.risk_amount,
            "notional": pos.notional,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _publish_trade_event(self, symbol: str, pos: OpenPosition, pnl: float, reason: str, approximate: bool) -> None:
        self.bus.publish("position.trade_closed", {
            "symbol": symbol,
            "side": pos.side,
            "pnl": pnl,
            "reason": reason,
            "strategy": pos.strategy,
            "entry_price": pos.entry_price,
            "exit_price": pos.stop_price if "hard_stop" in reason else None,
            "approximate": approximate,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _persist_state(self) -> None:
        from bot.runtime_state import RuntimeState
        if hasattr(self, "_last_equity"):
            state = RuntimeState(
                mode="PAPER" if config.ALPACA_PAPER else "LIVE",
                started_at=datetime.now(timezone.utc),
                runtime_status="running",
                last_equity=self._last_equity,
                wins=self.wins,
                losses=self.losses,
                consecutive_losses=self.consecutive_losses,
                cumulative_realized_pnl=self.cumulative_pnl,
                open_positions=self.portfolio.to_dict(),
                last_processed_bar={},
            )
            self.state_store.save(state)

    def set_equity(self, equity: float) -> None:
        """Called by the broker service to update current equity."""
        self._last_equity = equity


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    PositionManager().run()
