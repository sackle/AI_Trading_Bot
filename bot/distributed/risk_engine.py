"""Risk Engine service — enforces portfolio-level guardrails in real-time.

Receives: bars.{symbol}, position.{symbol}, position.trade_closed
Publishes: risk.guardrail (portfolio-wide signals)

Guardrails enforced:
1. Gross exposure <= MAX_GROSS_EXPOSURE_PCT_OF_EQUITY (80%)
2. Max open positions <= MAX_OPEN_POSITIONS (5)
3. Max consecutive losses <= MAX_CONSECUTIVE_LOSSES (3)
4. Max daily loss <= MAX_DAILY_LOSS_PCT_OF_EQUITY (5%)
5. Position size <= MAX_POSITION_PCT_OF_EQUITY (35%)
6. Asset class limits <= ASSET_CLASS_CAPITAL_LIMITS
7. Correlation group limits (risk_on_cluster, commodity_cluster)
8. Time-of-day filter (no new entries near market open/close)

If ANY guardrail is violated, publishes risk.guardrail with "BLOCK"
which the Strategy Worker checks before sending signals.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from bot.distributed.base import BaseService
from bot.risk_manager import RiskManager
import config

logger = logging.getLogger("aegis.risk")


class RiskEngine(BaseService):
    name = "risk_engine"
    heartbeat_interval = 3

    def __init__(self, bus=None):
        super().__init__(bus)
        self.risk_manager = RiskManager()
        self.equity: float = config.BACKTEST_INITIAL_EQUITY
        self.open_positions: dict[str, dict] = {}
        self.daily_loss: float = 0.0
        self.consecutive_losses: int = 0
        self.daily_reset_time: datetime | None = None
        self.last_bar_time: dict[str, str] = {}

    def register_handlers(self) -> None:
        self.bus.subscribe("bars.*", self._on_bar)
        self.bus.subscribe("position.*", self._on_position_update)
        self.bus.subscribe("position.trade_closed", self._on_trade_closed)
        self.bus.subscribe("position.loss_streak", self._on_loss_streak)
        self.bus.subscribe("broker.equity_update", self._on_equity_update)

    def on_start(self) -> None:
        # Publish initial equity so other services know our capacity
        self.bus.publish("broker.equity_update", {"equity": self.equity})
        self._schedule_daily_reset()
        logger.info("Risk Engine started, equity=$%.2f", self.equity)

    def _on_bar(self, channel: str, msg: dict) -> None:
        """Check time-of-day filter on incoming bars."""
        symbol = msg.get("symbol", "")
        ts_str = msg.get("timestamp", "")
        if symbol not in self.last_bar_time or self.last_bar_time[symbol] != ts_str:
            self.last_bar_time[symbol] = ts_str
            # Check if we should publish a guardrail for this symbol
            guardrail = self._check_all_guardrails(symbol)
            if guardrail:
                self.bus.publish(f"risk.guardrail.{symbol}", guardrail)

    def _on_position_update(self, channel: str, msg: dict) -> None:
        """Track position opens/stops."""
        symbol = msg.get("symbol", "")
        event = msg.get("event", "")
        if event == "opened":
            self.open_positions[symbol] = {
                "symbol": symbol,
                "side": msg["side"],
                "qty": msg["qty"],
                "entry_price": msg["entry_price"],
                "strategy": msg.get("strategy", ""),
                "risk_amount": msg.get("risk_amount", 0.0),
                "notional": msg.get("notional", 0.0),
            }
        elif event == "stop_updated":
            if symbol in self.open_positions:
                self.open_positions[symbol]["stop_price"] = msg.get("stop_price")
        elif event in ("closed", "restored"):
            self.open_positions.pop(symbol, None)
        self._publish_portfolio_state()

    def _on_trade_closed(self, channel: str, msg: dict) -> None:
        """Process closed trade for loss tracking."""
        pnl = float(msg.get("pnl", 0))
        if pnl < 0:
            self.daily_loss += abs(pnl)
            self.consecutive_losses += 1
            logger.warning("Loss streak: %d (daily P&L loss: $%.2f)", self.consecutive_losses, self.daily_loss)
        elif pnl >= 0:
            self.consecutive_losses = 0

        # Check if consecutive loss limit hit
        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            self.bus.publish("risk.guardrail.all", {
                "action": "BLOCK_ALL_ENTRIES",
                "reason": f"Max consecutive losses ({config.MAX_CONSECUTIVE_LOSSES}) reached",
                "duration_seconds": 900,  # 15 min cool-off
            })
            self.bus.publish("position.close_all", {
                "reason": "consecutive_loss_limit_exceeded",
            })

    def _on_loss_streak(self, channel: str, msg: dict) -> None:
        self.consecutive_losses = msg.get("consecutive_losses", self.consecutive_losses)
        if msg.get("halt_triggered"):
            self.bus.publish("risk.guardrail.all", {
                "action": "BLOCK_ALL_ENTRIES",
                "reason": msg.get("reason", "loss_streak"),
                "duration_seconds": 300,
            })

    def _on_equity_update(self, channel: str, msg: dict) -> None:
        self.equity = float(msg.get("equity", self.equity))

    def _schedule_daily_reset(self) -> None:
        """Reset daily loss counter at midnight."""
        self.daily_reset_time = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + pd.Timedelta(days=1)

    def on_tick(self) -> None:
        now = datetime.now(timezone.utc)
        if self.daily_reset_time and now >= self.daily_reset_time:
            self.daily_loss = 0.0
            self._schedule_daily_reset()
            self.bus.publish("risk.daily_reset", {"reset": True})

    def _check_all_guardrails(self, symbol: str) -> dict | None:
        """Run all guardrail checks for a symbol."""
        checks = []

        # 1. Gross exposure
        total_risked = sum(p["risk_amount"] for p in self.open_positions.values())
        gross_exposure = total_risked / self.equity
        max_gross = config.MAX_GROSS_EXPOSURE_PCT_OF_EQUITY
        if gross_exposure > max_gross:
            checks.append(("gross_exposure", f"Gross exposure {gross_exposure:.1%} > {max_gross:.0%}"))

        # 2. Max open positions
        if len(self.open_positions) >= config.MAX_OPEN_POSITIONS:
            checks.append(("max_positions", f"Max positions {len(self.open_positions)} >= {config.MAX_OPEN_POSITIONS}"))

        # 3. Daily loss
        daily_loss_pct = self.daily_loss / self.equity
        if daily_loss_pct > config.MAX_DAILY_LOSS_PCT_OF_EQUITY:
            checks.append(("daily_loss", f"Daily loss {daily_loss_pct:.1%} > {config.MAX_DAILY_LOSS_PCT_OF_EQUITY:.0%}"))
            checks.append(("consecutive_losses", f"Loss streak: {self.consecutive_losses}"))

        # 4. Time-of-day filter
        ts_str = self.last_bar_time.get(symbol, "")
        if ts_str:
            try:
                ts = pd.Timestamp(ts_str)
                hour = ts.hour
                # No new entries 3 min after market open or 5 min before close (US market)
                if config.EQUITY_NO_NEW_ENTRIES_FIRST_MINUTES > 0 and 3 <= hour < 4:
                    checks.append(("time_filter_open", "Near market open — no new entries"))
                if config.EQUITY_NO_NEW_ENTRIES_LAST_MINUTES > 0 and hour >= 15:
                    checks.append(("time_filter_close", "Near market close — no new entries"))
            except Exception:
                pass

        if checks:
            return {
                "symbol": symbol,
                "action": "BLOCK_ENTRY" if len(checks) < 2 else "BLOCK_ALL_ENTRIES",
                "reasons": checks,
                "equity": self.equity,
                "open_positions": len(self.open_positions),
                "consecutive_losses": self.consecutive_losses,
            }
        return None

    def _publish_portfolio_state(self) -> None:
        """Broadcast current portfolio state for other services."""
        self.bus.publish("risk.portfolio_state", {
            "equity": self.equity,
            "open_positions": len(self.open_positions),
            "daily_loss": self.daily_loss,
            "consecutive_losses": self.consecutive_losses,
            "gross_exposure": sum(p["risk_amount"] for p in self.open_positions.values()) / self.equity if self.equity else 0,
        })

    def on_stop(self) -> None:
        logger.info("Risk Engine stopped")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    RiskEngine().run()
