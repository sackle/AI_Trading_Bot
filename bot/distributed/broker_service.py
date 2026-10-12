"""Broker service — executes trades and publishes equity updates.

This service wraps the Broker class and acts as the ONLY service that
can place orders. All other services send close_position / open_position
events, but this service is the single point of execution.

Receives: broker.* control messages
Publishes: broker.equity_update, broker.position_filled, broker.order_status
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from bot.distributed.base import BaseService
from bot.brokers.base import BaseBroker
from bot.brokers import create_broker
import config

logger = logging.getLogger("aegis.broker")


class BrokerService(BaseService):
    name = "broker_service"
    heartbeat_interval = 10

    def __init__(self, bus=None, broker: BaseBroker | None = None):
        super().__init__(bus)
        self.broker = broker or create_broker()

    def register_handlers(self) -> None:
        self.bus.subscribe("broker.open_position", self._on_open)
        self.bus.subscribe("broker.close_position", self._on_close)
        self.bus.subscribe("broker.get_bars", self._on_get_bars)

    def on_start(self) -> None:
        self._publish_account_state()
        logger.info("Broker Service started")

    def on_tick(self) -> None:
        self._publish_account_state()

    def _publish_account_state(self) -> None:
        try:
            account = self.broker.get_account_info()  # fixed: was get_account() which doesn't exist
            equity = float(account.get("equity", config.BACKTEST_INITIAL_EQUITY))  # fixed: was config.INITIAL_EQUITY
            buying_power = float(account.get("buying_power", 0))
            self.bus.publish("broker.account_update", {
                "equity": equity,
                "buying_power": buying_power,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.debug("Could not fetch account: %s", exc)

    def _on_open(self, channel: str, msg: dict) -> None:
        symbol = msg.get("symbol", "")
        side = msg.get("side", "long")
        qty = msg.get("qty", 0)
        try:
            order_id = self.broker.submit_market_order(symbol, qty, side)  # fixed: was broker.place_order() which doesn't exist
            self.bus.publish("broker.order_status", {
                "symbol": symbol,
                "status": "submitted",
                "order_id": order_id,
                "qty": qty,
                "side": side,
            })
        except Exception as exc:
            logger.error("Order failed for %s: %s", symbol, exc)
            self.bus.publish("broker.order_status", {
                "symbol": symbol,
                "status": "rejected",
                "error": str(exc),
            })

    def _on_close(self, channel: str, msg: dict) -> None:
        symbol = msg.get("symbol", "")
        try:
            self.broker.close_position_market(symbol)  # fixed: was broker.close_position(symbol, qty) which doesn't exist
            self.bus.publish("broker.order_status", {
                "symbol": symbol,
                "status": "submitting_close",
            })
        except Exception as exc:
            logger.error("Close order failed for %s: %s", symbol, exc)

    def _on_get_bars(self, channel: str, msg: dict) -> None:
        """Handle bar-fetch requests from strategy workers."""
        symbol = msg.get("symbol", "")
        timeframe_amount = msg.get("timeframe_amount", 1)
        timeframe_unit = msg.get("timeframe_unit", "Day")
        limit = msg.get("limit", 2000)
        try:
            bars = self.broker.get_bars(symbol, timeframe_amount, timeframe_unit, limit=limit)
            # Publish bars back to the originating service
            bar_list = bars.tail(limit).to_dict("records")
            self.bus.publish(f"bars.data.{symbol}", {
                "symbol": symbol,
                "bars": bar_list,
                "timeframe": f"{timeframe_amount} {timeframe_unit}",
            })
        except Exception as exc:
            logger.error("Bar fetch failed for %s: %s", symbol, exc)

    def on_stop(self) -> None:
        logger.info("Broker Service stopped")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    BrokerService().run()
