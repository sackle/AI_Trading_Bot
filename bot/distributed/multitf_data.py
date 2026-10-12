"""Multi-Timeframe Data Service — fetches and caches bars at multiple resolutions.

In Phase 2, the bot needs bars at 1-minute, 5-minute, 15-minute, 1-hour,
and 4-hour timeframes to enable:
1. Microstructure indicators (VWAP, order flow on 1min bars)
2. Multi-scale confirmation (3/4 timeframe confluence)
3. Liquidity-aware sizing (use 5min volume profile)

This service sits between the broker and the strategy workers,
pre-aggregating bars at each desired resolution and publishing them
as bars.{symbol}.{resolution} events.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from bot.distributed.base import BaseService
from bot.distributed.utils import load_symbol_list
import config

logger = logging.getLogger("aegis.multitf")

# Timeframes to publish (amount, unit, label)
TIMEFRAMES: list[tuple[int, str, str]] = [
    (1, "Minute", "1min"),
    (5, "Minute", "5min"),
    (15, "Minute", "15min"),
    (1, "Hour", "1h"),
    (4, "Hour", "4h"),
]


class MultiTimeframeDataService(BaseService):
    """Fetches bars at multiple timeframes and publishes them.

    Publishes: bars.{symbol}.{resolution}
    Example:  bars.SPY.1min, bars.SPY.15min, bars.BTC.USD.1h

    This service runs on a configurable interval (default: every 30s)
    to fetch fresh bars from the broker at each resolution.
    """

    name = "multitf_data"
    heartbeat_interval = 30  # fetch new bars every 30s

    def __init__(self, bus=None, broker: Any = None):
        super().__init__(bus)
        self.broker = broker
        self.symbols: list[str] = []
        self._last_fetch: dict[str, str] = {}  # symbol:resolution -> last timestamp

    def register_handlers(self) -> None:
        self.bus.subscribe("multitf.subscribe", self._on_subscribe)

    def on_start(self) -> None:
        self.symbols = list(config.ALL_EQUITY_SYMBOLS) + list(config.ALL_CRYPTO_SYMBOLS)
        logger.info("Multi-TF Data Service started with %d symbols, %d timeframes",
                     len(self.symbols), len(TIMEFRAMES))

    def on_tick(self) -> None:
        """Fetch bars at all timeframes for all symbols."""
        if self.broker is None:
            from bot.brokers import create_broker
            self.broker = create_broker()

        bars_fetched = 0
        for symbol in self.symbols[:20]:  # limit to top 20 for now
            for tf_amount, tf_unit, tf_label in TIMEFRAMES:
                key = f"{symbol}.{tf_label}"
                try:
                    bars = self.broker.get_bars(symbol, tf_amount, tf_unit, limit=10)
                    if bars.empty:
                        continue

                    latest = bars.iloc[-1]
                    ts = pd.Timestamp(bars.index[-1]).isoformat()

                    # Only publish if this is a new bar
                    if self._last_fetch.get(key) == ts:
                        continue
                    self._last_fetch[key] = ts

                    bar_event = {
                        "symbol": symbol,
                        "timeframe": tf_label,
                        "tf_amount": tf_amount,
                        "tf_unit": tf_unit,
                        "open": float(latest["open"]),
                        "high": float(latest["high"]),
                        "low": float(latest["low"]),
                        "close": float(latest["close"]),
                        "volume": float(latest["volume"]),
                        "timestamp": ts,
                    }
                    self.bus.publish(f"bars.{symbol}.{tf_label}", bar_event)
                    bars_fetched += 1
                except Exception as exc:
                    logger.debug("Failed %s %s: %s", symbol, tf_label, exc)

        if bars_fetched > 0:
            logger.debug("Fetched %d bars across all symbols/timeframes", bars_fetched)

    def on_stop(self) -> None:
        logger.info("Multi-TF Data Service stopped")

    def _on_subscribe(self, channel: str, msg: dict) -> None:
        symbol = msg.get("symbol", "")
        if symbol and symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info("Subscribed to %s on Multi-TF Data Service", symbol)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    MultiTimeframeDataService().run()
