"""Market Data Feeder service — streams real-time bars via the broker.

Responsibilities:
1. Poll Alpaca for new bars on each symbol at the configured timeframe
2. Publish bars.{symbol} events to the message bus
3. Handle new/removed symbols dynamically (subscribe/unsubscribe)
4. Track the last processed bar timestamp per symbol (deduplication)

Each bar event: {symbol, timeframe, open, high, low, close, volume, timestamp}
Services subscribe to bars.* and react (strategy workers, analyst, etc.)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from bot.distributed.base import BaseService
from bot.brokers.base import BaseBroker
from bot.brokers import create_broker
import config

logger = logging.getLogger("aegis.feeder")


class MarketDataFeeder(BaseService):
    name = "market_data_feeder"
    heartbeat_interval = 30  # publish market data every 3s (faster than main loop)

    def __init__(self, bus=None, symbols: list[str] | None = None, broker: BaseBroker | None = None):
        super().__init__(bus)
        self.broker = broker or create_broker()
        # Load ALL configured symbols (equity + crypto) — deduplicated, order-preserved.
        if symbols:
            self.symbols = list(symbols)
        else:
            seen: set[str] = set()
            self.symbols = []
            for sym in (
                list(config.MEAN_REVERSION_SYMBOLS)
                + list(config.MOMENTUM_SYMBOLS)
                + list(config.TREND_SYMBOLS)
                + list(config.ALL_CRYPTO_SYMBOLS)
            ):
                if sym and sym not in seen:
                    seen.add(sym)
                    self.symbols.append(sym)
        self.last_bar_ts: dict[str, datetime] = {}
        # Small delay between sequential bar fetches to avoid Alpaca rate limits
        self._fetch_delay = 0.25  # seconds

    def register_handlers(self) -> None:
        self.bus.subscribe("feeder.subscribe", self._on_subscribe)
        self.bus.subscribe("feeder.unsubscribe", self._on_unsubscribe)

    def on_start(self) -> None:
        logger.info("Market Data Feeder starting with %d symbols", len(self.symbols))

    def on_tick(self) -> None:
        """Fetch latest bars for all symbols and publish them."""
        for symbol in self.symbols:
            try:
                strategy_name = config.symbol_strategy(symbol)
                if strategy_name == "mean_reversion":
                    tf_amount, tf_unit = config.MEAN_REVERSION_TIMEFRAME_AMOUNT, config.MEAN_REVERSION_TIMEFRAME_UNIT
                elif strategy_name == "momentum_breakout":
                    tf_amount, tf_unit = config.MOMENTUM_TIMEFRAME_AMOUNT, config.MOMENTUM_TIMEFRAME_UNIT
                elif strategy_name == "trend_following":
                    tf_amount, tf_unit = config.TREND_TIMEFRAME_AMOUNT, config.TREND_TIMEFRAME_UNIT
                else:
                    # Default for unknown strategies (e.g. screener-added symbols)
                    tf_amount, tf_unit = 5, "Minute"

                bars = self.broker.get_bars(symbol, tf_amount, tf_unit, limit=config.BAR_FETCH_LIMIT)
                if bars.empty:
                    time.sleep(self._fetch_delay)
                    continue

                latest = bars.iloc[-1]
                ts = pd.Timestamp(bars.index[-1])
                is_first_fetch = symbol not in self.last_bar_ts
                # Deduplicate: only publish if this is a new bar
                if not is_first_fetch and ts <= self.last_bar_ts[symbol]:
                    time.sleep(self._fetch_delay)
                    continue
                self.last_bar_ts[symbol] = ts

                # On first fetch: publish full bar history so strategy workers warm up immediately
                if is_first_fetch:
                    bars_reset = bars.reset_index()
                    ts_col = bars_reset.columns[0]  # typically 'timestamp'
                    history_rows = []
                    for _, row in bars.iterrows():
                        history_rows.append({
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                            "timestamp": pd.Timestamp(row.name).isoformat(),
                        })
                    self.bus.publish(f"bars.history.{symbol}", {
                        "symbol": symbol,
                        "bars": history_rows,
                        "count": len(history_rows),
                    })
                    logger.info("Published %d historical bars for %s", len(history_rows), symbol)

                # Build latest bar event (always published for live tracking)
                bar_event = {
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "timeframe": f"{tf_amount} {tf_unit}",
                    "open": float(latest["open"]),
                    "high": float(latest["high"]),
                    "low": float(latest["low"]),
                    "close": float(latest["close"]),
                    "volume": float(latest["volume"]),
                    "timestamp": ts.isoformat(),
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }
                self.bus.publish(f"bars.{symbol}", bar_event)

            except Exception as exc:
                logger.warning("Error fetching bars for %s: %s", symbol, exc)

            # Rate-limit: avoid Alpaca "too many requests" errors
            time.sleep(self._fetch_delay)

    def _on_subscribe(self, channel: str, msg: dict) -> None:
        symbol = msg.get("symbol")
        if symbol and symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info("Subscribed to %s", symbol)

    def _on_unsubscribe(self, channel: str, msg: dict) -> None:
        symbol = msg.get("symbol")
        if symbol and symbol in self.symbols:
            self.symbols.remove(symbol)
            logger.info("Unsubscribed from %s", symbol)

    def on_stop(self) -> None:
        logger.info("Market Data Feeder stopped.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    MarketDataFeeder().run()
