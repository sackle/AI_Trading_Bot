"""Strategy Worker service — processes bars and generates trade signals.

Each instance handles one strategy class across multiple symbols.
Stateless (no position state — that lives in Position Manager).

Receives: bars.{symbol} events from Market Data Feeder
Sends:   signal.{symbol} events to Position Manager

Strategy classes supported:
- MeanReversionStrategy (mean-reverting equities)
- MomentumBreakoutStrategy (crypto, growth stocks)
- TrendFollowingStrategy (commodity ETFs, dividend stocks)

Usage:
    python -m bot.distributed.strategy_worker mean_reversion
    python -m bot.distributed.strategy_worker momentum_breakout
    python -m bot.distributed.strategy_worker trend_following
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd

from bot.distributed.base import BaseService
from bot.distributed.utils import load_symbol_list
import config

logger = logging.getLogger("aegis.strategy")


class StrategyWorker(BaseService):
    name = "strategy_worker"
    heartbeat_interval = 3

    def __init__(self, bus=None, strategy_name: str = "mean_reversion"):
        super().__init__(bus)
        self.strategy_name = strategy_name
        self.name = f"strategy_worker.{strategy_name}"
        self.strategies: dict[str, Any] = {}
        # Track recent bars per symbol for strategy evaluation
        self.bar_windows: dict[str, list[dict]] = {}
        self.max_window = 500  # configurable via config.BAR_FETCH_LIMIT

    def register_handlers(self) -> None:
        # Subscribe to bars for symbols in our strategy universe
        self.bus.subscribe("bars.*", self._on_bar)
        # Also subscribe to history bursts so we warm up immediately on startup
        self.bus.subscribe("bars.history.*", self._on_bar_history)

    def on_start(self) -> None:
        symbols = load_symbol_list(self.strategy_name)
        for symbol in symbols:
            self._init_strategy(symbol)
        logger.info(
            "[%s] Started with %d symbols", self.strategy_name, len(self.strategies)
        )

    def _init_strategy(self, symbol: str) -> None:
        """Build a fresh strategy instance for this symbol."""
        if self.strategy_name == "mean_reversion":
            from bot.strategies.mean_reversion import MeanReversionStrategy
            # Get std_multiplier from symbol-specific params or default to 1.5
            params = config.MEAN_REVERSION_PARAMS.get(symbol, {"std_multiplier": 1.5})
            self.strategies[symbol] = MeanReversionStrategy(
                symbol=symbol,
                std_multiplier=params["std_multiplier"],
                lookback=config.MEAN_REVERSION_LOOKBACK,
            )
        elif self.strategy_name == "momentum_breakout":
            from bot.strategies.momentum_breakout import MomentumBreakoutStrategy
            self.strategies[symbol] = MomentumBreakoutStrategy(
                symbol=symbol,
                lookback=config.MOMENTUM_LOOKBACK,
                volume_multiplier=config.MOMENTUM_VOLUME_MULTIPLIER,
            )
        elif self.strategy_name == "trend_following":
            from bot.strategies.trend_following import TrendFollowingStrategy
            self.strategies[symbol] = TrendFollowingStrategy(
                symbol=symbol,
                fast_period=50,
                slow_period=120,
                allow_short=True,
            )
        elif self.strategy_name == "red_green_breakout":
            from bot.strategies.red_green_breakout import RedGreenBreakoutStrategy
            self.strategies[symbol] = RedGreenBreakoutStrategy(symbol=symbol)

    def _on_bar_history(self, channel: str, msg: dict) -> None:
        """Bulk-load historical bars into the window for warm-up."""
        symbol = msg.get("symbol", "")
        if symbol not in self.strategies:
            return  # not in our strategy's universe

        bars = msg.get("bars", [])
        if not bars:
            return

        window = self.bar_windows.setdefault(symbol, [])
        for bar in bars:
            window.append(bar)
        # Trim to max window size
        if len(window) > self.max_window:
            self.bar_windows[symbol] = window[-self.max_window:]
        logger.info("[%s] Warmed up %d bars for %s (total window: %d)",
                    self.strategy_name, len(bars), symbol, len(self.bar_windows[symbol]))

    def _on_bar(self, channel: str, msg: dict) -> None:
        """Process a new bar from the market data feeder."""
        symbol = msg.get("symbol", "")
        if symbol not in self.strategies:
            return  # not in our strategy's universe

        strategy = self.strategies[symbol]

        # Maintain rolling bar window
        window = self.bar_windows.setdefault(symbol, [])
        window.append(msg)
        if len(window) > self.max_window:
            window.pop(0)

        # Need minimum bars before evaluating
        min_bars = getattr(strategy, "min_bars_required", 20)
        if len(window) < min_bars:
            return

        # Convert to DataFrame for strategy
        df = pd.DataFrame(window)

        # Check for existing position via position manager query
        position_state = self._get_position_state(symbol)

        # If no open position, check for entry signal
        if position_state is None or position_state == "flat":
            signal = strategy.evaluate_entry(df)
            if signal in ("long", "short"):
                self._send_signal(symbol, signal, df, strategy)

        # If we have an open position, check for exit
        elif position_state in ("long", "short"):
            side = position_state
            should_exit = strategy.evaluate_exit(df, side)
            if should_exit:
                self._send_signal(symbol, "exit", df, strategy)

    def _get_position_state(self, symbol: str) -> str | None:
        """Query the position manager for the current state of a symbol.

        Uses a sync bus request — in production would use a proper
        request/response RPC pattern."""
        # The position manager publishes events; for state, we check
        # the latest position event from the bus. Since services share
        # Redis, we can peek at the latest position event.
        state = self.bus._r.get(f"aegis:state:position:{symbol}")
        if state:
            import json
            try:
                data = json.loads(state)
                return data.get("side")
            except json.JSONDecodeError:
                pass
        return None  # flat or unknown

    def _send_signal(self, symbol: str, signal: str, df: pd.DataFrame, strategy: Any) -> None:
        """Publish a signal event to the message bus."""
        latest = df.iloc[-1]
        signal_event = {
            "symbol": symbol,
            "strategy": self.strategy_name,
            "signal": signal,
            "confidence": self._calculate_confidence(strategy, df),
            "price": float(latest["close"]),
            "timestamp": latest.get("timestamp", pd.Timestamp.now().isoformat()),
            "atr": float(self._calculate_atr(df)),
            "strategy_params": {},
        }
        self.bus.publish(f"signal.{symbol}", signal_event)
        logger.debug("[%s] Signal: %s %s @ %.2f", self.strategy_name, symbol, signal, float(latest["close"]))

    def _calculate_confidence(self, strategy: Any, df: pd.DataFrame) -> float:
        """Score signal confidence (0.0-1.0) based on market filters."""
        score = 0.5  # base confidence

        # ATR regime filter check
        if len(df) >= 50:
            atr = self._calculate_atr(df.tail(20))
            price = float(df.iloc[-1]["close"])
            atr_pct = atr / price
            bounds = config.ATR_PCT_BOUNDS_BY_STRATEGY.get(self.strategy_name, (0.0005, 0.10))
            low_b, high_b = bounds
            if low_b <= atr_pct <= high_b:
                score += 0.2
            else:
                score -= 0.1

        # Trend alignment filter (if enabled)
        if config.ENABLE_TREND_ALIGNMENT_FILTER and self.strategy_name in config.TREND_ALIGNMENT_BY_STRATEGY:
            if config.TREND_ALIGNMENT_BY_STRATEGY[self.strategy_name] and len(df) >= 200:
                ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                ema200 = df["close"].ewm(span=200, adjust=False).mean().iloc[-1]
                if ema50 > ema200:
                    score += 0.15
                else:
                    score -= 0.15

        return max(0.0, min(1.0, score))

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Simple ATR calculation."""
        if len(df) < period:
            return float(df["close"].pct_change().abs().mean() * float(df.iloc[-1]["close"]))
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def on_tick(self) -> None:
        # Periodically update position state in Redis for other services
        pass

    def on_stop(self) -> None:
        logger.info("[%s] Stopped", self.strategy_name)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    strategy_name = sys.argv[1] if len(sys.argv) > 1 else "mean_reversion"
    StrategyWorker(strategy_name=strategy_name).run()
