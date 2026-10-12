"""Analyst Worker service — publishes market regime and sentiment signals.

Receives: bars.* events (to trigger analysis)
Publishes: analyst.signal.{symbol}
           analyst.regime (market regime changes)
           analyst.sentiment.{symbol} (sentiment scores)

This service replaces the monolithic MarketAnalystBot in the distributed
architecture. It runs faster due to:
- 60s sentiment cache TTL (was 300s)
- 5min macro cache TTL (was 15min)
- 10min HMM refit (was 24h)
- 15-second minimum confidence threshold (was 20)

Analyst services are decoupled: if sentiment fails, regime still runs.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from bot.distributed.base import BaseService
import config

logger = logging.getLogger("aegis.analyst")


class AnalystWorker(BaseService):
    name = "analyst_worker"
    heartbeat_interval = 15  # 15s analyst cycle (was 30s)

    def __init__(self, bus=None):
        super().__init__(bus)
        self.bar_cache: dict[str, list[dict]] = {}
        self.max_cache = 500
        self.last_regime: str | None = None
        self.last_sentiment: dict[str, float] = {}

        # Initialize sub-engines lazily
        self._sentiment_engine = None
        self._macro_monitor = None
        self._regime_detector = None

    def register_handlers(self) -> None:
        self.bus.subscribe("bars.*", self._on_bar)

    def on_start(self) -> None:
        # Pre-warm caches with a broad market scan
        logger.info("Analyst Worker started (15s cycle, 10min refit)")

    def _on_bar(self, channel: str, msg: dict) -> None:
        """Cache incoming bars for regime/sentiment analysis."""
        symbol = msg.get("symbol", "")
        if not symbol:
            return

        cache = self.bar_cache.setdefault(symbol, [])
        cache.append(msg)
        if len(cache) > self.max_cache:
            cache.pop(0)

    def on_tick(self) -> None:
        """Run analyst cycle: regime → sentiment → publish signals."""
        results = []

        # 1. Regime analysis (if we have SPY data)
        regime = self._detect_regime()
        if regime != self.last_regime:
            self.last_regime = regime
            self.bus.publish("analyst.regime", {
                "regime": regime,
                "timestamp": pd.Timestamp.now().isoformat(),
                "confidence": self._regime_confidence(),
            })

        # 2. Sentiment analysis for cached symbols
        for symbol in list(self.bar_cache.keys()):
            sentiment = self._analyze_sentiment(symbol)
            if sentiment is not None:
                self.last_sentiment[symbol] = sentiment
                confidence = self._sentiment_confidence(symbol, sentiment)
                if confidence >= config.ANALYST_MIN_CONFIDENCE_FOR_ENTRY:
                    self.bus.publish(f"analyst.sentiment.{symbol}", {
                        "symbol": symbol,
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "regime": regime,
                        "timestamp": pd.Timestamp.now().isoformat(),
                    })

        # 3. Clear stale bar caches (>500 bars)
        for symbol in list(self.bar_cache.keys()):
            if len(self.bar_cache[symbol]) > self.max_cache:
                self.bar_cache[symbol] = self.bar_cache[symbol][-self.max_cache:]

    def _detect_regime(self) -> str:
        """Detect market regime using HMM on SPY bars."""
        spy_bars = self.bar_cache.get("SPY", [])
        if len(spy_bars) < 20:
            return "unknown"

        try:
            if self._regime_detector is None:
                from bot.analyst.regime import RegimeDetector
                self._regime_detector = RegimeDetector()
            return self._regime_detector.detect(spy_bars)
        except Exception as exc:
            logger.debug("Regime detection error: %s", exc)
            return "unknown"

    def _regime_confidence(self) -> float:
        return 0.8 if self.last_regime in ("bull", "strong_bull") else 0.6

    def _analyze_sentiment(self, symbol: str) -> float | None:
        """Analyze sentiment for a symbol. Returns score in [-1, 1]."""
        bars = self.bar_cache.get(symbol, [])
        if len(bars) < 5:
            return None

        try:
            if self._sentiment_engine is None:
                from bot.analyst.sentiment import SentimentEngine
                self._sentiment_engine = SentimentEngine()
            return self._sentiment_engine.analyze(symbol, bars)
        except Exception:
            # Fallback: simple price momentum sentiment
            closes = [b["close"] for b in bars[-10:]]
            if len(closes) < 5:
                return None
            recent_change = (closes[-1] - closes[0]) / closes[0]
            return max(-1.0, min(1.0, recent_change * 10)) if recent_change != 0 else None

    def _sentiment_confidence(self, symbol: str, sentiment: float) -> float:
        """Convert sentiment magnitude to confidence score."""
        return min(1.0, abs(sentiment) * 10 + 0.3)

    def on_stop(self) -> None:
        logger.info("Analyst Worker stopped")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    AnalystWorker().run()
