from __future__ import annotations

import os
import logging
import pandas as pd
from typing import Any

import config
from bot.memory import TradingMemory
from .sentiment import SentimentEngine, SentimentResult
from .regime import RegimeDetector
from .scanner import MarketScanner
from .macro import MacroMonitor
from .aggregator import SignalAggregator, MarketSignal, create_aggregator


class MarketAnalystBot:
    def __init__(
        self,
        broker: Any = None,
        config_overrides: dict | None = None,
        memory: TradingMemory | None = None,
    ):
        self.broker = broker
        self.memory = memory or TradingMemory(config.runtime_path("aegis_memory.db"))
        self.sentiment_engine = SentimentEngine()
        self.regime_detector = RegimeDetector()
        self.scanner = MarketScanner()
        self.macro_monitor = MacroMonitor()
        self.aggregator = create_aggregator(
            mode=getattr(config, "ANALYST_AGGREGATOR_MODE", "auto")
        )
        self.log_file = getattr(
            config,
            "ANALYST_SIGNALS_LOG_PATH",
            os.path.join(getattr(config, "RUNTIME_DIR", "runtime"), "analyst_signals.csv"),
        )
        if not os.path.isabs(self.log_file):
            self.log_file = config.runtime_path(self.log_file) if hasattr(config, "runtime_path") else self.log_file

        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def analyze_all(self, symbols: list[str], bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, MarketSignal]:
        results = {}
        try:
            sentiments = self.sentiment_engine.analyze(symbols)
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {e}")
            sentiments = {}
            
        try:
            macro = self.macro_monitor.check()
        except Exception as e:
            logging.error(f"Macro monitor failed: {e}")
            macro = self.macro_monitor._neutral_result(pd.Timestamp.now())
            
        for symbol in symbols:
            bars = bars_by_symbol.get(symbol, pd.DataFrame())
            
            try:
                regime = self.regime_detector.detect(symbol, bars)
            except Exception as e:
                logging.error(f"Regime detection failed for {symbol}: {e}")
                regime = self.regime_detector._neutral_result(pd.Timestamp.now())

            # The intelligence dashboard reads regime confidence from the
            # shared memory database. Persist only valid detector output; a
            # neutral fallback has confidence 0 and should remain unknown.
            if regime.confidence > 0:
                try:
                    probabilities = regime.state_probabilities or {}
                    self.memory.record_regime(
                        getattr(regime.regime, "value", str(regime.regime)),
                        float(regime.confidence),
                        {
                            "TRENDING": probabilities.get("TRENDING", 0.0),
                            "MEAN_REVERTING": probabilities.get("MEAN_REVERTING", 0.0),
                            "CRISIS": probabilities.get(
                                "CRISIS", probabilities.get("HIGH_VOLATILITY", 0.0)
                            ),
                        },
                    )
                except Exception:
                    logging.exception("Failed to persist regime confidence for %s", symbol)
                
            try:
                scan = self.scanner.scan(symbol, bars)
            except Exception as e:
                logging.error(f"Scanner failed for {symbol}: {e}")
                scan = self.scanner.scan(symbol, pd.DataFrame())
                
            sentiment = sentiments.get(symbol, SentimentResult(symbol, 0.0, 0.0, 0, [], pd.Timestamp.now()))
            
            signal = self.aggregator.aggregate(symbol, sentiment, regime, scan, macro)
            results[symbol] = signal
            self._log_signal(signal)
            
        return results

    def analyze_symbol(self, symbol: str, bars: pd.DataFrame) -> MarketSignal:
        res = self.analyze_all([symbol], {symbol: bars})
        return res.get(symbol)

    def _log_signal(self, signal: MarketSignal):
        if not signal:
            return
            
        try:
            row = pd.DataFrame([{
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'regime': signal.regime,
                'sentiment_score': signal.components.get('sentiment', 0.0),
                'regime_score': signal.components.get('regime', 0.0),
                'technical_score': signal.components.get('technical', 0.0),
                'macro_score': signal.components.get('macro', 0.0)
            }])
            
            if not os.path.exists(self.log_file):
                row.to_csv(self.log_file, index=False)
            else:
                row.to_csv(self.log_file, mode='a', header=False, index=False)
        except Exception as e:
            logging.error(f"Failed to log signal to {self.log_file}: {e}")
