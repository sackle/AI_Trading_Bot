from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from .sentiment import SentimentResult
from .regime import RegimeResult, MarketRegime
from .scanner import ScanResult
from .macro import MacroResult
import os
import json
import threading

try:
    import xgboost as xgb
    import numpy as np
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class MarketSignal:
    symbol: str
    direction: str
    confidence: int
    components: dict[str, float]
    regime: str
    timestamp: datetime


class SignalAggregator:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {
            'sentiment': 0.25,
            'regime': 0.30,
            'technical': 0.30,
            'macro': 0.15
        }

    def aggregate(
        self,
        symbol: str,
        sentiment: SentimentResult,
        regime: RegimeResult,
        scan: ScanResult,
        macro: MacroResult
    ) -> MarketSignal:
        
        sent_score = sentiment.score
        
        regime_score = 0.0
        if regime.regime == MarketRegime.TRENDING:
            if scan.macd_signal == "bullish" or scan.macd_histogram > 0:
                regime_score = 0.5
            elif scan.macd_signal == "bearish" or scan.macd_histogram < 0:
                regime_score = -0.5
        elif regime.regime == MarketRegime.MEAN_REVERTING:
            if scan.rsi_signal == "oversold":
                regime_score = 0.8
            elif scan.rsi_signal == "overbought":
                regime_score = -0.8
                
        tech_score = 0.0
        if scan.rsi_signal == "oversold" or scan.macd_signal == "bullish":
            tech_score += 0.5
        elif scan.rsi_signal == "overbought" or scan.macd_signal == "bearish":
            tech_score -= 0.5
        if scan.volume_signal == "high":
            tech_score *= 1.5
        tech_score = max(min(tech_score, 1.0), -1.0)
        
        macro_score = 0.0
        if macro.macro_environment == "RISK_ON":
            macro_score = 0.5
        elif macro.macro_environment == "RISK_OFF":
            macro_score = -0.5
            
        components = {
            'sentiment': sent_score,
            'regime': regime_score,
            'technical': tech_score,
            'macro': macro_score
        }
        
        total_score = sum(components[k] * self.weights.get(k, 0) for k in components)
        
        direction = "neutral"
        if total_score > 0.2:
            direction = "bullish"
        elif total_score < -0.2:
            direction = "bearish"
            
        confidence = int(min(abs(total_score) * 100, 100))
        
        return MarketSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            components=components,
            regime=regime.regime.value if hasattr(regime.regime, 'value') else str(regime.regime),
            timestamp=datetime.now()
        )

    def record_outcome(self, symbol: str, actual_return: float) -> None:
        """No-op on the weighted aggregator; overridden by XGBoostAggregator."""
        return

class XGBoostAggregator(SignalAggregator):
    def __init__(self, weights: dict[str, float] | None = None, min_samples: int = 100):
        super().__init__(weights)
        self.min_samples = min_samples
        self._training_buffer = []
        self._recent_signals = {}
        self.model = None
        self.lock = threading.Lock()
        self.model_path = os.path.join("runtime", "xgb_aggregator_model.json")
        self._load_model()
        
    def _load_model(self):
        if not HAS_XGBOOST:
            return
        with self.lock:
            if os.path.exists(self.model_path):
                try:
                    self.model = xgb.XGBClassifier()
                    self.model.load_model(self.model_path)
                except Exception:
                    self.model = None

    def _save_model(self):
        if not HAS_XGBOOST or self.model is None:
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)

    def aggregate(
        self,
        symbol: str,
        sentiment: SentimentResult,
        regime: RegimeResult,
        scan: ScanResult,
        macro: MacroResult
    ) -> MarketSignal:
        base_signal = super().aggregate(symbol, sentiment, regime, scan, macro)
        
        features = [
            base_signal.components['sentiment'],
            base_signal.components['regime'],
            base_signal.components['technical'],
            base_signal.components['macro']
        ]
        
        with self.lock:
            self._recent_signals[symbol] = features

            if not HAS_XGBOOST or self.model is None:
                return base_signal

            X = np.array([features])
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            
            direction = "bullish" if pred == 1 else "bearish"
            confidence = int(max(proba) * 100)
            
            return MarketSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                components=base_signal.components,
                regime=base_signal.regime,
                timestamp=base_signal.timestamp
            )

    def record_outcome(self, symbol: str, actual_return: float):
        with self.lock:
            if symbol not in self._recent_signals:
                return
            
            features = self._recent_signals.pop(symbol)
            label = 1 if actual_return > 0 else 0
            self._training_buffer.append((features, label))
            
            if len(self._training_buffer) >= self.min_samples and HAS_XGBOOST:
                self._train_model()
                self._training_buffer.clear()
                
    def _train_model(self):
        X = np.array([item[0] for item in self._training_buffer])
        y = np.array([item[1] for item in self._training_buffer])
        
        # Check if we have both classes, if not skip training to avoid errors
        if len(np.unique(y)) < 2:
            return
            
        if self.model is None:
            self.model = xgb.XGBClassifier(eval_metric='logloss')
            
        self.model.fit(X, y)
        self._save_model()

def create_aggregator(mode: str = "auto") -> SignalAggregator:
    """Factory: 'weighted' for simple, 'xgboost' for ML, 'auto' tries XGBoost first."""
    mode = (mode or "auto").strip().lower()
    if mode == "weighted":
        return SignalAggregator()
    if mode in ("xgboost", "auto"):
        return XGBoostAggregator() if HAS_XGBOOST else SignalAggregator()
    return SignalAggregator()
